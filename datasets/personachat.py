import json
import logging
from json import JSONDecodeError
from dataclasses import dataclass
from typing import List, Tuple, Any

from goodai.helpers.json_helper import sanitize_and_parse_json

from dataset_interfaces.interface import DatasetInterface, TestExample
from utils.llm import make_system_message, make_user_message, GPT_4_TURBO_BEST

try:
    from datasets import load_dataset as hf_load_dataset
except ImportError:
    hf_load_dataset = None

_EVAL_SYSTEM_PROMPT = """
You are evaluating whether a set of recalled persona facts matches the original facts.
For each original fact, determine if the recalled list contains a statement that
captures the same meaning (it does not need to be verbatim).

Respond in JSON with this format:
{
  "matches": [
    {"original": "<fact>", "matched": true/false, "closest_recall": "<closest recalled text or null>"}
  ]
}
""".strip()


@dataclass
class PersonaChatDataset(DatasetInterface):
    name: str = "PersonaChat"
    description: str = (
        "Present persona facts as conversation statements, add dialogue filler turns, "
        "then quiz the agent on the persona facts it should remember."
    )
    reset_message: str = "Please forget all persona facts you have learned about me."
    num_persona_facts: int = 4
    num_filler_turns: int = 6

    def __post_init__(self):
        if hf_load_dataset is None:
            raise ImportError(
                "The HuggingFace `datasets` library is required for the PersonaChat dataset. "
                "Install it with: pip install datasets"
            )
        self._hf_data = hf_load_dataset("bavard/personachat_truecased", split="train")

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        examples: List[TestExample] = []
        indices = list(range(len(self._hf_data)))
        self.random.shuffle(indices)

        generated = 0
        idx_cursor = 0

        while generated < num_examples and idx_cursor < len(indices):
            # Collect persona facts from one conversation entry
            entry = self._hf_data[indices[idx_cursor]]
            persona_lines: List[str] = list(entry.get("personality", []))
            idx_cursor += 1

            if len(persona_lines) < self.num_persona_facts:
                continue

            selected_facts = self.random.sample(persona_lines, self.num_persona_facts)

            # Gather filler dialogue turns from neighbouring entries
            filler_utterances = self._collect_filler_turns(indices, idx_cursor)

            script: List[str] = []
            is_question: List[bool] = []

            # Present persona facts as natural statements
            intro = (
                "Let me tell you a few things about myself. Please remember these facts:\n"
                + "\n".join(f"- {fact.strip()}" for fact in selected_facts)
            )
            script.append(intro)
            is_question.append(False)

            # Insert filler dialogue turns
            for utterance in filler_utterances[:self.num_filler_turns]:
                script.append(utterance)
                is_question.append(False)

            # Ask the agent to recall the persona facts
            question = (
                "Earlier in our conversation I told you several facts about myself. "
                "Can you recall all of them? Please respond as a JSON list of strings, "
                'for example: ["fact 1", "fact 2", ...]'
            )
            script.append(question)
            is_question.append(True)

            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=[selected_facts],
                is_question=is_question,
            )
            examples.append(example)
            generated += 1

        return examples

    def _collect_filler_turns(self, indices: List[int], cursor: int) -> List[str]:
        """Gather dialogue utterances from nearby dataset entries for use as filler."""
        filler: List[str] = []
        search_range = min(cursor + 20, len(indices))
        for i in range(cursor, search_range):
            entry = self._hf_data[indices[i]]
            utterances = entry.get("history", [])
            if not utterances:
                # Fall back to 'candidates' if 'history' is empty
                utterances = entry.get("candidates", [])
            for utt in utterances:
                text = utt.strip()
                if text:
                    filler.append(text)
                if len(filler) >= self.num_filler_turns:
                    return filler
        return filler

    def evaluate_correct(
        self, questions: List[str], responses: List[str], expected_answers: List[Any]
    ) -> Tuple[int, int, List[str]]:
        original_facts: List[str] = expected_answers[0]
        max_score = len(original_facts)
        score = 0
        reasoning: List[str] = []

        # Try to parse the agent's response as a JSON list
        recalled_facts: List[str] = []
        try:
            parsed = sanitize_and_parse_json(responses[0])
            if isinstance(parsed, list):
                recalled_facts = [str(f) for f in parsed]
            elif isinstance(parsed, dict) and "facts" in parsed:
                recalled_facts = [str(f) for f in parsed["facts"]]
            else:
                recalled_facts = [str(parsed)]
        except (JSONDecodeError, ValueError):
            # Fall back: treat the entire response as free text
            recalled_facts = [responses[0]]

        # Use LLM-based evaluation to match recalled facts against originals
        eval_context = [
            make_system_message(_EVAL_SYSTEM_PROMPT),
            make_user_message(json.dumps({
                "original_facts": original_facts,
                "recalled_facts": recalled_facts,
            })),
        ]

        try:
            eval_response = self.ask_llm(eval_context, model=GPT_4_TURBO_BEST, max_tokens=512)
            eval_result = sanitize_and_parse_json(eval_response)
            matches = eval_result.get("matches", [])
            for match in matches:
                if match.get("matched", False):
                    score += 1
                    reasoning.append(
                        f"Matched: \"{match['original']}\" -> \"{match.get('closest_recall', '?')}\""
                    )
                else:
                    reasoning.append(f"Not recalled: \"{match['original']}\"")
        except (JSONDecodeError, ValueError, KeyError, TypeError) as exc:
            msg = f"LLM evaluation failed ({repr(exc)}), falling back to string matching."
            logging.warning(msg)
            reasoning.append(msg)
            score, reasoning = self._fallback_string_match(original_facts, recalled_facts, reasoning)

        return score, max_score, reasoning

    @staticmethod
    def _fallback_string_match(
        original_facts: List[str], recalled_facts: List[str], reasoning: List[str]
    ) -> Tuple[int, List[str]]:
        """Simple substring matching as a fallback when LLM evaluation fails."""
        score = 0
        recalled_lower = " ".join(recalled_facts).lower()
        for fact in original_facts:
            # Check if key words from the fact appear in the recalled text
            words = [w for w in fact.lower().split() if len(w) > 3]
            matched_words = sum(1 for w in words if w in recalled_lower)
            if words and matched_words / len(words) >= 0.5:
                score += 1
                reasoning.append(f"Partial string match for: \"{fact}\"")
            else:
                reasoning.append(f"No string match for: \"{fact}\"")
        return score, reasoning
