import logging
from json import JSONDecodeError
from dataclasses import dataclass
from typing import List, Tuple, Any

from goodai.helpers.json_helper import sanitize_and_parse_json
from dataset_interfaces.interface import DatasetInterface, TestExample

logger = logging.getLogger(__name__)

VALID_DOMAINS = {"hotel", "restaurant", "taxi", "attraction", "train"}


def _load_hf_dataset(split: str = "test"):
    """Load MultiWOZ v2.2 from HuggingFace.

    Because this module lives inside a local ``datasets/`` package that shadows
    the HuggingFace ``datasets`` library, a plain ``from datasets import
    load_dataset`` resolves to the wrong package.  We work around this by
    temporarily hiding the local package from ``sys.modules`` and ``sys.path``
    while importing the external one.
    """
    import importlib
    import pathlib
    import sys

    # If the HuggingFace module is already loaded, reuse it.
    hf = sys.modules.get("datasets")
    if hf is not None and hasattr(hf, "load_dataset"):
        return hf.load_dataset("multi_woz_v22", split=split, trust_remote_code=True)

    # Stash the local ``datasets`` package so importlib finds the HF one.
    project_root = str(pathlib.Path(__file__).resolve().parent.parent)
    saved_path = sys.path[:]
    saved_modules = {}
    for key in list(sys.modules):
        if key == "datasets" or key.startswith("datasets."):
            saved_modules[key] = sys.modules.pop(key)

    sys.path = [p for p in sys.path if p != project_root and p != ""]

    try:
        hf_datasets = importlib.import_module("datasets")
        if not hasattr(hf_datasets, "load_dataset"):
            raise ImportError
    except ImportError:
        raise ImportError(
            "The HuggingFace `datasets` library is required for the MultiWOZ "
            "dataset.  Install it with:  pip install datasets"
        )
    finally:
        # Restore the original path and local modules so the rest of the
        # application keeps working.
        sys.path = saved_path
        sys.modules.update(saved_modules)

    return hf_datasets.load_dataset(
        "multi_woz_v22", split=split, trust_remote_code=True,
    )


def _extract_ground_truth(dialogue: dict) -> dict[str, dict[str, str]]:
    """Build the accumulated ground-truth dialogue state from all user turns.

    Returns a dict keyed by domain, each value being a dict of slot -> value.
    """
    turns = dialogue["turns"]
    state: dict[str, dict[str, str]] = {}

    for i in range(len(turns["utterance"])):
        # Only user turns (speaker==0) carry annotated state in MultiWOZ v2.2.
        if turns["speaker"][i] != 0:
            continue
        frames = turns["frames"][i]
        for j in range(len(frames["service"])):
            svc = frames["service"][j]
            turn_state = frames["state"][j]
            slot_names = turn_state["slots_values"]["slots_values_name"]
            slot_values = turn_state["slots_values"]["slots_values_list"]
            if not slot_names:
                continue
            if svc not in state:
                state[svc] = {}
            for name, vals in zip(slot_names, slot_values):
                # Strip the "domain-" prefix (e.g. "hotel-name" -> "name").
                short = name.split("-", 1)[-1] if "-" in name else name
                state[svc][short] = vals[0] if len(vals) == 1 else vals[0]
    return state


def _build_script(dialogue: dict) -> List[str]:
    """Convert a MultiWOZ dialogue into a list of script messages."""
    turns = dialogue["turns"]
    script: list[str] = [
        "I'm going to show you a conversation between a user and a booking assistant. "
        "Pay close attention to every detail mentioned, because afterwards I will ask "
        "you to recall specific information from the dialogue."
    ]
    for i in range(len(turns["utterance"])):
        speaker = "User" if turns["speaker"][i] == 0 else "Assistant"
        script.append(f'{speaker} said: "{turns["utterance"][i]}"')
    return script


def _build_question(ground_truth: dict[str, dict[str, str]]) -> str:
    """Build the final recall question."""
    domains = sorted(ground_truth.keys())
    domain_list = ", ".join(domains)
    slot_hint_parts = []
    for d in domains:
        slots = sorted(ground_truth[d].keys())
        slot_hint_parts.append(f'"{d}": {{{", ".join(repr(s) + ": ..." for s in slots)}}}')
    slot_hint = "{\n  " + ",\n  ".join(slot_hint_parts) + "\n}"

    return (
        f"The conversation you just read covered these domains: {domain_list}.\n"
        "Based on what was discussed, recall all the slot values that were established "
        "for each domain. Respond ONLY with a JSON object in this exact structure "
        "(fill in the values from the conversation):\n"
        f"{slot_hint}"
    )


def _normalize(value: str) -> str:
    """Lower-case, strip, and collapse whitespace for fuzzy comparison."""
    return " ".join(value.lower().strip().split())


@dataclass
class MultiWOZDataset(DatasetInterface):
    name: str = "MultiWOZ"
    description: str = (
        "Present a multi-domain booking dialogue (hotel, restaurant, taxi, attraction, "
        "train) and ask the agent to recall the dialogue state — slot values for each "
        "domain — after the conversation ends."
    )
    reset_message: str = (
        "Forget the conversation I just showed you and all the booking details."
    )
    min_domains: int = 2
    max_domains: int = 3

    def generate_examples(self, num_examples: int) -> List[TestExample]:
        dataset = _load_hf_dataset("test")

        # Collect eligible dialogues: those whose services span the desired
        # number of *valid* domains and that produce non-empty ground truth.
        candidates: list[dict] = []
        for dialogue in dataset:
            services = [s for s in dialogue["services"] if s in VALID_DOMAINS]
            if not (self.min_domains <= len(services) <= self.max_domains):
                continue
            gt = _extract_ground_truth(dialogue)
            if len(gt) < self.min_domains:
                continue
            candidates.append({"dialogue": dialogue, "ground_truth": gt})

        self.random.shuffle(candidates)

        examples: list[TestExample] = []
        for idx, cand in enumerate(candidates[:num_examples]):
            dialogue = cand["dialogue"]
            ground_truth = cand["ground_truth"]

            script = _build_script(dialogue)
            question = _build_question(ground_truth)
            script.append(question)

            is_question = [False] * (len(script) - 1) + [True]

            example = TestExample(
                dataset_generator=self,
                script=script,
                expected_responses=[ground_truth],
                is_question=is_question,
                example_id=dialogue.get("dialogue_id", str(idx)),
            )
            examples.append(example)

        return examples

    def evaluate_correct(
        self,
        questions: List[str],
        responses: List[str],
        expected_answers: List[Any],
    ) -> Tuple[int, int, List[str]]:
        """Compare recalled slots against ground truth.

        Scoring:
        - Each ground-truth slot is worth 1 point.
        - A slot scores 1 if the predicted value matches (case-insensitive,
          whitespace-normalized).
        """
        ground_truth: dict[str, dict[str, str]] = expected_answers[0]

        total_slots = sum(len(slots) for slots in ground_truth.values())
        if total_slots == 0:
            return 0, 0, ["No ground-truth slots to evaluate."]

        # Parse the agent's JSON response.
        try:
            predicted = sanitize_and_parse_json(responses[0])
        except (ValueError, JSONDecodeError) as exc:
            msg = f"Could not parse agent response as JSON: {exc}"
            logger.warning(msg)
            return 0, total_slots, [msg]

        if not isinstance(predicted, dict):
            msg = f"Expected a JSON object, got {type(predicted).__name__}."
            return 0, total_slots, [msg]

        correct = 0
        reasoning: list[str] = []

        for domain, slots in ground_truth.items():
            pred_domain = predicted.get(domain, {})
            if not isinstance(pred_domain, dict):
                reasoning.append(f"{domain}: expected dict, got {type(pred_domain).__name__}")
                continue
            for slot, expected_val in slots.items():
                pred_val = pred_domain.get(slot)
                if pred_val is None:
                    reasoning.append(f'{domain}/{slot}: missing (expected "{expected_val}")')
                    continue
                if _normalize(str(pred_val)) == _normalize(str(expected_val)):
                    correct += 1
                else:
                    reasoning.append(
                        f'{domain}/{slot}: got "{pred_val}", expected "{expected_val}"'
                    )

        if correct == total_slots:
            reasoning.insert(0, f"All {total_slots} slots matched.")
        else:
            reasoning.insert(0, f"{correct}/{total_slots} slots matched.")

        return correct, total_slots, reasoning
