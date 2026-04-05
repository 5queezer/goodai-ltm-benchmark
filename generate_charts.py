"""Generate radar + bar charts for each model's baseline vs dream comparison."""
import json
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

BASE = "data/tests/MuninnDB Benchmark Full - 1k/results"
DATASETS = ["Colours", "Locations Directions", "NameList", "Shopping", "Spy Meeting", "Trigger Response"]
LABELS_SHORT = ["Colours", "Locations\nDirections", "NameList", "Shopping", "Spy\nMeeting", "Trigger\nResponse"]

MODELS = [
    ("Nemotron", "Nemotron Baseline", "Nemotron Dream", "nemotron-120b (free)"),
    ("GPT4oMini", "GPT4oMini Baseline", "GPT4oMini Dream", "gpt-4o-mini ($0.15/M)"),
    ("GLM4.5Air", "GLM4.5Air Baseline", "GLM4.5Air Dream", "GLM 4.5 Air (free)"),
    ("Gemma4-26B", "Gemma4-26B Baseline", "Gemma4-26B Dream", "Gemma 4 26B A4B ($0.13/M)"),
    ("Qwen3.6Plus", "Qwen3.6Plus Baseline", "Qwen3.6Plus Dream", "Qwen 3.6 Plus (free)"),
]


def get_scores(agent):
    scores = {}
    for ds in DATASETS:
        ds_dir = os.path.join(BASE, agent, ds)
        vals = []
        if os.path.isdir(ds_dir):
            for f in sorted(glob.glob(os.path.join(ds_dir, "*.json"))):
                with open(f) as fh:
                    d = json.load(fh)
                    vals.append(d.get("score", 0))
        scores[ds] = sum(vals) / len(vals) if vals else 0
    return scores


def make_chart(model_key, baseline_agent, dream_agent, model_label, outpath):
    baseline = get_scores(baseline_agent)
    dream = get_scores(dream_agent)

    b_vals = [baseline[ds] for ds in DATASETS]
    d_vals = [dream[ds] for ds in DATASETS]
    b_avg = np.mean(b_vals)
    d_avg = np.mean(d_vals)

    fig, (ax_radar, ax_bar) = plt.subplots(1, 2, figsize=(16, 7),
                                            subplot_kw={"projection": "polar"} if True else {})
    # Actually need mixed projections
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    ax_bar = axes[1]

    # Radar chart
    ax_radar = fig.add_subplot(121, polar=True)
    axes[0].set_visible(False)

    N = len(DATASETS)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    angles += angles[:1]

    b_plot = b_vals + b_vals[:1]
    d_plot = d_vals + d_vals[:1]

    ax_radar.set_theta_offset(np.pi / 2)
    ax_radar.set_theta_direction(-1)
    ax_radar.set_rlabel_position(30)
    ax_radar.set_ylim(0, 1.0)
    ax_radar.set_yticks([0.2, 0.4, 0.6, 0.8])

    ax_radar.plot(angles, b_plot, "o-", color="#e74c3c", linewidth=2, label="Baseline (no dream)")
    ax_radar.fill(angles, b_plot, alpha=0.15, color="#e74c3c")
    ax_radar.plot(angles, d_plot, "o-", color="#2ecc71", linewidth=2, label="With Dream Engine")
    ax_radar.fill(angles, d_plot, alpha=0.15, color="#2ecc71")

    ax_radar.set_xticks(angles[:-1])
    ax_radar.set_xticklabels(LABELS_SHORT, fontsize=10)
    ax_radar.legend(loc="upper right", bbox_to_anchor=(1.3, 1.15), fontsize=9)
    ax_radar.set_title("Per-Dataset Accuracy", fontsize=13, fontweight="bold", pad=20)

    # Bar chart
    x = np.arange(N)
    width = 0.35
    ax_bar.bar(x - width/2, b_vals, width, color="#e74c3c", alpha=0.75, label="Baseline")
    ax_bar.bar(x + width/2, d_vals, width, color="#2ecc71", alpha=0.75, label="Dream Engine")

    ax_bar.axhline(y=b_avg, color="#e74c3c", linestyle="--", alpha=0.5, linewidth=1)
    ax_bar.axhline(y=d_avg, color="#2ecc71", linestyle="--", alpha=0.5, linewidth=1)
    ax_bar.text(N - 0.5, b_avg + 0.01, f"avg {b_avg:.2f}", color="#e74c3c", fontsize=9, ha="right")
    ax_bar.text(N - 0.5, d_avg + 0.01, f"avg {d_avg:.2f}", color="#2ecc71", fontsize=9, ha="right")

    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels([ds for ds in DATASETS], rotation=30, ha="right", fontsize=10)
    ax_bar.set_ylabel("Score", fontsize=11)
    ax_bar.set_ylim(0, max(max(b_vals), max(d_vals)) * 1.25 + 0.05)
    ax_bar.legend(fontsize=9)
    ax_bar.set_title("Side-by-Side Comparison", fontsize=13, fontweight="bold")

    delta = d_avg - b_avg
    sign = "+" if delta >= 0 else ""
    fig.suptitle(
        f"MuninnDB Dream Engine — GoodAI LTM Benchmark\n"
        f"(1k memory span, {model_label}, gpt-4o-mini eval, vault-isolated)",
        fontsize=14, fontweight="bold", y=0.98,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    fig.savefig(outpath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {outpath}  (baseline={b_avg:.2f}, dream={d_avg:.2f}, delta={sign}{delta:.2f})")


# Generate per-model charts
os.makedirs("benchmark_charts", exist_ok=True)
for model_key, b_agent, d_agent, label in MODELS:
    outpath = f"benchmark_charts/{model_key.lower()}_comparison.png"
    make_chart(model_key, b_agent, d_agent, label, outpath)

# Also generate a summary multi-model bar chart
fig, ax = plt.subplots(figsize=(12, 6))
model_labels = []
baselines = []
dreams = []
for model_key, b_agent, d_agent, label in MODELS:
    b = get_scores(b_agent)
    d = get_scores(d_agent)
    baselines.append(np.mean([b[ds] for ds in DATASETS]))
    dreams.append(np.mean([d[ds] for ds in DATASETS]))
    model_labels.append(label.split(" (")[0])

x = np.arange(len(MODELS))
width = 0.35
bars_b = ax.bar(x - width/2, baselines, width, color="#e74c3c", alpha=0.75, label="Baseline")
bars_d = ax.bar(x + width/2, dreams, width, color="#2ecc71", alpha=0.75, label="Dream Engine")

for i, (b, d) in enumerate(zip(baselines, dreams)):
    delta = d - b
    sign = "+" if delta >= 0 else ""
    ax.text(i + width/2, d + 0.01, f"{sign}{delta:.2f}", ha="center", fontsize=9,
            color="#2ecc71" if delta >= 0 else "#e74c3c", fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(model_labels, fontsize=11)
ax.set_ylabel("Overall Score", fontsize=12)
ax.set_ylim(0, 0.5)
ax.legend(fontsize=10)
ax.set_title("MuninnDB Dream Engine — Model Comparison\n(GoodAI LTM Benchmark, 1k memory span, 6 datasets, vault-isolated)",
             fontsize=13, fontweight="bold")

plt.tight_layout()
fig.savefig("benchmark_charts/model_comparison.png", dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: benchmark_charts/model_comparison.png")

print("\nDone!")
