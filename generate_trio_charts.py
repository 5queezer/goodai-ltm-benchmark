#!/usr/bin/env python3
"""Generate benchmark comparison charts: shared-vault vs vault-isolated, Gemini + GLM."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent / "data" / "charts"
OUT_DIR.mkdir(parents=True, exist_ok=True)

datasets = ["Colours", "NameList", "Shopping", "Trigger\nResponse", "Locations\nDirections", "Spy\nMeeting"]

# --- Vault-isolated results (correct methodology) ---
gemini_iso = {
    "Baseline": [0.500, 0.500, 0.433, 1.000, 0.500, 0.000],
    "Best (1,2,5)": [0.000, 0.500, 0.433, 1.000, 0.000, 0.000],
    "Full Phases": [0.000, 0.700, 0.544, 1.000, 0.000, 0.000],
}

# --- Shared-vault results (for comparison) ---
gemini_shared = {
    "Baseline": [0.500, 0.400, 0.433, 1.000, 0.000, 0.000],
    "Best (1,2,5)": [0.000, 0.500, 0.600, 1.000, 0.000, 0.000],
    "Full Phases": [0.000, 0.500, 0.489, 1.000, 0.000, 0.000],
}

glm_shared = {
    "Baseline": [0.000, 0.500, 0.167, 1.000, 0.000, 0.167],
    "Best (1,2,5)": [0.000, 0.400, 0.167, 1.000, 0.000, 0.000],
    "Full Phases": [0.000, 0.300, 0.000, 1.000, 0.000, 0.000],
}

composites_iso = {
    "Baseline": 0.489,
    "Full Phases": 0.374,
    "Best (1,2,5)": 0.322,
}

composites_all = {
    "Gemini iso\nBaseline": 0.489,
    "Gemini iso\nFull Phases": 0.374,
    "Gemini iso\nBest (1,2,5)": 0.322,
    "Gemini shared\nBaseline": 0.389,
    "Gemini shared\nFull Phases": 0.331,
    "Gemini shared\nBest (1,2,5)": 0.350,
    "GLM shared\nBaseline": 0.306,
    "GLM shared\nFull Phases": 0.217,
    "GLM shared\nBest (1,2,5)": 0.261,
}

# Ablation phase impact (from 50-trial Optuna study, shared-vault)
phase_names = ["Transitive\nInference", "Orient", "Semantic\nDedup", "Schema\nPromotion",
               "Dream\nJournal", "Relevance\nDecay", "LLM\nAdjudication", "Bidirectional\nStability"]
phase_deltas = [+0.022, +0.007, +0.006, +0.004, +0.003, -0.011, -0.011, -0.014]

C = {"baseline": "#4CAF50", "best": "#2196F3", "full": "#FF9800"}

# ── Chart 1: Vault-isolated per-dataset (main result) ──
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(datasets))
w = 0.25
ax.bar(x - w, gemini_iso["Baseline"], w, label="Baseline (no dream)", color=C["baseline"], edgecolor="white")
ax.bar(x,     gemini_iso["Full Phases"], w, label="Full Phases", color=C["full"], edgecolor="white")
ax.bar(x + w, gemini_iso["Best (1,2,5)"], w, label="Best (1,2,5)", color=C["best"], edgecolor="white")
ax.set_ylabel("Score", fontsize=12)
ax.set_title("Gemini 3.1 Flash Lite — Vault-Isolated", fontsize=13, fontweight="bold")
ax.set_xticks(x)
ax.set_xticklabels(datasets, fontsize=10)
ax.set_ylim(0, 1.15)
ax.legend(loc="upper right", fontsize=10)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bars in ax.containers:
    for bar in bars:
        h = bar.get_height()
        if h > 0:
            ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                    ha="center", va="bottom", fontsize=8)
plt.tight_layout()
fig.savefig(OUT_DIR / "gemini_vault_isolated.png", dpi=150)
plt.close()

# ── Chart 2: Shared vs Isolated side-by-side ──
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
for ax, data, title in [
    (ax1, gemini_iso, "Vault-Isolated (correct)"),
    (ax2, gemini_shared, "Shared Vault"),
]:
    ax.bar(x - w, data["Baseline"], w, label="Baseline", color=C["baseline"], edgecolor="white")
    ax.bar(x,     data["Full Phases"], w, label="Full Phases", color=C["full"], edgecolor="white")
    ax.bar(x + w, data["Best (1,2,5)"], w, label="Best (1,2,5)", color=C["best"], edgecolor="white")
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(datasets, fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for bars in ax.containers:
        for bar in bars:
            h = bar.get_height()
            if h > 0:
                ax.text(bar.get_x() + bar.get_width()/2, h + 0.02, f"{h:.2f}",
                        ha="center", va="bottom", fontsize=7)
ax1.set_ylabel("Score", fontsize=12)
ax1.legend(loc="upper left", fontsize=9)
plt.suptitle("Gemini — Vault Isolation Impact", fontsize=14, fontweight="bold", y=1.02)
plt.tight_layout()
fig.savefig(OUT_DIR / "isolation_comparison.png", dpi=150, bbox_inches="tight")
plt.close()

# ── Chart 3: Full composite overview ──
fig, ax = plt.subplots(figsize=(9, 5.5))
names = list(composites_all.keys())
values = list(composites_all.values())
color_map = []
for n in names:
    if "Baseline" in n: color_map.append(C["baseline"])
    elif "Best" in n: color_map.append(C["best"])
    else: color_map.append(C["full"])
bars = ax.barh(names, values, color=color_map, edgecolor="white", height=0.6)
ax.set_xlim(0, 0.6)
ax.set_xlabel("Composite Score", fontsize=12)
ax.set_title("All Runs — Composite Score", fontsize=13, fontweight="bold")
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
# Separator lines
ax.axhline(2.5, color="gray", linewidth=0.8, linestyle="--")
ax.axhline(5.5, color="gray", linewidth=0.8, linestyle="--")
for bar, val in zip(bars, values):
    ax.text(val + 0.006, bar.get_y() + bar.get_height()/2, f"{val:.3f}",
            va="center", fontsize=10, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "all_composites.png", dpi=150)
plt.close()

# ── Chart 4: Ablation phase impact ──
fig, ax = plt.subplots(figsize=(8, 4.5))
colors = ["#4CAF50" if d > 0.005 else "#F44336" if d < -0.005 else "#9E9E9E" for d in phase_deltas]
y = np.arange(len(phase_names))
bars = ax.barh(y, phase_deltas, color=colors, edgecolor="white", height=0.6)
ax.set_yticks(y)
ax.set_yticklabels(phase_names, fontsize=10)
ax.set_xlabel("Avg Score Delta (with − without)", fontsize=11)
ax.set_title("Dream Phase Impact — Bayesian Ablation (50 trials)", fontsize=13, fontweight="bold")
ax.axvline(0, color="black", linewidth=0.8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
for bar, val in zip(bars, phase_deltas):
    offset = 0.001 if val >= 0 else -0.001
    ha = "left" if val >= 0 else "right"
    ax.text(val + offset, bar.get_y() + bar.get_height()/2, f"{val:+.3f}",
            va="center", ha=ha, fontsize=9, fontweight="bold")
plt.tight_layout()
fig.savefig(OUT_DIR / "ablation_phase_impact.png", dpi=150)
plt.close()

# ── Chart 5: Radar vault-isolated ──
fig, ax = plt.subplots(figsize=(7, 6), subplot_kw=dict(polar=True))
radar_ds = ["Colours", "NameList", "Shopping", "Trigger\nResp.", "Locations\nDir.", "Spy\nMeeting"]
angles = np.linspace(0, 2 * np.pi, len(radar_ds), endpoint=False).tolist()
angles += angles[:1]
for label, color in [("Baseline", C["baseline"]), ("Full Phases", C["full"]), ("Best (1,2,5)", C["best"])]:
    vals = gemini_iso[label] + gemini_iso[label][:1]
    ax.plot(angles, vals, "o-", color=color, label=label, linewidth=2, markersize=5)
    ax.fill(angles, vals, alpha=0.08, color=color)
ax.set_xticks(angles[:-1])
ax.set_xticklabels(radar_ds, fontsize=9)
ax.set_ylim(0, 1.1)
ax.set_title("Gemini — Vault-Isolated Radar", fontsize=13, fontweight="bold", pad=20)
ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1), fontsize=10)
plt.tight_layout()
fig.savefig(OUT_DIR / "radar_vault_isolated.png", dpi=150, bbox_inches="tight")
plt.close()

print(f"Charts saved to {OUT_DIR}/")
for f in sorted(OUT_DIR.glob("*.png")):
    print(f"  {f.name}")
