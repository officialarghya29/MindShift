"""GitHub social preview (1280×640) — real engine output, on-brand.

Renders the demo conversation's tension curve with its strongest turning
point, headline metrics, and the logo. Run:  python scripts/make_social_preview.py
"""
from __future__ import annotations

import json
import sys
import warnings

sys.path.insert(0, ".")
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

CYAN, PINK, GREEN, PURPLE = "#00E5FF", "#FF5C8A", "#7CFFB2", "#B388FF"
BG, PANEL = "#0B0F1A", "#111827"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "text.color": "#F9FAFB", "axes.edgecolor": "#4B5563",
    "xtick.color": "#D1D5DB", "ytick.color": "#D1D5DB",
    "grid.color": "#263244", "axes.grid": True, "grid.alpha": .4,
    "font.family": "DejaVu Sans",
})


def main() -> None:
    demo = json.load(open("evaluation/results/demo_report.json", encoding="utf-8"))
    summary = json.load(open("evaluation/results/summary.json", encoding="utf-8"))
    fm = summary["full_metrics"]
    bench = json.load(open("evaluation/results/benchmarks.json", encoding="utf-8"))

    fig = plt.figure(figsize=(12.8, 6.4), dpi=100)

    # ---- left: tension curve with the strongest turning point ----
    ax = fig.add_axes([0.055, 0.13, 0.52, 0.72])
    tension = [m["tension"] for m in demo["messages"]]
    xs = np.arange(1, len(tension) + 1)
    ax.plot(xs, tension, color=CYAN, lw=3, marker="o", ms=7, mfc=CYAN, mec=BG,
            mew=1.4, zorder=3)
    ax.fill_between(xs, tension, color=CYAN, alpha=.10)
    ax.axhspan(60, 100, color=PINK, alpha=.08)
    if demo["turning_points"]:
        tp = max(demo["turning_points"], key=lambda t: abs(t["tension_change"]))
        ax.axvline(tp["message_id"], color=PINK, ls="--", lw=1.8, ymax=0.84)
    ax.set_ylim(0, 100)
    ax.set_xlabel("message #", fontsize=13)
    ax.set_ylabel("tension (0–100)", fontsize=13)
    ax.tick_params(labelsize=11.5)
    ax.set_title("REAL OUTPUT — held-out conversation, detected turning points",
                 loc="left", fontsize=13.5, color=CYAN, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # ---- right: headline metrics ----
    ax2 = fig.add_axes([0.665, 0.13, 0.30, 0.72])
    rows = [("Sarcasm AUC", fm["sarcasm"]["roc_auc"], CYAN),
            ("Irony AUC", fm["irony"]["roc_auc"], PURPLE),
            ("Passive-aggr. AUC", fm["passive_aggression"]["roc_auc"], PINK),
            ("Tension R²", max(fm["tension"]["r2"], 0), GREEN)][::-1]
    y = np.arange(len(rows))
    ax2.barh(y, [v for _, v, _ in rows], 0.62,
             color=[c for _, _, c in rows], edgecolor=BG)
    for b, (_, v, _) in zip(ax2.barh(y, [v for _, v, _ in rows], 0.62,
                                     color=[c for _, _, c in rows],
                                     edgecolor=BG), rows):
        ax2.text(v + .004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                 va="center", fontsize=13.5, fontweight="bold", color="white")
    ax2.set_yticks(y, [n for n, _, _ in rows], fontsize=13.5)
    ax2.set_xlim(0.85, 1.005)
    ax2.set_xticks([])
    ax2.set_title("TEST-SPLIT METRICS (89 conversations)", loc="left",
                  fontsize=13.5, color=GREEN, fontweight="bold", pad=10)
    for s in ("top", "right"):
        ax2.spines[s].set_visible(False)
    ax2.grid(False)

    # ---- footer strip ----
    rate = bench["scaling"][-1]["messages_per_second"]
    fig.text(0.055, 0.035,
             f"≈{rate:.0f} msg/s flat to 1,000-message chats   ·   8.9 MB peak RAM   "
             f"·   39 tests · CI-gated   ·   officialarghya29/MindShift",
             fontsize=13, color="#9CA3AF")

    # ---- logo + title band ----
    logo = Image.open("assets/logo/cerebro_logo_128.png")
    ax_img = fig.add_axes([0.028, 0.80, 0.115, 0.115 * logo.height / logo.width],
                          zorder=10)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo))
    fig.text(0.155, 0.885, "CEREBRO — Conversation Intelligence",
             fontsize=27, fontweight="bold", color=CYAN, va="center")
    fig.text(0.155, 0.815,
             "sentiment · emotion · tone · sarcasm · tension · turning points — with the evidence",
             fontsize=15.5, color="#D1D5DB", va="center")

    fig.savefig("docs/social_preview.png")
    plt.close(fig)
    im = Image.open("docs/social_preview.png")
    print(f"docs/social_preview.png — {im.width}×{im.height}")


if __name__ == "__main__":
    main()
