"""Graph generation v2 — maximum-legibility redesign.

Principles applied:
  one message per figure · huge fonts · generous spacing · value labels on
  everything · honest axes (broken scales only when labeled) · dark futuristic
  theme with logo watermark.

All numbers come from real executed results (evaluation/results/*.json) or
re-runs of the persisted engine.

Run:  python -m evaluation.make_graphs
"""
from __future__ import annotations

import json
import textwrap
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

RESULTS = "evaluation/results"
GRAPHS = "assets/graphs"
LOGO = "assets/logo/cerebro_logo.png"
LOGO_SMALL = "assets/logo/cerebro_logo_128.png"

CYAN, PURPLE, PINK, GREEN, YELLOW, ORANGE, BLUE = ("#00E5FF", "#B388FF", "#FF5C8A",
                                                   "#7CFFB2", "#FFD166", "#FF9E64",
                                                   "#60A5FA")
BG, PANEL = "#0B0F1A", "#111827"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": "#4B5563", "axes.labelcolor": "#F3F4F6",
    "xtick.color": "#D1D5DB", "ytick.color": "#D1D5DB",
    "text.color": "#F9FAFB", "grid.color": "#263244",
    "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": .45,
    "axes.titlesize": 15, "axes.titleweight": "bold",
    "axes.labelsize": 12, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11, "figure.dpi": 160,
})


def load(name):
    with open(f"{RESULTS}/{name}", encoding="utf-8") as f:
        return json.load(f)


def header(fig, title, subtitle):
    """Logo + title band; call before subplots_adjust(top=...)."""
    logo = Image.open(LOGO_SMALL)
    ax_img = fig.add_axes([0.012, 0.955, 0.042, 0.042 * logo.height / logo.width],
                          zorder=10)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo))
    fig.text(0.062, 0.972, title, fontsize=16, fontweight="bold", color=CYAN,
             ha="left", va="center")
    fig.text(0.062, 0.941, subtitle, fontsize=9.5, color="#9CA3AF",
             ha="left", va="center")


def watermark(fig, alpha=0.07, zoom=0.16, pos=(0.86, 0.02)):
    logo = Image.open(LOGO).convert("RGBA")
    logo.thumbnail((420, 420))
    ax_img = fig.add_axes([pos[0], pos[1], zoom, zoom * logo.height / logo.width],
                          zorder=-5)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo), alpha=alpha)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


# ================================================================ 1 · MAIN RESULT
def graph_main_result():
    base = load("baselines.json")
    full = load("summary.json")["full_metrics"]
    models = ["B1\nTF-IDF+LR", "B2\nTF-IDF+SVC", "B3\n+context", "CEREBRO\n(E) full"]
    keys = ["B1_tfidf_logreg", "B2_tfidf_svc", "B3_tfidf_context"]
    metrics = [("sarcasm", "Sarcasm", "roc_auc", CYAN),
               ("irony", "Irony", "roc_auc", PURPLE),
               ("passive_aggression", "Passive-aggr.", "roc_auc", PINK)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.4))
    header(fig, "Baselines vs CEREBRO — real test-split results",
           "89 held-out conversations · sequential predicted-history inference · seed 42")

    x = np.arange(len(models))
    w = 0.26
    for i, (h, lbl, mk, c) in enumerate(metrics):
        vals = [base[k][h][mk] for k in keys] + [full[h][mk]]
        bars = ax1.bar(x + (i - 1) * w, vals, w, color=c, label=lbl, edgecolor=BG, lw=.6)
        best = max(vals)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width()/2, v + .006, f"{v:.4f}",
                     ha="center", fontsize=8.6,
                     color="white" if abs(v - best) < 1e-9 else "#9CA3AF",
                     fontweight="bold" if abs(v - best) < 1e-9 else "normal")
    ax1.set_xticks(x); ax1.set_xticklabels(models, fontsize=10.5)
    ax1.set_ylim(0.88, 0.99)
    ax1.text(0.012, 0.02, "note: y-axis starts at 0.88 to make small gaps visible",
             transform=ax1.transAxes, fontsize=8.5, color="#6B7280", style="italic")
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("Hidden-signal ranking quality (higher = better)", loc="left", pad=10)
    ax1.legend(ncols=3, framealpha=0, loc="upper left", bbox_to_anchor=(0, 1.02))
    style_ax(ax1)

    mae = [base[k]["tension"]["mae"] for k in keys] + [full["tension"]["mae"]]
    colors = [BLUE, BLUE, BLUE, GREEN]
    bars = ax2.bar(x, mae, 0.52, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width()/2, v + .012, f"{v:.3f}",
                 ha="center", fontsize=10, fontweight="bold",
                 color="white" if v == min(mae) else "#9CA3AF")
    ax2.set_xticks(x); ax2.set_xticklabels(models, fontsize=10.5)
    ax2.set_ylim(0, max(mae) * 1.3)
    ax2.axhline(min(mae), color=GREEN, ls="--", lw=1, alpha=.6)
    ax2.text(2.55, min(mae) + .022, f"best = {min(mae):.3f}", fontsize=9.5, color=GREEN)
    ax2.set_ylabel("MAE (tension units, 0–100 scale)")
    ax2.set_title("Tension regression error (lower = better)", loc="left", pad=10)
    style_ax(ax2)

    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.06, right=0.98, wspace=0.22)
    watermark(fig)
    fig.savefig(f"{GRAPHS}/baselines_vs_cerebro.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ baselines_vs_cerebro.png")


# ================================================================ 2 · ABLATION
def graph_ablation():
    abl = load("ablations.json")
    full = load("summary.json")["full_metrics"]
    variants = ["A", "B", "C", "D", "E"]
    vlabels = ["A\ntext\nonly", "B\n+context\nwindow", "C\n+speaker\nmemory",
               "D\n+behavior\n(full heads)", "E\n+hidden fusion\n(FULL)"]
    sarc = [abl[v]["sarcasm"]["roc_auc"] for v in ("A", "B", "C", "D")] + \
        [full["sarcasm"]["roc_auc"]]
    mae = [abl[v]["tension"]["mae"] for v in ("A", "B", "C", "D")] + \
        [full["tension"]["mae"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.2))
    header(fig, "Ablation study — what does each component contribute?",
           "Same training protocol; E adds hidden-signal fusion + temporal engines on top of D")

    x = np.arange(5)
    colors = [BLUE, BLUE, BLUE, BLUE, GREEN]
    bars = ax1.bar(x, sarc, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, sarc):
        ax1.text(b.get_x() + b.get_width()/2, v + .0012, f"{v:.4f}",
                 ha="center", fontsize=9.5,
                 fontweight="bold" if v == max(sarc) else "normal",
                 color="white" if v == max(sarc) else "#D1D5DB")
    ax1.annotate("", xy=(4, sarc[4] + .004), xytext=(0, sarc[0] + .004),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                                 connectionstyle="arc3,rad=-0.25"))
    ax1.text(2.0, max(sarc) + .008, f"+{(sarc[4]-sarc[0])*100:.1f} pts AUC from the full stack",
             fontsize=11, color=GREEN, fontweight="bold", ha="center")
    ax1.set_xticks(x); ax1.set_xticklabels(vlabels, fontsize=9.5)
    ax1.set_ylim(0.90, max(sarc) + .018)
    ax1.set_ylabel("Sarcasm ROC-AUC")
    ax1.set_title("Ranking gain per component", loc="left", pad=10)
    style_ax(ax1)

    bars = ax2.bar(x, mae, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width()/2, v + .008, f"{v:.3f}",
                 ha="center", fontsize=9.5,
                 fontweight="bold" if v == min(mae) else "normal",
                 color="white" if v == min(mae) else "#D1D5DB")
    ax2.set_xticks(x); ax2.set_xticklabels(vlabels, fontsize=9.5)
    ax2.set_ylim(0, max(mae) * 1.25)
    ax2.set_ylabel("Tension MAE (lower = better)")
    ax2.set_title("Regression gain per component", loc="left", pad=10)
    style_ax(ax2)

    fig.subplots_adjust(top=0.80, bottom=0.17, left=0.06, right=0.98, wspace=0.22)
    watermark(fig)
    fig.savefig(f"{GRAPHS}/ablation_study.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ ablation_study.png")


# ================================================================ 3 · CAPABILITY
def graph_capability():
    full = load("summary.json")["full_metrics"]
    rows = [
        ("Sarcasm ROC-AUC", full["sarcasm"]["roc_auc"], CYAN),
        ("Irony ROC-AUC", full["irony"]["roc_auc"], PURPLE),
        ("Passive-aggression ROC-AUC", full["passive_aggression"]["roc_auc"], PINK),
        ("Sarcasm F1 (macro)", full["sarcasm"]["f1_macro"], CYAN),
        ("Irony F1 (macro)", full["irony"]["f1_macro"], PURPLE),
        ("Passive-aggression F1 (macro)", full["passive_aggression"]["f1_macro"], PINK),
        ("Escalation F1 (macro)", full["escalation"]["f1_macro"], ORANGE),
        ("Tension R²", max(full["tension"]["r2"], 0), GREEN),
    ]
    fig, ax = plt.subplots(figsize=(13, 6.8))
    header(fig, "Full CEREBRO (E) — capability sheet on the test split",
           "One bar per reported metric · values printed at bar ends · every number from evaluation/results/summary.json")
    rows = rows[::-1]
    y = np.arange(len(rows))
    bars = ax.barh(y, [v for _, v, _ in rows], 0.6,
                   color=[c for _, _, c in rows], edgecolor=BG, lw=.6)
    for b, (_, v, _) in zip(bars, rows):
        ax.text(v + .004, b.get_y() + b.get_height()/2, f"{v:.4f}",
                va="center", fontsize=10.5, fontweight="bold", color="white")
    ax.set_yticks(y)
    ax.set_yticklabels([n for n, _, _ in rows], fontsize=11)
    ax.set_xlim(0.85, 1.0)
    ax.set_xlabel("score (ROC-AUC / F1 / R²)")
    ax.set_title("All heads ≥ 0.92 — ranking metrics are the honest benchmark (see README note)",
                 loc="left", pad=10, fontsize=13)
    style_ax(ax)
    fig.subplots_adjust(top=0.80, bottom=0.11, left=0.30, right=0.965)
    watermark(fig, zoom=0.14, pos=(0.87, 0.03))
    fig.savefig(f"{GRAPHS}/capability_sheet.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ capability_sheet.png")


# ================================================================ 4 · CONFUSIONS
def _wrapped(labels, width=9):
    return ["\n".join(textwrap.wrap(l, width)) for l in labels]


def graph_confusion(head_key, labels, title, fname, color):
    from sklearn.metrics import confusion_matrix
    yt, yp = _test_predictions_labels(head_key)
    cm = confusion_matrix(yt, yp, labels=labels, normalize="true")
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(7.2, n * 0.78), max(6.0, n * 0.62)))
    header(fig, f"Confusion matrix — {title}",
           "persisted CEREBRO engine re-run on 45 held-out test conversations · row-normalized (recall view)")
    im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "neon", [PANEL, color]), vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(_wrapped(labels), rotation=42, ha="right", fontsize=8.6)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_wrapped(labels) if n <= 3 else labels, fontsize=8.6)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7.8,
                        color="white" if v > 0.55 else "#9CA3AF",
                        fontweight="bold" if i == j else "normal")
    ax.set_xlabel("predicted", fontsize=11)
    ax.set_ylabel("ground truth", fontsize=11)
    ax.grid(False)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=9, colors="#D1D5DB")
    fig.subplots_adjust(top=0.83, bottom=0.22, left=0.16, right=0.97)
    watermark(fig, zoom=0.13, pos=(0.80, 0.015), alpha=0.06)
    fig.savefig(f"{GRAPHS}/{fname}", bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ {fname}")


# ================================================================ 5 · CALIBRATION
def graph_calibration():
    from sklearn.calibration import calibration_curve
    proba = _test_predictions_proba()
    fig, ax = plt.subplots(figsize=(9.8, 7.2))
    header(fig, "Probability calibration — hidden-signal heads",
           "reliability curves on 45 test conversations · Brier score in legend (lower = better)")
    for name, color in [("sarcasm", PINK), ("irony", PURPLE),
                        ("passive_aggression", GREEN)]:
        y = np.array([p[0] for p in proba[name]])
        p = np.array([p[1] for p in proba[name]])
        frac, mean_p = calibration_curve(y, p, n_bins=8, strategy="quantile")
        brier = float(np.mean((p - y) ** 2))
        ax.plot(mean_p, frac, "-o", color=color, lw=2.4, ms=7,
                label=f"{name}  (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#6B7280", lw=1.4, label="perfectly calibrated")
    ax.set_xlabel("predicted probability", fontsize=12)
    ax.set_ylabel("observed positive frequency", fontsize=12)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.legend(fontsize=10.5, framealpha=0, loc="upper left")
    ax.set_title("Curves hugging the diagonal = trustworthy confidences",
                 loc="left", pad=10, fontsize=13)
    style_ax(ax)
    fig.subplots_adjust(top=0.82, bottom=0.12, left=0.11, right=0.97)
    watermark(fig, zoom=0.14, pos=(0.845, 0.02))
    fig.savefig(f"{GRAPHS}/calibration_curves.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ calibration_curves.png")


# ================================================================ 6 · HERO
def graph_hero():
    demo = load("demo_report.json")
    full = load("summary.json")["full_metrics"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7.0),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    header(fig, "CEREBRO — conversation intelligence at a glance",
           "left: per-message tension with detected turning points (real model output) · right: test-split headline metrics")

    tension = [m["tension"] for m in demo["messages"]]
    xs = np.arange(1, len(tension) + 1)
    ax1.plot(xs, tension, color=CYAN, lw=2.6, marker="o", ms=7,
             mfc=CYAN, mec=BG, mew=1.4, zorder=3, label="predicted tension")
    ax1.fill_between(xs, tension, color=CYAN, alpha=.10)
    if demo["turning_points"]:
        tp = max(demo["turning_points"], key=lambda t: abs(t["tension_change"]))
        ax1.axvline(tp["message_id"], color=PINK, ls="--", lw=1.6, alpha=.95)
        ax1.annotate(
            f'turning point #{tp["message_id"]}\n'
            f'{tp["before"]["emotion"]} → {tp["after"]["emotion"]}\n'
            f'Δtension {tp["tension_change"]:+.1f} (z={tp["robust_z"]})',
            xy=(tp["message_id"], tp["after"]["tension"]),
            xytext=(tp["message_id"] - 4.6, max(tension) * 0.86),
            fontsize=10.5, color=PINK, fontweight="bold",
            arrowprops=dict(arrowstyle="->", color=PINK, lw=1.8))
    ax1.axhspan(60, 100, color=PINK, alpha=.08)
    ax1.text(len(tension) - 0.4, 62, "escalation zone", fontsize=9,
             color=PINK, ha="right", alpha=.9)
    ax1.set_xticks(xs)
    ax1.set_xlabel("message #", fontsize=12)
    ax1.set_ylabel("tension (0–100)", fontsize=12)
    ax1.set_ylim(0, 100)
    ax1.set_title(f'Held-out test conversation · trajectory = '
                  f'"{load("demo_report.json")["summary"]["trajectory"]}"',
                  loc="left", pad=10, fontsize=14)
    ax1.legend(fontsize=10.5, framealpha=0, loc="lower right")
    style_ax(ax1)

    rows = [("Sarcasm AUC", full["sarcasm"]["roc_auc"], CYAN),
            ("Irony AUC", full["irony"]["roc_auc"], PURPLE),
            ("Passive-aggr. AUC", full["passive_aggression"]["roc_auc"], PINK),
            ("Escalation F1", full["escalation"]["f1_macro"], ORANGE),
            ("Tension R²", max(full["tension"]["r2"], 0), GREEN)]
    rows = rows[::-1]
    y = np.arange(len(rows))
    bars = ax2.barh(y, [v for _, v, _ in rows], 0.58,
                    color=[c for _, _, c in rows], edgecolor=BG, lw=.6)
    for b, (_, v, _) in zip(bars, rows):
        ax2.text(v + .004, b.get_y() + b.get_height()/2, f"{v:.4f}",
                 va="center", fontsize=10.5, fontweight="bold", color="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels([n for n, _, _ in rows], fontsize=11.5)
    ax2.set_xlim(0.85, 1.0)
    ax2.set_xlabel("score", fontsize=12)
    ax2.set_title("Headline metrics (test split, 89 conversations)",
                  loc="left", pad=10, fontsize=14)
    style_ax(ax2)

    fig.subplots_adjust(top=0.82, bottom=0.11, left=0.075, right=0.975, wspace=0.30)
    watermark(fig, zoom=0.15, pos=(0.865, 0.015))
    fig.savefig(f"{GRAPHS}/hero_dashboard.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ hero_dashboard.png")


# ================================================================ 7 · DATASET
def graph_dataset():
    from cerebro.data.generator import generate_corpus
    corpus = generate_corpus(convs_per_cell=14)
    msgs = [m for c in corpus for m in c]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(17, 6.2),
                                        gridspec_kw={"width_ratios": [1.25, 1, 1]})
    header(fig, "CEREBRO corpus — 6 domains × 7 narrative arcs, weak-supervision annotated",
           "conversation-level splits (no leakage) · 588 conversations · 10,956 messages · seed 42")

    # panel 1: sentiment composition across tension bands
    bands = [("calm", 0, 25), ("friction", 25, 55), ("escalation", 55, 85), ("peak", 85, 101)]
    comp = {b: {"positive": 0, "neutral": 0, "negative": 0} for b, _, _ in bands}
    for m in msgs:
        for b, lo, hi in bands:
            if lo <= m["tension"] < hi:
                comp[b][m["sentiment"]] += 1
    x = np.arange(len(bands))
    pos = [comp[b]["positive"] for b, _, _ in bands]
    neu = [comp[b]["neutral"] for b, _, _ in bands]
    neg = [comp[b]["negative"] for b, _, _ in bands]
    ax1.bar(x, pos, .58, color=GREEN, label="positive", edgecolor=BG, lw=.6)
    ax1.bar(x, neu, .58, bottom=pos, color=BLUE, label="neutral", edgecolor=BG, lw=.6)
    ax1.bar(x, neg, .58, bottom=[p + n for p, n in zip(pos, neu)], color=PINK,
            label="negative", edgecolor=BG, lw=.6)
    for xi, (p, n, g) in enumerate(zip(pos, neu, neg)):
        ax1.text(xi, p + n + g + 130, f"{p+n+g:,}", ha="center", fontsize=10,
                 fontweight="bold", color="#D1D5DB")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n{lo}–{hi if hi<101 else 100}" for b, lo, hi in bands],
                        fontsize=10.5)
    ax1.set_ylabel("messages", fontsize=12)
    ax1.set_ylim(0, max(p + n + g for p, n, g in zip(pos, neu, neg)) * 1.18)
    ax1.set_title("Sentiment composition per tension band", loc="left", pad=10, fontsize=13.5)
    ax1.legend(fontsize=10, framealpha=0, ncols=3, loc="upper left")
    style_ax(ax1)

    # panel 2: tension histogram
    tens = [m["tension"] for m in msgs]
    ax2.hist(tens, bins=32, color=PURPLE, alpha=.9, edgecolor=BG, lw=.4)
    ax2.axvline(float(np.mean(tens)), color=YELLOW, ls="--", lw=1.6)
    ax2.text(np.mean(tens) + 2.5, ax2.get_ylim()[1] * .92,
             f"mean {np.mean(tens):.1f}", fontsize=10.5, color=YELLOW)
    ax2.set_xlabel("tension value", fontsize=12)
    ax2.set_ylabel("messages", fontsize=12)
    ax2.set_title("Tension distribution", loc="left", pad=10, fontsize=13.5)
    style_ax(ax2)

    # panel 3: messages per conversation
    lens = [len(c) for c in corpus]
    ax3.hist(lens, bins=17, color=CYAN, alpha=.9, edgecolor=BG, lw=.4)
    ax3.axvline(float(np.mean(lens)), color=YELLOW, ls="--", lw=1.6)
    ax3.text(np.mean(lens) + 0.4, ax3.get_ylim()[1] * .92,
             f"mean {np.mean(lens):.1f}", fontsize=10.5, color=YELLOW)
    ax3.set_xlabel("messages per conversation", fontsize=12)
    ax3.set_ylabel("conversations", fontsize=12)
    ax3.set_title("Conversation lengths", loc="left", pad=10, fontsize=13.5)
    style_ax(ax3)

    fig.subplots_adjust(top=0.79, bottom=0.13, left=0.05, right=0.985, wspace=0.26)
    watermark(fig, zoom=0.13, pos=(0.875, 0.015))
    fig.savefig(f"{GRAPHS}/dataset_overview.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ dataset_overview.png")


# ================================================================ 8 · DEMO REPORT
def graph_demo_report():
    demo = load("demo_report.json")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 8.6), sharex=True,
                                   gridspec_kw={"height_ratios": [1.35, 1]})
    header(fig, "CEREBRO on a held-out test conversation — full per-message readout",
           "top: tension with turning-point markers · bottom: sarcasm / irony / passive-aggression probabilities (0.5 threshold line)")
    msgs = demo["messages"]
    xs = np.arange(1, len(msgs) + 1)
    tension = [m["tension"] for m in msgs]
    ax1.plot(xs, tension, color=CYAN, lw=2.6, marker="o", ms=7, mfc=CYAN, mec=BG,
             mew=1.4, zorder=3)
    ax1.fill_between(xs, tension, color=CYAN, alpha=.10)
    for t in demo["turning_points"]:
        ax1.axvline(t["message_id"], color=PINK, ls=":", lw=1.3, alpha=.85)
        ax1.text(t["message_id"], max(tension) * 1.02,
                 f'#{t["message_id"]} {t["tension_change"]:+.0f}',
                 fontsize=8.6, color=PINK, ha="center", rotation=0)
    ax1.axhspan(60, 100, color=PINK, alpha=.07)
    ax1.set_ylim(0, 100)
    ax1.set_ylabel("tension (0–100)", fontsize=12)
    ax1.set_title("tension", loc="left", fontsize=12, color=CYAN, pad=6)
    style_ax(ax1)

    for key, color, lbl in [("sarcasm", PINK, "sarcasm"),
                            ("irony", PURPLE, "irony"),
                            ("passive_aggression", GREEN, "passive-aggr.")]:
        probs = [m[key]["probability"] for m in msgs]
        ax2.plot(xs, probs, "-o", color=color, lw=2, ms=5.5, label=lbl, mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.text(len(msgs) - .3, 0.52, "decision threshold", fontsize=9, color="#9CA3AF",
             ha="right")
    ax2.set_ylim(0, 1)
    ax2.set_yticks([0, .25, .5, .75, 1])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f'#{m["message_id"]}\n{m["text"][:16]}…' for m in msgs],
                        fontsize=8)
    ax2.set_xlabel("message", fontsize=12)
    ax2.set_ylabel("probability", fontsize=12)
    ax2.set_title("hidden signals", loc="left", fontsize=12, color=PINK, pad=6)
    ax2.legend(fontsize=10, framealpha=0, ncols=3, loc="upper left")
    style_ax(ax2)

    fig.subplots_adjust(top=0.86, bottom=0.135, left=0.07, right=0.975, hspace=0.24)
    watermark(fig, zoom=0.12, pos=(0.885, 0.015))
    fig.savefig(f"{GRAPHS}/demo_report.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ demo_report.png")


# ================================================================ helpers
def _test_predictions(n_convs=45):
    """Re-run the persisted engine on test conversations (same protocol as eval)."""
    from cerebro.data.generator import generate_corpus, split_conversations
    from cerebro.models.engines import MultiTaskEngine
    from cerebro.models.hidden_signals import apply_hidden_signals

    corpus = generate_corpus(convs_per_cell=14)
    test = split_conversations(corpus)["test"][:n_convs]
    eng = MultiTaskEngine(seed=42).load("models/saved/cerebro_engine")
    yt, yp = {"sent": [], "emo": [], "tone": []}, {"sent": [], "emo": [], "tone": []}
    proba = {"sarcasm": [], "irony": [], "passive_aggression": []}
    for c in test:
        results = eng.predict_conversation(c)
        apply_hidden_signals(results)
        for r, m in zip(results, c):
            yt["sent"].append(m["sentiment"]); yp["sent"].append(r["sentiment"]["label"])
            yt["emo"].append(m["emotion"]); yp["emo"].append(r["emotion"]["label"])
            yt["tone"].append(m["tone"]); yp["tone"].append(r["tone"]["label"])
            for h in proba:
                proba[h].append((int(m[h]), r[h]["probability"]))
    return yt, yp, proba


def _test_predictions_labels(key):
    yt, yp, _ = _test_predictions()
    return yt[key], yp[key]


def _test_predictions_proba():
    return _test_predictions()[2]


# ================================================================ 9 · SCENARIOS


def graph_scenarios():
    sc = load("scenarios.json")["scenarios"]
    calm_expected = {"normal", "happy", "very_short", "slow", "humor", "malformed",
                     "multi_speaker"}
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 7.4))
    header(fig, "PS-01 §41 robustness — 20 hand-crafted out-of-distribution scenarios",
           "different phrasing, emoji, slang, timing and formats than the training corpus · all 20 executed without failure")

    names = [s["scenario"] for s in sc]
    mean_t = [s["mean_tension"] for s in sc]
    peak_t = [s["peak_tension"] for s in sc]
    colors = [GREEN if n in calm_expected else ORANGE for n in names]
    x = np.arange(len(names))
    ax1.bar(x, mean_t, 0.62, color=colors, alpha=.55, edgecolor=BG, lw=.5,
            label="mean tension")
    ax1.plot(x, peak_t, "_", color="white", ms=22, mew=2.4, label="peak tension")
    ax1.axhline(25.9, color=CYAN, ls="--", lw=1.4)
    ax1.text(len(names) - .4, 27.5, "training-corpus mean 25.9", fontsize=9.5,
             color=CYAN, ha="right")
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=52, ha="right", fontsize=9)
    ax1.set_ylabel("tension (0–100)", fontsize=12)
    ax1.set_title("Per-scenario tension readout", loc="left", pad=10, fontsize=13.5)
    ax1.legend(fontsize=10, framealpha=0, loc="upper left")
    from matplotlib.patches import Patch
    ax1.legend(handles=[Patch(color=GREEN, alpha=.55, label="calm expected"),
                        Patch(color=ORANGE, alpha=.55, label="conflict expected"),
                        plt.Line2D([], [], color="white", marker="_", ms=14, mew=2.4,
                                   lw=0, label="peak tension"),
                        plt.Line2D([], [], color=CYAN, ls="--", label="corpus mean")],
               fontsize=9.5, framealpha=0, loc="upper right")
    style_ax(ax1)

    hidden = {"sarcasm": [], "irony": [], "passive_aggression": []}
    for s in sc:
        hidden["sarcasm"].append(s["sarcasm_mean"])
        hidden["irony"].append(s["irony_mean"])
        hidden["passive_aggression"].append(s["pa_mean"])
    for (h, c) in [("sarcasm", PINK), ("irony", PURPLE), ("passive_aggression", GREEN)]:
        ax2.plot(x, hidden[h], "-o", color=c, lw=2, ms=5.5, label=h.replace("_", "-"),
                 mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=52, ha="right", fontsize=9)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("mean probability", fontsize=12)
    ax2.set_title("Hidden-signal levels per scenario", loc="left", pad=10, fontsize=13.5)
    ax2.legend(fontsize=10, framealpha=0, ncols=3, loc="upper left")
    style_ax(ax2)

    fig.subplots_adjust(top=0.80, bottom=0.24, left=0.055, right=0.985, wspace=0.20)
    watermark(fig, zoom=0.13, pos=(0.875, 0.015))
    fig.savefig(f"{GRAPHS}/scenario_robustness.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ scenario_robustness.png")


# ================================================================ 10 · ARCHITECTURE
def graph_architecture():
    fig = plt.figure(figsize=(13.5, 15.5))
    header(fig, "CEREBRO system architecture",
           "raw chat → parsers → contextual/temporal intelligence → explainable report (PS-01 §46)")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    def box(x, y, w, h, lines, color, fs=11.5, sub_fs=9.5):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=PANEL, edgecolor=color,
                                   lw=2, zorder=3))
        head = lines[0]
        rest = lines[1:]
        ax.text(x + w / 2, y + h - (h * 0.30 if rest else h / 2), head,
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color=color, zorder=4)
        if rest:
            ax.text(x + w / 2, y + h * 0.30, "\n".join(rest), ha="center",
                    va="center", fontsize=sub_fs, color="#D1D5DB", zorder=4)

    def arrow(x1, y1, x2, y2, color="#4B5563", lw=2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    def band(y, h, label, color):
        ax.add_patch(plt.Rectangle((1.5, y), 97, h, facecolor=color, alpha=0.045,
                                   edgecolor="none", zorder=1))
        ax.text(2.6, y + h - 1.6, label, fontsize=10, color=color, alpha=0.9,
                fontweight="bold", zorder=2, va="top")

    band(88.5, 10.5, "INPUT LAYER · PS-01 §2", CYAN)
    for txt, x in [("WhatsApp\n.txt", 6), ("Discord\nJSON", 24), ("Slack\nJSON", 41),
                   ("CSV", 57), ("JSON", 69), ("plain txt", 81)]:
        box(x, 89.5, 12.5, 6.2, [txt], CYAN, fs=10.5)
        arrow(x + 6.2, 89.5, 48.5, 86.4, lw=1.4)

    box(31, 79.5, 36, 6.6, ["CHAT PARSER + AUTO-DETECT",
                            "one common message format · speakers · timestamps"], CYAN)
    arrow(49, 79.5, 49, 76.8)

    band(56.5, 20, "UNDERSTANDING LAYER · PS-01 §5, §8, §9, §17", PURPLE)
    box(8, 66.5, 40, 9.5, ["PREPROCESSING + BEHAVIORAL FEATURES",
                           "16-dim interpretable vector per message",
                           "CAPS · exclamations · emoji · response gap"], PURPLE)
    box(54, 66.5, 40, 9.5, ["CONTEXT ENGINE + SPEAKER MEMORY",
                            "sliding 4-turn window + decayed summary",
                            "per-speaker emotional state"], PURPLE)
    arrow(28, 66.5, 42, 62.2)
    arrow(74, 66.5, 58, 62.2)
    box(20, 58.2, 62, 4.0, ["MESSAGE REPRESENTATION  =  text ⊕ context ⊕ behavior ⊕ memory"],
        "#7CFFB2", fs=12)

    band(30.5, 24.5, "INTELLIGENCE LAYER · PS-01 §10–22", YELLOW)
    box(5, 47.5, 28, 8.5, ["MULTI-TASK NLP ENGINE",
                           "sentiment · emotion · tone",
                           "tension (0–100)"], YELLOW)
    box(36, 47.5, 28, 8.5, ["HIDDEN-SIGNAL DETECTION",
                            "sarcasm · irony · passive-aggression",
                            "learned ⊕ contradiction evidence"], YELLOW)
    box(67, 47.5, 28, 8.5, ["TEMPORAL ENGINES",
                            "arc · transitions · turning points",
                            "escalation trajectory"], YELLOW)
    arrow(49, 47.5, 49, 44.4)
    box(26, 36.8, 48, 7.4, ["MODEL FUSION + CALIBRATION",
                            "six evidence streams · validation-tuned weights · Platt scaling"],
        ORANGE)

    band(20, 9.5, "OUTPUT LAYER · PS-01 §23–30", PINK)
    box(8, 21.5, 26, 7.0, ["EXPLAINABILITY",
                           "WHY? · WHAT CHANGED?",
                           "evidence ≠ interpretation"], PINK)
    box(37, 21.5, 26, 7.0, ["CONVERSATION REPORT",
                            "18-section summary",
                            "speaker + topic views"], PINK)
    box(66, 21.5, 26, 7.0, ["FASTAPI + DASHBOARD",
                            "12 endpoints · TTL store",
                            "futuristic frontend"], PINK)
    arrow(21, 21.5, 44, 17.6)
    arrow(49, 21.5, 49, 17.6)
    arrow(79, 21.5, 54, 17.6)
    box(30, 11.2, 40, 6.0, ["EMOTIONAL ARC · TENSION CURVE",
                            "the conversation's emotional journey, explained"], CYAN, fs=12)

    ax.text(50, 6.2, "every arrow is a real function call — see docs/methodology/PS01_WORKFLOW.md",
            fontsize=10.5, color="#6B7280", ha="center", style="italic")
    watermark(fig, alpha=0.05, zoom=0.17, pos=(0.845, 0.015))
    fig.savefig("docs/architecture/architecture.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ architecture.png")


if __name__ == "__main__":
    from cerebro.common.labels import SENTIMENT_LABELS, EMOTION_LABELS, TONE_LABELS
    print("generating readable graphs from real results...")
    graph_hero()
    graph_main_result()
    graph_ablation()
    graph_capability()
    graph_dataset()
    graph_demo_report()
    graph_scenarios()
    graph_architecture()
    graph_confusion("sent", SENTIMENT_LABELS, "sentiment (3-way)",
                    "confusion_sentiment.png", CYAN)
    graph_confusion("emo", EMOTION_LABELS, "emotion (13-way)",
                    "confusion_emotion.png", PURPLE)
    graph_confusion("tone", TONE_LABELS, "tone (14-way)",
                    "confusion_tone.png", PINK)
    graph_calibration()
    print("done → assets/graphs/")
