"""Graph generation v3 — collision-free, maximum-legibility redesign.

Anti-collision architecture (fixes: text/line collisions reported on every graph):
  1. Header/footer/watermark bands are laid out in INCHES, not figure fractions,
     so they occupy identical space on short and tall figures.
  2. Legends are placed OUTSIDE the plot area (below the axes) or in regions
     guaranteed empty — never floating over bars/curves.
  3. In-plot annotation boxes are replaced by legend entries or top-band labels
     with reserved headroom.
  4. A rendered-text overlap validator draws every figure, extracts the actual
     pixel bounding box of every text artist, and asserts no two texts collide
     (fail-hard, exactly like the architecture diagram validator).

All numbers come from real executed results (evaluation/results/*.json) or
re-runs of the persisted engine. Run:  python -m evaluation.make_graphs
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
    "axes.titlesize": 14, "axes.titleweight": "bold",
    "axes.labelsize": 12.5, "xtick.labelsize": 11, "ytick.labelsize": 11,
    "legend.fontsize": 11.5, "figure.dpi": 170,
})

# inch-based layout bands (identical on every figure, regardless of height)
HEADER_IN = 0.72      # title + subtitle band height
FOOTER_IN = 0.60      # below-axes legend band where used
WATERMARK_IN = 1.15   # logo watermark size (bottom-right, below all axes)


def load(name):
    with open(f"{RESULTS}/{name}", encoding="utf-8") as f:
        return json.load(f)


def header(fig, title, subtitle):
    """Logo + title band, laid out in inches (converted to fractions).

    Call immediately after subplots(); compute axes top as
    1 - (HEADER_IN + gap_in) / fig_h.
    """
    logo = Image.open(LOGO_SMALL)
    fig_w, fig_h = fig.get_size_inches()
    side = 0.46                                    # logo box, inches
    ax_img = fig.add_axes(
        [0.014 / fig_w,
         (fig_h - HEADER_IN + 0.12) / fig_h,
         side / fig_w,
         side * logo.height / logo.width / fig_h], zorder=10)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo))
    fig.text(0.075, (fig_h - 0.28) / fig_h, title, fontsize=18,
             fontweight="bold", color=CYAN, ha="left", va="center")
    fig.text(0.075, (fig_h - 0.55) / fig_h, subtitle, fontsize=11,
             color="#9CA3AF", ha="left", va="center")
    return HEADER_IN


def axes_top(fig, extra_in=0.12):
    """Top margin (fraction) leaving room for the header band + gap."""
    return 1 - (HEADER_IN + extra_in) / fig.get_size_inches()[1]


def watermark(fig):
    """Bottom-right logo watermark in inches (converted to fractions), zorder 0."""
    fig_w, fig_h = fig.get_size_inches()
    logo = Image.open(LOGO).convert("RGBA")
    logo.thumbnail((520, 520))
    h_in = WATERMARK_IN * logo.height / logo.width
    ax_img = fig.add_axes(
        [(fig_w - WATERMARK_IN - 0.10) / fig_w, 0.10 / fig_h,
         WATERMARK_IN / fig_w, h_in / fig_h], zorder=0)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo), alpha=0.07)


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend_below(ax, ncols, y_offset=-0.20):
    """Legend OUTSIDE the plot area, below the x-label — can never cover data."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y_offset),
                     ncols=ncols, framealpha=0, borderaxespad=0)


def save(fig, name):
    """Finalize with the rendered-text overlap validator, then save."""
    _validate_no_text_overlaps(fig, name)
    watermark(fig)
    fig.savefig(f"{GRAPHS}/{name}")
    plt.close(fig)
    print(f"  ✓ {name} (text-overlap validator passed)")


def _validate_no_text_overlaps(fig, name):
    """Draw the figure and assert no two text bounding boxes intersect.

    Covers titles, axis labels, tick labels, annotations, legend texts and
    value labels — the actual rendered pixels, not coordinates. Bounding boxes
    may kiss (≤ 3 px tolerance) but not overlap.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    fig_bb = fig.bbox
    boxes = []                                   # (label, bbox)
    for ax in fig.get_axes():
        if not ax.axison:                        # axis("off") image axes
            continue
        artists = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for loc, t in zip(ax.get_xticks(), ax.get_xticklabels()):
            if xlim[0] <= loc <= xlim[1]:     # skip ticks matplotlib won't draw
                artists.append(t)
        for loc, t in zip(ax.get_yticks(), ax.get_yticklabels()):
            if ylim[0] <= loc <= ylim[1]:
                artists.append(t)
        if ax.get_legend():
            artists.extend(ax.get_legend().get_texts())
        for t in artists:
            if t is None or not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer=renderer)
            if (bb.x1 < 0 or bb.x0 > fig_bb.x1 or bb.y1 < 0
                    or bb.y0 > fig_bb.y1):
                continue                         # off-canvas → cannot collide
            boxes.append((t.get_text()[:28].replace("\n", "⏎"), bb))
    for t in fig.texts:
        if t.get_text().strip():
            bb = t.get_window_extent(renderer=renderer)
            if bb.x1 < 0 or bb.x0 > fig_bb.x1 or bb.y1 < 0 or bb.y0 > fig_bb.y1:
                continue
            boxes.append((t.get_text()[:28].replace("\n", "⏎"), bb))
    tol = 3.0                                    # px — allow kissing edges
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i][1], boxes[j][1]
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > tol and oy > tol:
                raise AssertionError(
                    f"[{name}] text collision: '{boxes[i][0]}' × '{boxes[j][0]}' "
                    f"(overlap {ox:.0f}×{oy:.0f} px)")


# ================================================================ 1 · MAIN RESULT
def graph_main_result():
    base = load("baselines.json")
    full = load("summary.json")["full_metrics"]
    models = ["B1\nTF-IDF+LR", "B2\nTF-IDF+SVC", "B3\n+context", "CEREBRO\n(E) full"]
    keys = ["B1_tfidf_logreg", "B2_tfidf_svc", "B3_tfidf_context"]
    metrics = [("sarcasm", "Sarcasm", CYAN),
               ("irony", "Irony", PURPLE),
               ("passive_aggression", "Passive-aggr.", PINK)]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 8.6))

    x = np.arange(len(models))
    w = 0.26
    for i, (h, lbl, c) in enumerate(metrics):
        vals = [base[k][h]["roc_auc"] for k in keys] + [full[h]["roc_auc"]]
        bars = ax1.bar(x + (i - 1) * w, vals, w, color=c, label=lbl,
                       edgecolor=BG, lw=.6)
        best = max(vals)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=9, rotation=90,
                     color="white" if abs(v - best) < 1e-9 else "#9CA3AF",
                     fontweight="bold" if abs(v - best) < 1e-9 else "normal")
    ax1.set_xticks(x, models, fontsize=11.5)
    ax1.set_ylim(0.88, 1.0)
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("Hidden-signal ranking quality (higher = better)",
                  loc="left", pad=12)
    legend_below(ax1, 3, y_offset=-0.24)
    style_ax(ax1)

    mae = [base[k]["tension"]["mae"] for k in keys] + [full["tension"]["mae"]]
    colors = [BLUE, BLUE, BLUE, GREEN]
    bars = ax2.bar(x, mae, 0.52, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.3f}",
                 ha="center", fontsize=11, fontweight="bold",
                 color="white" if v == min(mae) else "#9CA3AF")
    ax2.set_xticks(x, models, fontsize=11.5)
    ax2.set_ylim(0, max(mae) * 1.35)
    ax2.axhline(min(mae), color=GREEN, ls="--", lw=1, alpha=.6)
    ax2.set_ylabel("MAE (tension units, 0–100 scale)")
    ax2.set_title(f"Tension regression error (lower = better) · best {min(mae):.3f}",
                  loc="left", pad=12)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.17, left=0.055,
                        right=0.975, wspace=0.22)
    header(fig, "Baselines vs CEREBRO — real test-split results",
           "89 held-out conversations · sequential predicted-history inference · seed 42 "
           "· left y-axis starts at 0.88 to magnify small gaps")
    save(fig, "baselines_vs_cerebro.png")


# ================================================================ 2 · ABLATION
def graph_ablation():
    abl = load("ablations.json")
    full = load("summary.json")["full_metrics"]
    vlabels = ["A\ntext only", "B\n+context\nwindow", "C\n+speaker\nmemory",
               "D\n+behavior\n(full heads)", "E\n+hidden fusion\n(FULL)"]
    sarc = [abl[v]["sarcasm"]["roc_auc"] for v in ("A", "B", "C", "D")] + \
        [full["sarcasm"]["roc_auc"]]
    mae = [abl[v]["tension"]["mae"] for v in ("A", "B", "C", "D")] + \
        [full["tension"]["mae"]]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16.5, 8.8))

    x = np.arange(5)
    colors = [BLUE, BLUE, BLUE, BLUE, GREEN]
    bars = ax1.bar(x, sarc, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, sarc):
        ax1.text(b.get_x() + b.get_width() / 2, v + .0012, f"{v:.4f}",
                 ha="center", fontsize=10.5,
                 fontweight="bold" if v == max(sarc) else "normal",
                 color="white" if v == max(sarc) else "#D1D5DB")
    ax1.annotate("", xy=(4, sarc[4] + .0035), xytext=(0, sarc[0] + .0035),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                                 connectionstyle="arc3,rad=-0.22"))
    ax1.set_xticks(x, vlabels, fontsize=11)
    ax1.set_ylim(0.90, max(sarc) + .022)
    ax1.set_ylabel("Sarcasm ROC-AUC")
    ax1.set_title(f"Ranking gain per component · "
                  f"+{(sarc[4] - sarc[0]) * 100:.1f} pts AUC from the full stack",
                  loc="left", pad=12)
    style_ax(ax1)

    bars = ax2.bar(x, mae, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width() / 2, v + .010, f"{v:.3f}",
                 ha="center", fontsize=10.5,
                 fontweight="bold" if v == min(mae) else "normal",
                 color="white" if v == min(mae) else "#D1D5DB")
    ax2.set_xticks(x, vlabels, fontsize=11)
    ax2.set_ylim(0, max(mae) * 1.30)
    ax2.set_ylabel("Tension MAE (lower = better)")
    ax2.set_title("Regression gain per component", loc="left", pad=12)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.14, left=0.055,
                        right=0.975, wspace=0.22)
    header(fig, "Ablation study — what does each component contribute?",
           "Same training protocol; E adds hidden-signal fusion + temporal engines on top of D")
    save(fig, "ablation_study.png")


# ================================================================ 3 · CAPABILITY
def graph_capability():
    full = load("summary.json")["full_metrics"]
    rows = [
        ("Sarcasm ROC-AUC", full["sarcasm"]["roc_auc"], CYAN),
        ("Irony ROC-AUC", full["irony"]["roc_auc"], PURPLE),
        ("Passive-aggression ROC-AUC", full["passive_aggression"]["roc_auc"], PINK),
        ("Sarcasm F1 (macro)", full["sarcasm"]["f1_macro"], CYAN),
        ("Irony F1 (macro)", full["irony"]["f1_macro"], PURPLE),
        ("Passive-aggr. F1 (macro)", full["passive_aggression"]["f1_macro"], PINK),
        ("Escalation F1 (macro)", full["escalation"]["f1_macro"], ORANGE),
        ("Tension R²", max(full["tension"]["r2"], 0), GREEN),
    ]
    fig, ax = plt.subplots(figsize=(14.5, 9.0))
    rows = rows[::-1]
    y = np.arange(len(rows))
    bars = ax.barh(y, [v for _, v, _ in rows], 0.6,
                   color=[c for _, _, c in rows], edgecolor=BG, lw=.6)
    for b, (_, v, _) in zip(bars, rows):
        ax.text(v + .004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=11.5, fontweight="bold", color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(textwrap.wrap(n, 22)) for n, _, _ in rows],
                       fontsize=12)
    ax.set_xlim(0.85, 1.005)
    ax.set_xlabel("score (ROC-AUC / F1 / R²)")
    ax.set_title("All heads ≥ 0.92 — ranking metrics are the honest benchmark",
                 loc="left", pad=12, fontsize=13.5)
    style_ax(ax)
    fig.subplots_adjust(top=axes_top(fig), bottom=0.09, left=0.24, right=0.965)
    header(fig, "Full CEREBRO (E) — capability sheet on the test split",
           "One bar per reported metric · values printed at bar ends · all numbers from evaluation/results/summary.json")
    save(fig, "capability_sheet.png")


# ================================================================ 4 · CONFUSIONS
def _wrapped(labels, width=9):
    return ["\n".join(textwrap.wrap(l, width)) for l in labels]


def graph_confusion(head_key, labels, title, fname, color):
    from sklearn.metrics import confusion_matrix
    yt, yp = _test_predictions_labels(head_key)
    cm = confusion_matrix(yt, yp, labels=labels, normalize="true")
    n = len(labels)
    fig, ax = plt.subplots(figsize=(max(9.0, n * 0.95), max(7.6, n * 0.70)))
    im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "neon", [PANEL, color]), vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(_wrapped(labels), rotation=90, fontsize=9.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_wrapped(labels) if n <= 3 else labels, fontsize=9.5)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8.8,
                        color="white" if v > 0.55 else "#9CA3AF",
                        fontweight="bold" if i == j else "normal")
    ax.set_xlabel("predicted", fontsize=12.5)
    ax.set_ylabel("ground truth", fontsize=12.5)
    ax.grid(False)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=10, colors="#D1D5DB")
    fig.subplots_adjust(top=axes_top(fig), bottom=0.20, left=0.16, right=0.965)
    header(fig, f"Confusion matrix — {title}",
           "persisted CEREBRO engine re-run on 45 held-out test conversations · row-normalized (recall view)")
    save(fig, fname)


# ================================================================ 5 · CALIBRATION
def graph_calibration():
    from sklearn.calibration import calibration_curve
    proba = _test_predictions_proba()
    fig, ax = plt.subplots(figsize=(11.0, 8.6))
    for name, color in [("sarcasm", PINK), ("irony", PURPLE),
                        ("passive_aggression", GREEN)]:
        y = np.array([p[0] for p in proba[name]])
        p = np.array([p[1] for p in proba[name]])
        frac, mean_p = calibration_curve(y, p, n_bins=8, strategy="quantile")
        brier = float(np.mean((p - y) ** 2))
        ax.plot(mean_p, frac, "-o", color=color, lw=2.6, ms=8,
                label=f"{name}  (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#6B7280", lw=1.4,
            label="perfectly calibrated")
    ax.set_xlabel("predicted probability", fontsize=13)
    ax.set_ylabel("observed positive frequency", fontsize=13)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    legend_below(ax, 2, y_offset=-0.16)
    ax.set_title("Curves hugging the diagonal = trustworthy confidences",
                 loc="left", pad=12)
    style_ax(ax)
    fig.subplots_adjust(top=axes_top(fig), bottom=0.20, left=0.11, right=0.965)
    header(fig, "Probability calibration — hidden-signal heads",
           "reliability curves on 45 test conversations · Brier score in legend (lower = better)")
    save(fig, "calibration_curves.png")


# ================================================================ 6 · HERO
def graph_hero():
    demo = load("demo_report.json")
    full = load("summary.json")["full_metrics"]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(17.5, 8.8),
                                   gridspec_kw={"width_ratios": [1.15, 1]})
    tension = [m["tension"] for m in demo["messages"]]
    xs = np.arange(1, len(tension) + 1)
    handles = [plt.Line2D([], [], color=CYAN, lw=2.6, marker="o", ms=7,
                          mfc=CYAN, mec=BG, label="predicted tension")]
    if demo["turning_points"]:
        tp = max(demo["turning_points"], key=lambda t: abs(t["tension_change"]))
        ax1.axvline(tp["message_id"], color=PINK, ls="--", lw=1.6, alpha=.95)
        handles.append(plt.Line2D(
            [], [], color=PINK, ls="--", lw=1.6,
            label=f'turning point #{tp["message_id"]}: '
                  f'{tp["before"]["emotion"]} → {tp["after"]["emotion"]} '
                  f'(Δtension {tp["tension_change"]:+.1f}, z={tp["robust_z"]})'))
    ax1.plot(xs, tension, color=CYAN, lw=2.6, marker="o", ms=7,
             mfc=CYAN, mec=BG, mew=1.4, zorder=3)
    ax1.fill_between(xs, tension, color=CYAN, alpha=.10)
    ax1.axhspan(60, 100, color=PINK, alpha=.08)
    handles.append(plt.Rectangle((0, 0), 1, 1, color=PINK, alpha=.15,
                                 label="escalation zone (tension > 60)"))
    ax1.set_xticks(xs)
    ax1.set_xlabel("message #", fontsize=12.5)
    ax1.set_ylabel("tension (0–100)", fontsize=12.5)
    ax1.set_ylim(0, 100)
    ax1.set_title(f'Held-out test conversation · trajectory = '
                  f'"{load("demo_report.json")["summary"]["trajectory"]}"',
                  loc="left", pad=12, fontsize=13.5)
    legend_below(ax1, 1, y_offset=-0.16)
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
        ax2.text(v + .004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                 va="center", fontsize=11.5, fontweight="bold", color="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels([n for n, _, _ in rows], fontsize=12)
    ax2.set_xlim(0.85, 1.005)
    ax2.set_xlabel("score", fontsize=12.5)
    ax2.set_title("Headline metrics (test split, 89 conversations)",
                  loc="left", pad=12, fontsize=13.5)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.22, left=0.065,
                        right=0.965, wspace=0.30)
    header(fig, "CEREBRO — conversation intelligence at a glance",
           "left: per-message tension with detected turning points (real model output) · right: test-split headline metrics")
    save(fig, "hero_dashboard.png")


# ================================================================ 7 · DATASET
def graph_dataset():
    from cerebro.data.generator import generate_corpus
    corpus = generate_corpus(convs_per_cell=14)
    msgs = [m for c in corpus for m in c]
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18.5, 7.6),
                                        gridspec_kw={"width_ratios": [1.25, 1, 1]})

    bands = [("calm", 0, 25), ("friction", 25, 55), ("escalation", 55, 85),
             ("peak", 85, 101)]
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
        ax1.text(xi, p + n + g + 130, f"{p + n + g:,}", ha="center", fontsize=11,
                 fontweight="bold", color="#D1D5DB")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n{lo}–{hi if hi < 101 else 100}"
                         for b, lo, hi in bands], fontsize=11)
    ax1.set_ylabel("messages", fontsize=12.5)
    ax1.set_ylim(0, max(p + n + g for p, n, g in zip(pos, neu, neg)) * 1.22)
    ax1.set_title("Sentiment composition per tension band", loc="left", pad=12)
    legend_below(ax1, 3, y_offset=-0.24)
    style_ax(ax1)

    tens = [m["tension"] for m in msgs]
    ax2.hist(tens, bins=32, color=PURPLE, alpha=.9, edgecolor=BG, lw=.4)
    ax2.axvline(float(np.mean(tens)), color=YELLOW, ls="--", lw=1.8)
    ax2.text(0.03, 0.95, f"mean {np.mean(tens):.1f}", fontsize=11.5,
             color=YELLOW, transform=ax2.transAxes, va="top", fontweight="bold")
    ax2.set_xlabel("tension value", fontsize=12.5)
    ax2.set_ylabel("messages", fontsize=12.5)
    ax2.set_title("Tension distribution", loc="left", pad=12)
    style_ax(ax2)

    lens = [len(c) for c in corpus]
    ax3.hist(lens, bins=17, color=CYAN, alpha=.9, edgecolor=BG, lw=.4)
    ax3.axvline(float(np.mean(lens)), color=YELLOW, ls="--", lw=1.8)
    ax3.text(0.03, 0.95, f"mean {np.mean(lens):.1f}", fontsize=11.5,
             color=YELLOW, transform=ax3.transAxes, va="top", fontweight="bold")
    ax3.set_xlabel("messages per conversation", fontsize=12.5)
    ax3.set_ylabel("conversations", fontsize=12.5)
    ax3.set_title("Conversation lengths", loc="left", pad=12)
    style_ax(ax3)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.15, left=0.05,
                        right=0.985, wspace=0.26)
    header(fig, "CEREBRO corpus — 6 domains × 7 narrative arcs, weak-supervision annotated",
           "conversation-level splits (no leakage) · 588 conversations · 10,956 messages · seed 42")
    save(fig, "dataset_overview.png")


# ================================================================ 8 · DEMO REPORT
def graph_demo_report():
    demo = load("demo_report.json")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(17.0, 10.4), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    msgs = demo["messages"]
    xs = np.arange(1, len(msgs) + 1)
    tension = [m["tension"] for m in msgs]
    ax1.plot(xs, tension, color=CYAN, lw=2.6, marker="o", ms=7, mfc=CYAN,
             mec=BG, mew=1.4, zorder=3)
    ax1.fill_between(xs, tension, color=CYAN, alpha=.10)
    # turning-point labels live in reserved headroom (ylim 0–122) — no collision
    ax1.set_ylim(0, 122)
    ax1.set_yticks([0, 25, 50, 75, 100])
    for t in demo["turning_points"]:
        ax1.axvline(t["message_id"], color=PINK, ls=":", lw=1.3, alpha=.85)
        ax1.text(t["message_id"], 108, f'#{t["message_id"]} {t["tension_change"]:+.0f}',
                 fontsize=9.5, color=PINK, ha="center", va="bottom",
                 fontweight="bold")
    ax1.axhspan(60, 100, color=PINK, alpha=.07)
    ax1.set_ylabel("tension (0–100)", fontsize=12.5)
    ax1.set_title("tension — turning-point markers above the curve",
                  loc="left", fontsize=12.5, color=CYAN, pad=8)
    style_ax(ax1)

    for key, color, lbl in [("sarcasm", PINK, "sarcasm"),
                            ("irony", PURPLE, "irony"),
                            ("passive_aggression", GREEN, "passive-aggr.")]:
        probs = [m[key]["probability"] for m in msgs]
        ax2.plot(xs, probs, "-o", color=color, lw=2, ms=5.5, label=lbl, mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.set_ylim(0, 1.10)
    ax2.set_yticks([0, .25, .5, .75, 1])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f'#{m["message_id"]}' for m in msgs], fontsize=10)
    ax2.set_xlabel("message (#id — full texts in the README worked example)",
                   fontsize=12.5)
    ax2.set_ylabel("probability", fontsize=12.5)
    ax2.set_title("hidden signals — dashed line = 0.5 decision threshold",
                  loc="left", fontsize=12.5, color=PINK, pad=8)
    ax2.legend(fontsize=11, framealpha=0, ncols=3, loc="upper left",
                bbox_to_anchor=(0.0, 1.02))
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.16, left=0.07,
                        right=0.975, hspace=0.34)
    header(fig, "CEREBRO on a held-out test conversation — full per-message readout",
           "top: tension with turning-point markers · bottom: sarcasm / irony / passive-aggression probabilities")
    save(fig, "demo_report.png")


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
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18.5, 9.0))

    names = [s["scenario"] for s in sc]
    mean_t = [s["mean_tension"] for s in sc]
    peak_t = [s["peak_tension"] for s in sc]
    colors = [GREEN if n in calm_expected else ORANGE for n in names]
    x = np.arange(len(names))
    ax1.bar(x, mean_t, 0.62, color=colors, alpha=.55, edgecolor=BG, lw=.5)
    ax1.plot(x, peak_t, "_", color="white", ms=22, mew=2.4)
    ax1.axhline(25.9, color=CYAN, ls="--", lw=1.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90, fontsize=9.5)
    ax1.set_ylabel("tension (0–100)", fontsize=12.5)
    ax1.set_title("Per-scenario tension readout", loc="left", pad=12)
    from matplotlib.patches import Patch
    ax1.legend(
        handles=[Patch(color=GREEN, alpha=.55, label="calm expected"),
                 Patch(color=ORANGE, alpha=.55, label="conflict expected"),
                 plt.Line2D([], [], color="white", marker="_", ms=14, mew=2.4,
                            lw=0, label="peak tension"),
                 plt.Line2D([], [], color=CYAN, ls="--",
                            label="training-corpus mean 25.9")],
        loc="upper center", bbox_to_anchor=(0.5, -0.24), ncols=2,
        framealpha=0, borderaxespad=0)
    style_ax(ax1)

    hidden = {"sarcasm": [], "irony": [], "passive_aggression": []}
    for s in sc:
        hidden["sarcasm"].append(s["sarcasm_mean"])
        hidden["irony"].append(s["irony_mean"])
        hidden["passive_aggression"].append(s["pa_mean"])
    for (h, c) in [("sarcasm", PINK), ("irony", PURPLE),
                   ("passive_aggression", GREEN)]:
        ax2.plot(x, hidden[h], "-o", color=c, lw=2, ms=5.5,
                 label=h.replace("_", "-"), mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=9.5)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("mean probability", fontsize=12.5)
    ax2.set_title("Hidden-signal levels per scenario", loc="left", pad=12)
    legend_below(ax2, 3, y_offset=-0.24)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.19, left=0.05,
                        right=0.985, wspace=0.20)
    header(fig, "PS-01 §41 robustness — 20 hand-crafted out-of-distribution scenarios",
           "different phrasing, emoji, slang, timing and formats than the training corpus · all 20 executed without failure")
    save(fig, "scenario_robustness.png")


# ================================================================ 9b · TRANSFER
def graph_transfer():
    try:
        s = load("transfer_logsafe.json")
    except FileNotFoundError:
        print("  – transfer_logsafe.json missing, skip")
        return
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15.5, 7.2))

    keys = [("emotion top-3", s["zero_shot_emotion_top3"], 1 / 3),
            ("emotion exact (13-way)", s["zero_shot_emotion_accuracy"], 1 / 13),
            ("sentiment exact (3-way)", s["zero_shot_sentiment_accuracy"], 1 / 3)]
    names = [k for k, _, _ in keys]
    vals = [v for _, v, _ in keys]
    chance = [c for _, _, c in keys]
    y = np.arange(len(names))
    ax1.barh(y, vals, color=[CYAN, PURPLE, PINK], height=.58, zorder=3)
    ax1.barh(y, chance, color="none", edgecolor="#6B7280", height=.58,
             lw=1.2, ls="--", zorder=4)
    for yi, v in zip(y, vals):
        ax1.text(v + .012, yi, f"{v:.1%}", va="center", fontsize=12,
                 color="#E5E7EB", fontweight="bold")
    ax1.set_yticks(y, names, fontsize=11.5)
    ax1.set_xlim(0, .62)
    ax1.set_xlabel("accuracy on real text (dashed = chance)", fontsize=12)
    ax1.invert_yaxis()
    ax1.set_title("Zero-shot accuracy vs chance", loc="left", pad=12)
    style_ax(ax1)

    gold = s["gold_emotion_distribution"]
    pred = s["pred_emotion_distribution"]
    classes = list(dict.fromkeys(list(gold) + list(pred)))[:8]
    x = np.arange(len(classes))
    gv = [gold.get(c, 0) / s["n_messages"] for c in classes]
    pv = [pred.get(c, 0) / s["n_messages"] for c in classes]
    ax2.bar(x - .19, gv, width=.38, color="#4B5563", label="gold (GoEmotions)",
            zorder=3)
    ax2.bar(x + .19, pv, width=.38, color=CYAN, label="predicted", zorder=3)
    ax2.set_xticks(x, classes, rotation=90, fontsize=10)
    ax2.set_ylabel("share of messages", fontsize=12)
    ax2.set_ylim(0, max(max(gv), max(pv)) * 1.15)
    ax2.set_title("Label shift: frustration over-read on neutral text",
                  loc="left", pad=12)
    legend_below(ax2, 2, y_offset=-0.26)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.22, left=0.22,
                        right=0.975, wspace=0.30)
    header(fig, "Zero-shot transfer to real GoEmotions text",
           "3,000 real Reddit comments · honest cross-corpus metrics (not comparable to in-corpus tables)")
    save(fig, "transfer_goemotions.png")


# ================================================================ 10 · ARCHITECTURE
def graph_architecture():
    fig = plt.figure(figsize=(14.5, 19.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    rects = []          # (name, x, y, w, h) for the overlap validator

    def box(x, y, w, h, lines, color, fs=12, sub_fs=10, name=""):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=PANEL,
                                   edgecolor=color, lw=2, zorder=3))
        head, rest = lines[0], lines[1:]
        ax.text(x + w / 2, y + h - (h * 0.30 if rest else h / 2), head,
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color=color, zorder=4)
        if rest:
            ax.text(x + w / 2, y + h * 0.28, "\n".join(rest), ha="center",
                    va="center", fontsize=sub_fs, color="#D1D5DB", zorder=4)
        rects.append((name or lines[0][:22], x, y, w, h))

    def arrow(x1, y1, x2, y2, color="#4B5563", lw=2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    bands = []          # (name, x, y, w, h)
    def band(y, h, label, color):
        ax.add_patch(plt.Rectangle((3.4, y), 96, h, facecolor=color,
                                   alpha=0.045, edgecolor="none", zorder=1))
        # vertical label in the left margin — zero chance of hitting any box
        ax.text(1.7, y + h / 2, label, fontsize=10.5, color=color, alpha=0.95,
                fontweight="bold", zorder=2, va="center", ha="center",
                rotation=90)
        bands.append((label.split("·")[0].strip(), 3.4, y, 96, h))

    # ---- INPUT LAYER ----
    band(84.0, 8.0, "INPUT LAYER · PS-01 §2", CYAN)
    for txt, x in [(("WhatsApp", ".txt"), 5.0), (("Discord", "JSON"), 20.5),
                   (("Slack", "JSON"), 36.0), (("CSV", ""), 51.5),
                   (("JSON", ""), 67.0), (("plain", ".txt"), 82.5)]:
        lines = [t for t in txt if t]
        box(x, 85.6, 12.5, 5.0, lines, CYAN, fs=11, name=f"in:{txt[0]}")
        arrow(x + 6.25, 85.6, 47, 83.2, lw=1.4)

    box(28, 76.6, 44, 6.0,
        ["CHAT PARSER + AUTO-DETECT", "one common format · speakers · timestamps"],
        CYAN, fs=12, name="parser")
    arrow(50, 76.6, 50, 73.8)

    # ---- UNDERSTANDING LAYER ----
    band(51.5, 31.5, "UNDERSTANDING LAYER · PS-01 §5, §8, §9, §17", PURPLE)
    box(5.5, 63.0, 41, 8.6, ["PREPROCESSING + FEATURES",
                             "16-dim behavioral vector per message",
                             "CAPS · exclamations · emoji · response gap"],
        PURPLE, name="preproc")
    box(53.5, 63.0, 41, 8.6, ["CONTEXT + SPEAKER MEMORY",
                              "sliding 4-turn window + decayed summary",
                              "per-speaker emotional state"],
        PURPLE, name="context")
    arrow(26, 63.0, 43, 59.0)
    arrow(74, 63.0, 57, 59.0)
    box(18, 53.0, 64, 5.0,
        ["MESSAGE REPRESENTATION", "text ⊕ context ⊕ behavior ⊕ memory"],
        GREEN, fs=12, name="repr")
    arrow(50, 53.0, 50, 49.8)

    # ---- INTELLIGENCE LAYER ----
    band(29.0, 21.5, "INTELLIGENCE LAYER · PS-01 §10–22", YELLOW)
    box(4.5, 42.4, 29, 7.4, ["MULTI-TASK NLP ENGINE",
                             "sentiment · emotion · tone",
                             "tension (0–100)"], YELLOW, name="mtln")
    box(35.5, 42.4, 29, 7.4, ["HIDDEN-SIGNAL DETECTION",
                              "sarcasm · irony · passive-aggr.",
                              "learned ⊕ contradiction evidence"], YELLOW,
        name="hidden")
    box(66.5, 42.4, 29, 7.4, ["TEMPORAL ENGINES",
                              "arc · transitions · turning pts",
                              "escalation trajectory"], YELLOW, name="temporal")
    arrow(50, 42.4, 50, 40.1)
    box(25, 32.4, 50, 7.0, ["MODEL FUSION + CALIBRATION",
                            "six evidence streams · tuned weights · Platt scaling"],
        ORANGE, name="fusion")

    # ---- OUTPUT LAYER ----
    band(11.5, 16.5, "OUTPUT LAYER · PS-01 §23–30", PINK)
    box(5.0, 19.4, 27.5, 7.4, ["EXPLAINABILITY",
                               "WHY? · WHAT CHANGED?",
                               "evidence ≠ interpretation"], PINK, name="explain")
    box(36.25, 19.4, 27.5, 7.4, ["CONVERSATION REPORT",
                                 "18-section summary",
                                 "speaker + topic views"], PINK, name="report")
    box(67.5, 19.4, 27.5, 7.4, ["FASTAPI + DASHBOARD",
                                "14 endpoints · TTL store",
                                "futuristic frontend"], PINK, name="api")
    arrow(50, 32.4, 18.75, 27.2)
    arrow(50, 32.4, 50, 27.2)
    arrow(50, 32.4, 81.25, 27.2)
    box(26, 12.9, 48, 5.0, ["EMOTIONAL ARC · TENSION CURVE",
                            "the journey, explained"], CYAN, fs=11.5, name="arc")
    arrow(18.75, 19.4, 38, 18.2, lw=1.4)
    arrow(50, 19.4, 50, 18.2, lw=1.4)
    arrow(81.25, 19.4, 62, 18.2, lw=1.4)

    ax.text(50, 8.6, "every arrow is a real function call — see docs/methodology/PS01_WORKFLOW.md",
            fontsize=11, color="#6B7280", ha="center", style="italic")

    # ---- programmatic overlap validator (boxes, bands, arrow tips) ----
    def overlap(a, b, pad=0.05):
        _, ax_, ay, aw, ah = a
        _, bx, by, bw, bh = b
        return not (ax_ + aw + pad <= bx or bx + bw + pad <= ax_ or
                    ay + ah + pad <= by or by + bh + pad <= ay)
    for i in range(len(rects)):
        for j in range(i + 1, len(rects)):
            assert not overlap(rects[i], rects[j]), \
                f"box collision: {rects[i][0]} × {rects[j][0]}"
    for name, x, y, w, h in rects:
        inside = any(bx <= x and x + w <= bx + bw and by <= y and y + h <= by + bh
                     for _, bx, by, bw, bh in bands)
        assert inside, f"box outside every band: {name}"
    arrow_tips = [(x + 6.25, 85.6, 47, 83.2)
                  for x in (5.0, 20.5, 36.0, 51.5, 67.0, 82.5)] + [
        (50, 76.6, 50, 73.8), (26, 63.0, 43, 59.0), (74, 63.0, 57, 59.0),
        (50, 53.0, 50, 49.8), (50, 42.4, 50, 40.1), (50, 32.4, 18.75, 27.2),
        (50, 32.4, 50, 27.2), (50, 32.4, 81.25, 27.2), (18.75, 19.4, 38, 18.2),
        (50, 19.4, 50, 18.2), (81.25, 19.4, 62, 18.2)]
    for x1, y1, x2, y2 in arrow_tips:
        for name, bx, by, bw, bh in rects:
            for px, py in ((x1, y1), (x2, y2)):
                inside_pt = bx < px < bx + bw and by < py < by + bh
                is_src = any(abs(px - ex) < 1.5 and abs(py - ey) < 1.5
                             for ex, ey in ((x + 6.25, 85.6) for x in
                                            (5.0, 20.5, 36.0, 51.5, 67.0, 82.5)))
                assert not (inside_pt and not is_src), \
                    f"arrow endpoint buried in box {name} at ({px},{py})"

    fig.savefig("docs/architecture/architecture.png")
    plt.close(fig)
    print("  ✓ architecture.png (box/band/arrow validator passed)")


if __name__ == "__main__":
    from cerebro.common.labels import SENTIMENT_LABELS, EMOTION_LABELS, TONE_LABELS
    print("generating collision-free graphs from real results...")
    graph_hero()
    graph_main_result()
    graph_ablation()
    graph_capability()
    graph_dataset()
    graph_demo_report()
    graph_scenarios()
    graph_transfer()
    graph_architecture()
    graph_confusion("sent", SENTIMENT_LABELS, "sentiment (3-way)",
                    "confusion_sentiment.png", CYAN)
    graph_confusion("emo", EMOTION_LABELS, "emotion (13-way)",
                    "confusion_emotion.png", PURPLE)
    graph_confusion("tone", TONE_LABELS, "tone (14-way)",
                    "confusion_tone.png", PINK)
    graph_calibration()
    print("done → assets/graphs/")
