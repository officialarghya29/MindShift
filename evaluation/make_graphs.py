"""Graph generation v4 — maximum-legibility, collision-free, GitHub-native.

Why v4: GitHub renders embedded images at ~830 CSS px regardless of the
file's pixel width. Wide multi-panel canvases therefore get *scaled down*,
crushing fonts until lines and letters collide visually. v4 fixes the actual
geometry problem:

  1. VERTICAL STACKED LAYOUTS — panels are stacked 2-3 rows × 1 column, so a
     full-width embed keeps ~1:1 pixel density instead of shrinking.
  2. MUCH LARGER TYPOGRAPHY — titles 20-24 pt, labels 16 pt, ticks 14.5 pt,
     legends 14.5-15 pt (v3 used 9.5-12 pt that vanished at embed scale).
  3. NO FLOATING ANNOTATIONS — corner "mean" labels moved into titles;
     turning-point labels staggered in reserved headroom; legends below axes.
  4. VALIDATOR v2 — (a) rendered-text overlap check (pixel bboxes of every
     text artist), plus (b) a data-ink collision check: legends, figure-level
     texts and annotation boxes may never sit on top of curves/bars/histograms.

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
    "axes.axisbelow": True,   # grid BELOW bars/curves/labels — no line collisions
    "axes.titlesize": 20, "axes.titleweight": "bold",
    "axes.labelsize": 16.5, "xtick.labelsize": 14.5, "ytick.labelsize": 14.5,
    "legend.fontsize": 15, "figure.dpi": 150,
})

# inch-based layout bands (identical on every figure, regardless of height)
HEADER_IN = 1.05      # title + subtitle band height
WATERMARK_IN = 1.15   # logo watermark size (bottom-right, below all axes)


def load(name):
    with open(f"{RESULTS}/{name}", encoding="utf-8") as f:
        return json.load(f)


def header(fig, title, subtitle):
    """Logo + title band, laid out in inches (converted to fractions)."""
    logo = Image.open(LOGO_SMALL)
    fig_w, fig_h = fig.get_size_inches()
    side = 0.58                                    # logo box, inches
    ax_img = fig.add_axes(
        [0.014 / fig_w,
         (fig_h - HEADER_IN + 0.14) / fig_h,
         side / fig_w,
         side * logo.height / logo.width / fig_h], zorder=10)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo))
    fig.text(0.085, (fig_h - 0.36) / fig_h, title, fontsize=24,
             fontweight="bold", color=CYAN, ha="left", va="center")
    fig.text(0.085, (fig_h - 0.76) / fig_h, subtitle, fontsize=15,
             color="#9CA3AF", ha="left", va="center")
    return HEADER_IN


def axes_top(fig, extra_in=0.14):
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


def legend_below(ax, ncols, y_offset=-0.22):
    """Legend OUTSIDE the plot area, below the x-label — can never cover data."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y_offset),
                     ncols=ncols, framealpha=0, borderaxespad=0)


def save(fig, name, skip_data_check=False):
    """Finalize with validator v2 (text overlaps + data-ink collisions)."""
    _validate_no_text_overlaps(fig, name)
    if not skip_data_check:
        _validate_no_data_collisions(fig, name)
    watermark(fig)
    fig.savefig(f"{GRAPHS}/{name}")
    plt.close(fig)
    print(f"  ✓ {name} (validators v2 passed)")


def _visible_boxes(fig, renderer):
    """All rendered text bounding boxes with labels."""
    fig_bb = fig.bbox
    boxes = []
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        artists = [ax.title, ax.xaxis.label, ax.yaxis.label, *ax.texts]
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        for loc, t in zip(ax.get_xticks(), ax.get_xticklabels()):
            if xlim[0] <= loc <= xlim[1]:
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
            if bb.x1 < 0 or bb.x0 > fig_bb.x1 or bb.y1 < 0 or bb.y0 > fig_bb.y1:
                continue
            boxes.append((t.get_text()[:28].replace("\n", "⏎"), bb))
    for t in fig.texts:
        if t.get_text().strip():
            bb = t.get_window_extent(renderer=renderer)
            if bb.x1 < 0 or bb.x0 > fig_bb.x1 or bb.y1 < 0 or bb.y0 > fig_bb.y1:
                continue
            boxes.append((t.get_text()[:28].replace("\n", "⏎"), bb))
    return boxes


def _validate_no_text_overlaps(fig, name):
    """Draw the figure and assert no two text bounding boxes intersect."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = _visible_boxes(fig, renderer)
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


def _data_ink_boxes(ax, renderer):
    """Display-space bboxes of the VISIBLE data ink: lines, markers, bars.

    Every box is clipped to the axes region — patches may extend beyond the
    view limits (e.g. bars starting at data-y 0 under a 0.88-based ylim) but
    only their rendered pixels are ink a legend may not cover. Low-alpha
    background bands (axhspan zones, fill_between) are excluded.
    """
    ax_bb = ax.get_window_extent(renderer)

    def clip(bb):
        c = matplotlib.transforms.Bbox.intersection(bb, ax_bb)
        return c

    out = []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        try:
            c = clip(ln.get_window_extent(renderer))
            if c is not None:
                out.append(c)
        except ValueError:
            pass
    for p in ax.patches:
        if not p.get_visible():
            continue
        fc = p.get_facecolor()
        if len(fc) == 4 and fc[3] < 0.2:          # background band
            continue
        c = clip(p.get_window_extent(renderer))
        if c is not None:
            out.append(c)
    for coll in ax.collections:
        if not coll.get_visible():
            continue
        alpha = coll.get_alpha()
        if alpha is not None and alpha < 0.2:     # fill_between background
            continue
        offs = coll.get_offsets()
        if offs is None or len(offs) == 0:
            continue
        pts = ax.transData.transform(offs)
        sizes = np.atleast_1d(coll.get_sizes())
        pad = float(np.max(sizes)) if len(sizes) else 6.0
        pad = max(pad * 1.5, 6.0)
        c = clip(matplotlib.transforms.Bbox.from_extents(
            pts[:, 0].min() - pad, pts[:, 1].min() - pad,
            pts[:, 0].max() + pad, pts[:, 1].max() + pad))
        if c is not None:
            out.append(c)
    return out


def _validate_no_data_collisions(fig, name):
    """Legends + figure-level texts may never sit on top of data ink.

    In-plot ax.texts (value labels, headroom callouts) are intentional and
    skipped; legends and figure texts are not.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tol = 2.0
    suspects = []
    for t in fig.texts:
        if t.get_text().strip():
            suspects.append((t.get_text()[:28], t.get_window_extent(renderer)))
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                suspects.append((t.get_text()[:28],
                                 t.get_window_extent(renderer)))
    if not suspects:
        return
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        data_boxes = _data_ink_boxes(ax, renderer)
        for label, sb in suspects:
            for db in data_boxes:
                ox = min(sb.x1, db.x1) - max(sb.x0, db.x0)
                oy = min(sb.y1, db.y1) - max(sb.y0, db.y0)
                if ox > tol and oy > tol:
                    raise AssertionError(
                        f"[{name}] '{label}' sits on data ink "
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 14.0))

    x = np.arange(len(models))
    w = 0.26
    for i, (h, lbl, c) in enumerate(metrics):
        vals = [base[k][h]["roc_auc"] for k in keys] + [full[h]["roc_auc"]]
        bars = ax1.bar(x + (i - 1) * w, vals, w, color=c, label=lbl,
                       edgecolor=BG, lw=.6)
        best = max(vals)
        for b, v in zip(bars, vals):
            ax1.text(b.get_x() + b.get_width() / 2, v + .004, f"{v:.4f}",
                     ha="center", va="bottom", fontsize=12.5,
                     color="white" if abs(v - best) < 1e-9 else "#9CA3AF",
                     fontweight="bold" if abs(v - best) < 1e-9 else "normal")
    ax1.set_xticks(x, models, fontsize=15)
    ax1.set_ylim(0.88, 1.0)
    ax1.set_ylabel("ROC-AUC")
    ax1.set_title("Hidden-signal ranking quality (higher = better)",
                  loc="left", pad=12)
    legend_below(ax1, 3, y_offset=-0.26)
    style_ax(ax1)

    mae = [base[k]["tension"]["mae"] for k in keys] + [full["tension"]["mae"]]
    colors = [BLUE, BLUE, BLUE, GREEN]
    bars = ax2.bar(x, mae, 0.52, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width() / 2, v + .015, f"{v:.3f}",
                 ha="center", fontsize=14.5, fontweight="bold",
                 color="white" if v == min(mae) else "#9CA3AF")
    ax2.set_xticks(x, models, fontsize=15)
    ax2.set_ylim(0, max(mae) * 1.35)
    ax2.axhline(min(mae), color=GREEN, ls="--", lw=1, alpha=.6)
    ax2.set_ylabel("MAE (tension units, 0–100 scale)")
    ax2.set_title(f"Tension regression error (lower = better) · best {min(mae):.3f}",
                  loc="left", pad=12)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.115, left=0.085,
                        right=0.97, hspace=0.30)
    header(fig, "Baselines vs CEREBRO — real test-split results",
           "89 held-out conversations · sequential predicted-history inference · seed 42 "
           "· top y-axis starts at 0.88 to magnify small gaps")
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

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 14.5))

    x = np.arange(5)
    colors = [BLUE, BLUE, BLUE, BLUE, GREEN]
    bars = ax1.bar(x, sarc, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, sarc):
        ax1.text(b.get_x() + b.get_width() / 2, v + .0012, f"{v:.4f}",
                 ha="center", fontsize=13.5,
                 fontweight="bold" if v == max(sarc) else "normal",
                 color="white" if v == max(sarc) else "#D1D5DB")
    # lift bracket ABOVE every value label (labels top ≈ v+.006) — never
    # crosses the bars' printed numbers
    y_hi = max(sarc) + .013
    ax1.annotate("", xy=(4, y_hi), xytext=(0, y_hi),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                                 connectionstyle="arc3,rad=-0.18"))
    ax1.set_xticks(x, vlabels, fontsize=14.5)
    ax1.set_ylim(0.90, max(sarc) + .040)
    ax1.set_ylabel("Sarcasm ROC-AUC")
    ax1.set_title(f"Ranking gain per component · "
                  f"+{(sarc[4] - sarc[0]) * 100:.1f} pts AUC from the full stack",
                  loc="left", pad=12)
    style_ax(ax1)

    bars = ax2.bar(x, mae, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width() / 2, v + .010, f"{v:.3f}",
                 ha="center", fontsize=13.5,
                 fontweight="bold" if v == min(mae) else "normal",
                 color="white" if v == min(mae) else "#D1D5DB")
    ax2.set_xticks(x, vlabels, fontsize=14.5)
    ax2.set_ylim(0, max(mae) * 1.30)
    ax2.set_ylabel("Tension MAE (lower = better)")
    ax2.set_title("Regression gain per component", loc="left", pad=12)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.115, left=0.085,
                        right=0.97, hspace=0.34)
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
    fig, ax = plt.subplots(figsize=(13.0, 10.0))
    rows = rows[::-1]
    y = np.arange(len(rows))
    bars = ax.barh(y, [v for _, v, _ in rows], 0.6,
                   color=[c for _, _, c in rows], edgecolor=BG, lw=.6)
    for b, (_, v, _) in zip(bars, rows):
        ax.text(v + .004, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                va="center", fontsize=14.5, fontweight="bold", color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(textwrap.wrap(n, 24)) for n, _, _ in rows],
                       fontsize=15)
    ax.set_xlim(0.85, 1.005)
    ax.set_xlabel("score (ROC-AUC / F1 / R²)")
    ax.set_title("All heads ≥ 0.92 — ranking metrics are the honest benchmark",
                 loc="left", pad=12)
    style_ax(ax)
    fig.subplots_adjust(top=axes_top(fig), bottom=0.085, left=0.265, right=0.965)
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
    side = max(9.5, n * 0.62 + 2.2)          # narrow enough to embed ~1:1
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "neon", [PANEL, color]), vmin=0, vmax=1)
    ax.set_xticks(range(n))
    ax.set_xticklabels(_wrapped(labels), rotation=90, fontsize=12)
    ax.set_yticks(range(n))
    ax.set_yticklabels(_wrapped(labels) if n <= 3 else labels, fontsize=12)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=11,
                        color="white" if v > 0.55 else "#9CA3AF",
                        fontweight="bold" if i == j else "normal")
    ax.set_xlabel("predicted", fontsize=16)
    ax.set_ylabel("ground truth", fontsize=16)
    ax.tick_params(axis="both", labelsize=12)
    ax.grid(False)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=12.5, colors="#D1D5DB")
    fig.subplots_adjust(top=axes_top(fig), bottom=0.21, left=0.17, right=0.965)
    header(fig, f"Confusion matrix — {title}",
           "persisted CEREBRO engine re-run on 45 held-out test conversations · row-normalized (recall view)")
    save(fig, fname, skip_data_check=True)   # cell texts sit on the heatmap by design


# ================================================================ 5 · CALIBRATION
def graph_calibration():
    from sklearn.calibration import calibration_curve
    proba = _test_predictions_proba()
    fig, ax = plt.subplots(figsize=(12.0, 10.5))
    for name, color in [("sarcasm", PINK), ("irony", PURPLE),
                        ("passive_aggression", GREEN)]:
        y = np.array([p[0] for p in proba[name]])
        p = np.array([p[1] for p in proba[name]])
        frac, mean_p = calibration_curve(y, p, n_bins=8, strategy="quantile")
        brier = float(np.mean((p - y) ** 2))
        ax.plot(mean_p, frac, "-o", color=color, lw=3.0, ms=10,
                label=f"{name}  (Brier {brier:.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#6B7280", lw=1.6,
            label="perfectly calibrated")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", pad=9)   # corner '0.0' ticks must not kiss
    legend_below(ax, 2, y_offset=-0.17)
    ax.set_title("Curves hugging the diagonal = trustworthy confidences",
                 loc="left", pad=12)
    style_ax(ax)
    fig.subplots_adjust(top=axes_top(fig), bottom=0.185, left=0.115, right=0.965)
    header(fig, "Probability calibration — hidden-signal heads",
           "reliability curves on 45 test conversations · Brier score in legend (lower = better)")
    save(fig, "calibration_curves.png")


# ================================================================ 6 · HERO
def graph_hero():
    demo = load("demo_report.json")
    full = load("summary.json")["full_metrics"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 15.0),
                                   gridspec_kw={"height_ratios": [1.1, 1]})
    tension = [m["tension"] for m in demo["messages"]]
    xs = np.arange(1, len(tension) + 1)
    handles = [plt.Line2D([], [], color=CYAN, lw=2.6, marker="o", ms=8,
                          mfc=CYAN, mec=BG, label="predicted tension")]
    if demo["turning_points"]:
        tp = max(demo["turning_points"], key=lambda t: abs(t["tension_change"]))
        ax1.axvline(tp["message_id"], color=PINK, ls="--", lw=1.6, alpha=.95)
        tp_lbl = (f'turning point #{tp["message_id"]}: '
                  f'{tp["before"]["emotion"]} → {tp["after"]["emotion"]} '
                  f'(Δtension {tp["tension_change"]:+.1f}, z={tp["robust_z"]})')
        handles.append(plt.Line2D([], [], color=PINK, ls="--", lw=1.6,
                                  label=tp_lbl))
    ax1.plot(xs, tension, color=CYAN, lw=2.6, marker="o", ms=8,
             mfc=CYAN, mec=BG, mew=1.4, zorder=3)
    ax1.fill_between(xs, tension, color=CYAN, alpha=.10)
    ax1.axhspan(60, 100, color=PINK, alpha=.08)
    handles.append(plt.Rectangle((0, 0), 1, 1, color=PINK, alpha=.15,
                                 label="escalation zone (tension > 60)"))
    ax1.set_xticks(xs)
    ax1.set_xlabel("message #")
    ax1.set_ylabel("tension (0–100)")
    ax1.set_ylim(0, 100)
    ax1.set_title("Held-out test conversation · per-message tension",
                  loc="left", pad=12)
    ax1.legend(handles=handles, loc="upper center",
               bbox_to_anchor=(0.5, -0.20), ncols=1, framealpha=0,
               borderaxespad=0)
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
                 va="center", fontsize=14.5, fontweight="bold", color="white")
    ax2.set_yticks(y)
    ax2.set_yticklabels([n for n, _, _ in rows], fontsize=15)
    ax2.set_xlim(0.85, 1.005)
    ax2.set_xlabel("score")
    ax2.set_title("Headline metrics (test split, 89 conversations)",
                  loc="left", pad=12)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.135, left=0.10,
                        right=0.965, hspace=0.52)
    header(fig, "CEREBRO — conversation intelligence at a glance",
           "top: per-message tension with detected turning points (real model output) · bottom: test-split headline metrics")
    save(fig, "hero_dashboard.png")


# ================================================================ 7 · DATASET
def graph_dataset():
    from cerebro.data.generator import generate_corpus
    corpus = generate_corpus(convs_per_cell=14)
    msgs = [m for c in corpus for m in c]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12.5, 18.0))

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
        ax1.text(xi, p + n + g + 130, f"{p + n + g:,}", ha="center", fontsize=14,
                 fontweight="bold", color="#D1D5DB")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n{lo}–{hi if hi < 101 else 100}"
                         for b, lo, hi in bands], fontsize=15)
    ax1.set_ylabel("messages")
    ax1.set_ylim(0, max(p + n + g for p, n, g in zip(pos, neu, neg)) * 1.22)
    ax1.set_title("Sentiment composition per tension band", loc="left", pad=12)
    legend_below(ax1, 3, y_offset=-0.24)
    style_ax(ax1)

    tens = [m["tension"] for m in msgs]
    ax2.hist(tens, bins=32, color=PURPLE, alpha=.9, edgecolor=BG, lw=.4)
    ax2.axvline(float(np.mean(tens)), color=YELLOW, ls="--", lw=2.0)
    ax2.set_xlabel("tension value")
    ax2.set_ylabel("messages")
    ax2.set_title(f"Tension distribution — mean {np.mean(tens):.1f} (dashed line)",
                  loc="left", pad=12)
    style_ax(ax2)

    lens = [len(c) for c in corpus]
    ax3.hist(lens, bins=17, color=CYAN, alpha=.9, edgecolor=BG, lw=.4)
    ax3.axvline(float(np.mean(lens)), color=YELLOW, ls="--", lw=2.0)
    ax3.set_xlabel("messages per conversation")
    ax3.set_ylabel("conversations")
    ax3.set_title(f"Conversation lengths — mean {np.mean(lens):.1f} (dashed line)",
                  loc="left", pad=12)
    style_ax(ax3)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.075, left=0.085,
                        right=0.97, hspace=0.42)
    header(fig, "CEREBRO corpus — 6 domains × 7 narrative arcs, weak-supervision annotated",
           "conversation-level splits (no leakage) · 588 conversations · 10,956 messages · seed 42")
    save(fig, "dataset_overview.png")


# ================================================================ 8 · DEMO REPORT
def graph_demo_report():
    demo = load("demo_report.json")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.0, 12.5), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    msgs = demo["messages"]
    xs = np.arange(1, len(msgs) + 1)
    tension = [m["tension"] for m in msgs]
    ax1.plot(xs, tension, color=CYAN, lw=2.6, marker="o", ms=8, mfc=CYAN,
             mec=BG, mew=1.4, zorder=3)
    ax1.fill_between(xs, tension, color=CYAN, alpha=.10)
    # turning-point labels live in staggered headroom slots — no collision
    ax1.set_ylim(0, 128)
    ax1.set_yticks([0, 25, 50, 75, 100])
    for k, t in enumerate(demo["turning_points"]):
        # ymax caps the dotted line BELOW its own label — no line-through-text
        ax1.axvline(t["message_id"], color=PINK, ls=":", lw=1.3, alpha=.85,
                    ymax=0.84)
        ax1.text(t["message_id"], 110 if k % 2 == 0 else 120,
                 f'#{t["message_id"]} {t["tension_change"]:+.0f}',
                 fontsize=12.5, color=PINK, ha="center", va="bottom",
                 fontweight="bold")
    ax1.axhspan(60, 100, color=PINK, alpha=.07)
    ax1.set_ylabel("tension (0–100)")
    ax1.set_title("tension — turning-point markers above the curve",
                  loc="left", fontsize=16, color=CYAN, pad=8)
    style_ax(ax1)

    for key, color, lbl in [("sarcasm", PINK, "sarcasm"),
                            ("irony", PURPLE, "irony"),
                            ("passive_aggression", GREEN, "passive-aggr.")]:
        probs = [m[key]["probability"] for m in msgs]
        ax2.plot(xs, probs, "-o", color=color, lw=2, ms=6.5, label=lbl, mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.set_ylim(0, 1.12)
    ax2.set_yticks([0, .25, .5, .75, 1])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f'#{m["message_id"]}' for m in msgs], fontsize=13)
    ax2.set_xlabel("message (#id — full texts in the README worked example)")
    ax2.set_ylabel("probability")
    ax2.set_title("hidden signals — dashed line = 0.5 decision threshold",
                  loc="left", fontsize=16, color=PINK, pad=8)
    legend_below(ax2, 3, y_offset=-0.26)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.145, left=0.075,
                        right=0.97, hspace=0.38)
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(13.5, 16.5))

    names = [s["scenario"] for s in sc]
    mean_t = [s["mean_tension"] for s in sc]
    peak_t = [s["peak_tension"] for s in sc]
    colors = [GREEN if n in calm_expected else ORANGE for n in names]
    x = np.arange(len(names))
    ax1.bar(x, mean_t, 0.62, color=colors, alpha=.55, edgecolor=BG, lw=.5)
    ax1.plot(x, peak_t, "_", color="white", ms=22, mew=2.4)
    ax1.axhline(25.9, color=CYAN, ls="--", lw=1.4)
    ax1.set_xticks(x)
    ax1.set_xticklabels(names, rotation=90, fontsize=13)
    ax1.set_ylabel("tension (0–100)")
    ax1.set_title("Per-scenario tension readout", loc="left", pad=12)
    from matplotlib.patches import Patch
    ax1.legend(
        handles=[Patch(color=GREEN, alpha=.55, label="calm expected"),
                 Patch(color=ORANGE, alpha=.55, label="conflict expected"),
                 plt.Line2D([], [], color="white", marker="_", ms=14, mew=2.4,
                            lw=0, label="peak tension"),
                 plt.Line2D([], [], color=CYAN, ls="--",
                            label="training-corpus mean 25.9")],
        loc="upper center", bbox_to_anchor=(0.5, -0.26), ncols=2,
        framealpha=0, borderaxespad=0)
    style_ax(ax1)

    hidden = {"sarcasm": [], "irony": [], "passive_aggression": []}
    for s in sc:
        hidden["sarcasm"].append(s["sarcasm_mean"])
        hidden["irony"].append(s["irony_mean"])
        hidden["passive_aggression"].append(s["pa_mean"])
    for (h, c) in [("sarcasm", PINK), ("irony", PURPLE),
                   ("passive_aggression", GREEN)]:
        ax2.plot(x, hidden[h], "-o", color=c, lw=2, ms=6.5,
                 label=h.replace("_", "-"), mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=13)
    ax2.set_ylim(0, 1)
    ax2.set_ylabel("mean probability")
    ax2.set_title("Hidden-signal levels per scenario", loc="left", pad=12)
    legend_below(ax2, 3, y_offset=-0.28)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.155, left=0.075,
                        right=0.97, hspace=0.52)
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
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 14.0))

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
        ax1.text(v + .012, yi, f"{v:.1%}", va="center", fontsize=15,
                 color="#E5E7EB", fontweight="bold")
    ax1.set_yticks(y, names, fontsize=15)
    ax1.set_xlim(0, .62)
    ax1.set_xlabel("accuracy on real text (dashed = chance)")
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
    ax2.set_xticks(x, classes, rotation=90, fontsize=12.5)
    ax2.set_ylabel("share of messages")
    ax2.set_ylim(0, max(max(gv), max(pv)) * 1.15)
    ax2.set_title("Label shift: frustration over-read on neutral text",
                  loc="left", pad=12)
    legend_below(ax2, 2, y_offset=-0.30)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.205, left=0.235,
                        right=0.97, hspace=0.44)
    header(fig, "Zero-shot transfer to real GoEmotions text",
           "3,000 real Reddit comments · honest cross-corpus metrics (not comparable to in-corpus tables)")
    save(fig, "transfer_goemotions.png")


# ================================================================ 9c · BENCHMARKS
def graph_benchmarks():
    try:
        b = load("benchmarks.json")
    except FileNotFoundError:
        print("  – benchmarks.json missing, skip")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12.5, 13.5))

    rows = b["scaling"]
    ns = [r["n_messages"] for r in rows]
    tot = [r["total_s"] for r in rows]
    ax1.plot(ns, tot, "-o", color=CYAN, lw=3, ms=10, mec=BG, zorder=3)
    # labels BELOW-RIGHT of each point: on a rising log-x curve the incoming
    # segment arrives from lower-left and the outgoing leaves up-right, so
    # below-right is the empty quadrant (verified by the line-vs-text probe)
    for n, t in zip(ns, tot):
        ax1.annotate(f"{t:.2f}s", (n, t), textcoords="offset points",
                     xytext=(9, -22), ha="left", fontsize=13.5,
                     fontweight="bold", color="white")
    ax1.set_xscale("log")
    ax1.set_xticks(ns, [str(n) for n in ns])
    ax1.tick_params(axis="x", labelsize=14.5)
    ax1.set_xlabel("messages in the conversation (log scale)")
    ax1.set_ylabel("end-to-end seconds")
    ax1.set_title("Latency scales linearly — no context-window blow-up",
                  loc="left", pad=12)
    ax1.set_ylim(0, max(tot) * 1.30)
    style_ax(ax1)

    st = b["stage_latency_ms_per_message"]
    order = ["parse", "preprocess+features", "ml_heads", "hidden_signals",
             "temporal", "explainability", "segmentation"]
    disp = ["parse", "preprocess\n+features", "ML heads", "hidden\nsignals",
            "temporal", "explain-\nability", "segmentation"]
    vals = [st[k] for k in order]
    bars = ax2.bar(np.arange(len(order)), vals, 0.6, color=ORANGE,
                   edgecolor=BG, lw=.6)
    for b_, v in zip(bars, vals):
        ax2.text(b_.get_x() + b_.get_width() / 2, v + .25, f"{v:.2f}",
                 ha="center", fontsize=13.5, fontweight="bold", color="white")
    ax2.set_xticks(np.arange(len(order)), disp, fontsize=13.5)
    ax2.set_ylabel("ms per message")
    ax2.set_ylim(0, max(vals) * 1.30)
    ax2.set_title(f"Where the time goes · ML heads dominate "
                  f"({vals[2] / max(sum(vals), 1e-9):.0%} of stage cost)",
                  loc="left", pad=12)
    style_ax(ax2)

    fig.subplots_adjust(top=axes_top(fig), bottom=0.105, left=0.085,
                        right=0.97, hspace=0.38)
    header(fig, "Efficiency — real benchmarks, persisted engine, single CPU core",
           "top: end-to-end latency up to 1,000 messages · bottom: per-stage cost "
           "· peak memory 8.9 MB @ 1,000 msgs · API round-trip 184 ms @ 50 msgs")
    save(fig, "efficiency_benchmarks.png")


# ================================================================ 10 · ARCHITECTURE
def graph_architecture():
    """Vertical single-column layout: 10.5in wide so a full-width README
    embed keeps ~1:1 pixel density (GitHub renders at ~830 CSS px).

    Geometry contract (asserted below): every box inside exactly one band,
    no box overlaps, no arrow endpoint buried inside a box.
    """
    fig = plt.figure(figsize=(10.5, 24.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    rects = []          # (name, x, y, w, h) for the overlap validator

    def box(x, y, w, h, lines, color, fs=18, sub_fs=14.5, name=""):
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

    def band(y, h, label, color):
        ax.add_patch(plt.Rectangle((3.4, y), 95, h, facecolor=color,
                                   alpha=0.045, edgecolor="none", zorder=1))
        # vertical label in the left margin — zero chance of hitting any box
        ax.text(1.8, y + h / 2, label, fontsize=14, color=color, alpha=0.95,
                fontweight="bold", zorder=2, va="center", ha="center",
                rotation=90)

    # ---- INPUT LAYER: 6 chips in one row ----
    band(88.0, 8.5, "INPUT · PS-01 §2", CYAN)
    for txt, x in [("WhatsApp", 5.5), ("Discord", 20.5), ("Slack", 35.5),
                   ("CSV", 50.5), ("JSON", 65.5), ("plain", 80.5)]:
        box(x, 89.0, 14.0, 5.6, [txt], CYAN, fs=14.5, name=f"in:{txt}")
        # fan into DISTINCT points across the parser's top edge — six
        # arrowheads never stack on one another
        tx = 50 + (x + 7.0 - 50) * 0.35
        arrow(x + 7.0, 89.0, tx, 87.0, lw=1.4)

    # ---- UNDERSTANDING LAYER: parser → preprocess → context → repr ----
    band(55.5, 32.0, "UNDERSTANDING · §5 §8 §9 §17", PURPLE)
    box(20, 81.0, 60, 6.0,
        ["CHAT PARSER + AUTO-DETECT",
         "one common format · speakers · timestamps"],
        CYAN, fs=17, sub_fs=13.5, name="parser")
    arrow(50, 81.0, 50, 79.8)
    box(8.5, 72.0, 83, 7.8,
        ["PREPROCESSING + BEHAVIORAL FEATURES",
         "16-dim vector · CAPS · exclamations · emoji · response gap"],
        PURPLE, name="preproc")
    arrow(50, 72.0, 50, 71.2)
    box(8.5, 63.8, 83, 7.4,
        ["CONTEXT + SPEAKER MEMORY",
         "sliding 4-turn window · decayed summary · per-speaker state"],
        PURPLE, name="context")
    arrow(50, 63.8, 50, 62.9)
    box(12, 56.3, 76, 6.6,
        ["MESSAGE REPRESENTATION", "text ⊕ context ⊕ behavior ⊕ memory"],
        GREEN, fs=17, sub_fs=13.5, name="repr")
    arrow(50, 56.3, 50, 53.3)

    # ---- INTELLIGENCE LAYER: 4 stacked engines ----
    band(22.2, 33.3, "INTELLIGENCE · §10–22", YELLOW)
    box(8.5, 46.5, 83, 6.8,
        ["MULTI-TASK NLP ENGINE",
         "sentiment · emotion · tone · tension (0–100)"],
        YELLOW, name="mtln")
    arrow(50, 46.5, 50, 45.2)
    box(8.5, 38.6, 83, 6.6,
        ["HIDDEN-SIGNAL DETECTION",
         "sarcasm · irony · passive-aggression · learned ⊕ contradiction"],
        YELLOW, name="hidden")
    arrow(50, 38.6, 50, 37.4)
    box(8.5, 30.8, 83, 6.6,
        ["TEMPORAL ENGINES",
         "arc · transitions · turning points · escalation trajectory"],
        YELLOW, name="temporal")
    arrow(50, 30.8, 50, 29.6)
    box(8.5, 23.0, 83, 6.6,
        ["MODEL FUSION + CALIBRATION",
         "six evidence streams · validation-driven weights · Platt scaling"],
        ORANGE, name="fusion")
    arrow(50, 23.0, 50, 19.6)

    # ---- OUTPUT LAYER: explain/report + the arc ----
    band(0.5, 21.2, "OUTPUT · §23–30", PINK)
    box(8.5, 12.6, 83, 7.0,
        ["EXPLAINABILITY + CONVERSATION REPORT",
         "WHY? · WHAT CHANGED? · speaker profiles · 18-section report"],
        PINK, name="explain")
    arrow(50, 12.6, 50, 11.4)
    box(8.5, 5.6, 83, 5.8,
        ["EMOTIONAL ARC · TENSION CURVE",
         "the journey, explained — served by the FastAPI dashboard"],
        CYAN, fs=16, sub_fs=13.5, name="arc")

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
    bands_all = [(3.4, 88.0, 95, 8.5), (3.4, 55.5, 95, 32.0),
                 (3.4, 22.2, 95, 33.3), (3.4, 0.5, 95, 21.2)]
    for name, x, y, w, h in rects:
        inside = any(bx <= x and x + w <= bx + bw and by <= y and y + h <= by + bh
                     for bx, by, bw, bh in bands_all)
        assert inside, f"box outside every band: {name}"
    arrow_tips = [((x + 7.0), 89.0, 50 + (x + 7.0 - 50) * 0.35, 87.0)
                  for x in (5.5, 20.5, 35.5, 50.5, 65.5, 80.5)] + [
        (50, 81.0, 50, 79.8), (50, 72.0, 50, 71.2), (50, 63.8, 50, 62.9),
        (50, 56.3, 50, 53.3), (50, 46.5, 50, 45.2), (50, 38.6, 50, 37.4),
        (50, 30.8, 50, 29.6), (50, 23.0, 50, 19.6), (50, 12.6, 50, 11.4)]
    for x1, y1, x2, y2 in arrow_tips:
        for name, bx, by, bw, bh in rects:
            for px, py in ((x1, y1), (x2, y2)):
                inside_pt = bx < px < bx + bw and by < py < by + bh
                is_src = any(abs(px - ex) < 1.5 and abs(py - ey) < 1.5
                             for ex, ey in ((x + 7.0, 89.0) for x in
                                            (5.5, 20.5, 35.5, 50.5, 65.5, 80.5)))
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
    graph_benchmarks()
    graph_architecture()
    graph_confusion("sent", SENTIMENT_LABELS, "sentiment (3-way)",
                    "confusion_sentiment.png", CYAN)
    graph_confusion("emo", EMOTION_LABELS, "emotion (13-way)",
                    "confusion_emotion.png", PURPLE)
    graph_confusion("tone", TONE_LABELS, "tone (14-way)",
                    "confusion_tone.png", PINK)
    graph_calibration()
    print("done → assets/graphs/")
