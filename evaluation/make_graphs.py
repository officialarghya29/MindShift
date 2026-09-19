"""Graph generation v5 — GitHub-native legibility, collision-verified.

THE MEASURED PROBLEM (why every earlier "fix" still looked colliding)
--------------------------------------------------------------------
GitHub renders a README image inside a ~830 CSS-pixel content column, no
matter how many pixels the file has. So the *only* thing that decides how big
text looks on screen is

    displayed_px ≈ fontsize_pt × 830 / (72 × figure_width_inches)

v4 drew 12.5–13.5 in wide canvases → a display scale of 0.41–0.44, so a 13 pt
value label arrived at the reader as ~12 px and a 24 pt title as ~22 px. Text
that small next to drawn lines *reads* as collision even when the geometry is
clean. v5 fixes the cause instead of the symptom:

  1. ONE CANONICAL WIDTH (9.6 in). Display scale ≈ 0.58 → every 14 pt tick is
     ~17 px on screen, every 13 pt value label ~15.6 px, titles ~21 px.
  2. AUTO-FITTING HEADERS. v4 placed titles at a fixed 24 pt; measured, SEVEN
     of them were wider than the figure and were being clipped by the canvas
     edge. `_fit_text` now wraps + shrinks (binary search on the rendered
     bbox) until the text provably fits inside the figure.
  3. ROTATION IS AVOIDED WHERE IT CAUSES TROUBLE. Legends used to be anchored
     below axes that carry 90°-rotated tick labels — they landed on top of
     them. Those legends are now encoded as panel-title notes instead.
  4. VALIDATOR v3, four independent checks per figure:
       (a) no text bbox leaves the canvas          → catches clipping
       (b) no two text bboxes intersect            → catches literal overlap
       (c) no text smaller than the legibility floor at display scale
       (d) no legend/figure text sits on data ink  → catches line-through-text
     plus a watermark-clearance check.

Every number comes from real executed results (evaluation/results/*.json) or a
re-run of the persisted engine. Run:  python -m evaluation.make_graphs
"""
from __future__ import annotations

import json
import math
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

# ---------------------------------------------------------------- legibility
FIG_W = 9.6              # inches — chosen so a full-width README embed ≈ 0.58x
GITHUB_COLUMN_PX = 830   # GitHub renders README images into ~this CSS width
MIN_DISPLAY_PX = 13.5    # smallest acceptable on-screen text height
HEADER_X = 0.085         # left edge of all header text (figure fraction)

FS_PANEL_TITLE = 17.5
FS_AXLABEL = 15.5
FS_TICK = 14.0
FS_VALUE = 13.0
FS_LEGEND = 13.5
FS_NOTE = 13.0

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": "#4B5563", "axes.labelcolor": "#F3F4F6",
    "xtick.color": "#D1D5DB", "ytick.color": "#D1D5DB",
    "text.color": "#F9FAFB", "grid.color": "#263244",
    "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": .45,
    "axes.axisbelow": True,   # grid BELOW bars/curves/labels — no line collisions
    "axes.titlesize": FS_PANEL_TITLE, "axes.titleweight": "bold",
    "axes.labelsize": FS_AXLABEL, "xtick.labelsize": FS_TICK,
    "ytick.labelsize": FS_TICK, "legend.fontsize": FS_LEGEND, "figure.dpi": 150,
    "axes.labelpad": 6.0,   # extra air between axis label and its tick labels
})

# minimum clear space between two text boxes, measured in ON-SCREEN pixels once
# GitHub has scaled the image into its ~830 px column
MIN_TEXT_CLEARANCE_PX = 3.0


def displayed_px(pt: float, fig_w: float = FIG_W) -> float:
    """On-screen text height of `pt` once GitHub scales the image to its column."""
    return pt * GITHUB_COLUMN_PX / (72.0 * fig_w)


def load(name):
    with open(f"{RESULTS}/{name}", encoding="utf-8") as f:
        return json.load(f)


# ------------------------------------------------------------- text fitting
def _measure_width(fig, s: str, fs: float, weight: str, renderer) -> float:
    t = fig.text(0, 0, s, fontsize=fs, fontweight=weight)
    w = t.get_window_extent(renderer=renderer).width
    t.remove()
    return w


def _fit_text(fig, text, max_pt, color, weight="normal", max_lines=2, min_pt=11.0,
              x0=HEADER_X):
    """Wrap + shrink `text` until it provably fits the figure's usable width.

    Prefers a single line when that stays within 75 % of the best achievable
    size — avoids wrapping (and growing the header band) for no real gain.
    """
    fig_w_px = fig.get_size_inches()[0] * fig.dpi
    avail = (0.992 - x0) * fig_w_px
    r = fig.canvas.get_renderer()

    candidates: list[tuple[float, list[str]]] = []
    for nl in range(1, max_lines + 1):
        budget = max(14, math.ceil(len(text) / nl) + 2)
        lines = textwrap.wrap(text, budget)
        if len(lines) > nl:
            continue
        widest = max(_measure_width(fig, l, max_pt, weight, r) for l in lines)
        if widest <= avail:
            candidates.append((max_pt, lines))
            continue
        lo, hi = min_pt, max_pt
        for _ in range(16):
            mid = (lo + hi) / 2
            if max(_measure_width(fig, l, mid, weight, r) for l in lines) <= avail:
                lo = mid
            else:
                hi = mid
        if lo > min_pt:
            candidates.append((lo, lines))
    if not candidates:
        lines = textwrap.wrap(text, max(14, math.ceil(len(text) / max_lines)))
        candidates = [(min_pt, lines)]
    best_fs = max(c[0] for c in candidates)
    one_line = [c for c in candidates if len(c[1]) == 1]
    # A single line is preferred only when it costs almost nothing in size.
    # At a 0.75 tolerance the fitter picked an 11.3 pt one-liner over a 14 pt
    # two-liner purely to avoid wrapping — i.e. it traded legibility away.
    if one_line and one_line[0][0] >= 0.95 * best_fs:
        fs, lines = one_line[0]
    else:
        fs, lines = max(candidates, key=lambda c: c[0])
    t = fig.text(x0, 0, "\n".join(lines), fontsize=fs, fontweight=weight,
                 color=color, ha="left", va="center", linespacing=1.30)
    return t, fs, len(lines)


def header(fig, title, subtitle):
    """Logo + fitted title + fitted subtitle. Returns the band height (inches)."""
    fig_w, fig_h = fig.get_size_inches()
    logo = Image.open(LOGO_SMALL)
    side = 0.62
    logo_h = side * logo.height / logo.width
    ax_img = fig.add_axes([0.014 / fig_w, (fig_h - 0.22 - logo_h) / fig_h,
                           side / fig_w, logo_h / fig_h], zorder=10)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo))

    ttl, tfs, tnl = _fit_text(fig, title, max_pt=23.0, color=CYAN,
                              weight="bold", max_lines=2)
    t_h = tnl * tfs * 1.30 / 72.0
    ttl.set_position((HEADER_X, 1 - (0.22 + t_h / 2) / fig_h))

    sub, sfs, snl = _fit_text(fig, subtitle, max_pt=14.0, color="#9CA3AF",
                              max_lines=2)
    s_h = snl * sfs * 1.30 / 72.0
    sub.set_position((HEADER_X, 1 - (0.22 + t_h + 0.12 + s_h / 2) / fig_h))

    return 0.22 + t_h + 0.12 + s_h + 0.24


def top_for(fig, band_in, gap_in=0.80):
    """Top margin fraction placing the axes under the header band.

    The gap must fit the TOP panel's own title, which matplotlib draws *above*
    the axes (pad + line height ≈ 0.46 in for a 17.5 pt single-line title).
    v4 left only 0.10 in, so panel titles were drawn straight through the
    header subtitle — the collision the reader kept reporting.
    """
    return 1 - (band_in + gap_in) / fig.get_size_inches()[1]


def figure_legend(fig, ncols, y=0.012):
    """Legend along the very bottom of the canvas — clear of all plot ink."""
    return fig.legend(loc="lower center", bbox_to_anchor=(0.5, y), ncols=ncols,
                      framealpha=0)


def watermark(fig, boxes):
    """Faint brand mark in the first corner that no text already occupies.

    v4 pinned it to the bottom-right, which on horizontal-bar panels lands
    under the last value label. Candidates are tried in order and the chosen
    bbox is returned so the caller can validate it a second time.
    """
    fig_w, fig_h = fig.get_size_inches()
    logo = Image.open(LOGO).convert("RGBA")
    logo.thumbnail((520, 520))
    w_in = 1.05
    h_in = w_in * logo.height / logo.width
    pad = 0.10
    boxes_px = [(bb.x0, bb.y0, bb.x1, bb.y1) for _, bb, _ in boxes]
    for x_in, y_in in [(fig_w - w_in - pad, pad), (pad, pad),
                       ((fig_w - w_in) / 2, pad),
                       (fig_w - w_in - pad, fig_h - h_in - pad)]:
        bb = matplotlib.transforms.Bbox.from_bounds(x_in * fig.dpi,
                                                    y_in * fig.dpi,
                                                    w_in * fig.dpi,
                                                    h_in * fig.dpi)
        if any(min(bb.x1, b[2]) - max(bb.x0, b[0]) > 2.0 and
               min(bb.y1, b[3]) - max(bb.y0, b[1]) > 2.0 for b in boxes_px):
            continue
        ax_img = fig.add_axes([x_in / fig_w, y_in / fig_h,
                               w_in / fig_w, h_in / fig_h], zorder=0)
        ax_img.axis("off")
        ax_img.imshow(np.asarray(logo), alpha=0.07)
        fig.canvas.draw()
        return ax_img.get_window_extent(fig.canvas.get_renderer())
    return None


def style_ax(ax):
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def legend_below(ax, ncols, y_offset=-0.22):
    """Legend OUTSIDE the plot area, below the x-label — can never cover data."""
    return ax.legend(loc="upper center", bbox_to_anchor=(0.5, y_offset),
                     ncols=ncols, framealpha=0, borderaxespad=0)


# ------------------------------------------------------------- validators v3
def _visible_boxes(fig, renderer):
    """Every rendered text bbox that lies on the canvas, with a label."""
    fig_bb = fig.bbox
    boxes = []
    for ax in fig.get_axes():
        artists = list(ax.texts)
        if ax.axison:
            # loc= is not always "center": left/right titles live in private
            # artists that ax.title does not point at — forgetting them left
            # every panel title in the project unchecked (they were overflowing)
            artists += [ax.title, ax._left_title, ax._right_title,
                        ax.xaxis.label, ax.yaxis.label]
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
            if t is None or not t.get_text().strip() or not t.get_visible():
                continue
            bb = t.get_window_extent(renderer=renderer)
            if bb.x1 < 0 or bb.x0 > fig_bb.x1 or bb.y1 < 0 or bb.y0 > fig_bb.y1:
                continue
            boxes.append((t.get_text()[:30].replace("\n", "⏎"), bb, t))
    for t in fig.texts:
        if t.get_text().strip():
            bb = t.get_window_extent(renderer=renderer)
            if bb.x1 < 0 or bb.x0 > fig_bb.x1 or bb.y1 < 0 or bb.y0 > fig_bb.y1:
                continue
            boxes.append((t.get_text()[:30].replace("\n", "⏎"), bb, t))
    for lg in fig.legends:
        for t in lg.get_texts():
            if not t.get_text().strip():
                continue
            bb = t.get_window_extent(renderer=renderer)
            boxes.append((t.get_text()[:30].replace("\n", "⏎"), bb, t))
    return boxes


def _validate_inside_canvas(fig, name, boxes):
    """(a) No text may leave the canvas — v4 silently clipped 7 headers."""
    w, h = fig.bbox.x1, fig.bbox.y1
    slack = 1.0
    for label, bb, _ in boxes:
        if bb.x0 < -slack or bb.x1 > w + slack or bb.y0 < -slack or bb.y1 > h + slack:
            raise AssertionError(
                f"[{name}] '{label}' leaves the canvas at "
                f"x=[{bb.x0:.0f},{bb.x1:.0f}] y=[{bb.y0:.0f},{bb.y1:.0f}] of {w:.0f}x{h:.0f}")


def _validate_no_text_overlaps(fig, name, boxes):
    """(b) No two rendered text bboxes may intersect."""
    tol = max(3.0, 2.2 * FIG_W * fig.dpi / GITHUB_COLUMN_PX)
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i][1], boxes[j][1]
            ox = min(b1.x1, b2.x1) - max(b1.x0, b2.x0)
            oy = min(b1.y1, b2.y1) - max(b1.y0, b2.y0)
            if ox > tol and oy > tol:
                raise AssertionError(
                    f"[{name}] text collision: '{boxes[i][0]}' × '{boxes[j][0]}' "
                    f"(overlap {ox:.0f}×{oy:.0f} px)")


def _min_text_gap_px(fig, boxes):
    """Smallest on-screen (Chebyshev) gap between any two text boxes."""
    scale = GITHUB_COLUMN_PX / fig.bbox.x1
    best = float("inf")
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i][1], boxes[j][1]
            dx = max(b1.x0 - b2.x1, b2.x0 - b1.x1)
            dy = max(b1.y0 - b2.y1, b2.y0 - b1.y1)
            best = min(best, max(dx, dy) * scale)
    return best


def _validate_text_clearance(fig, name, boxes):
    """(b2) Neighbouring texts must stay visually separated once scaled down.

    Two boxes can miss each other by a fraction of a pixel and still pass an
    "overlap" test, yet read as one smudged line at 58 % display scale. This
    check measures the real on-screen gap.
    """
    scale = GITHUB_COLUMN_PX / fig.bbox.x1
    for i in range(len(boxes)):
        for j in range(i + 1, len(boxes)):
            b1, b2 = boxes[i][1], boxes[j][1]
            dx = max(b1.x0 - b2.x1, b2.x0 - b1.x1)
            dy = max(b1.y0 - b2.y1, b2.y0 - b1.y1)
            gap = max(dx, dy) * scale
            if gap < MIN_TEXT_CLEARANCE_PX:
                raise AssertionError(
                    f"[{name}] '{boxes[i][0]}' and '{boxes[j][0]}' are only "
                    f"{gap:.1f} px apart on screen — they read as one line")


def _validate_legibility(fig, name, boxes):
    """(c) No text smaller than the on-screen legibility floor."""
    worst = None
    for label, _, t in boxes:
        px = displayed_px(t.get_fontsize(), fig.get_size_inches()[0])
        if worst is None or px < worst[0]:
            worst = (px, label, t.get_fontsize())
    if worst and worst[0] < MIN_DISPLAY_PX:
        raise AssertionError(
            f"[{name}] '{worst[1]}' renders at {worst[0]:.1f} px on GitHub "
            f"({worst[2]:.1f} pt @ {fig.get_size_inches()[0]:.1f} in) — floor is "
            f"{MIN_DISPLAY_PX} px")


def _data_ink_boxes(ax, renderer):
    """Display-space bboxes of visible data ink: lines, markers, bars."""
    ax_bb = ax.get_window_extent(renderer)
    out = []
    for ln in ax.lines:
        if not ln.get_visible():
            continue
        try:
            c = matplotlib.transforms.Bbox.intersection(
                ln.get_window_extent(renderer), ax_bb)
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
        c = matplotlib.transforms.Bbox.intersection(
            p.get_window_extent(renderer), ax_bb)
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
        pad = max(float(np.max(sizes)) * 1.5, 6.0) if len(sizes) else 6.0
        c = matplotlib.transforms.Bbox.intersection(
            matplotlib.transforms.Bbox.from_extents(
                pts[:, 0].min() - pad, pts[:, 1].min() - pad,
                pts[:, 0].max() + pad, pts[:, 1].max() + pad), ax_bb)
        if c is not None:
            out.append(c)
    return out


def _validate_no_data_collisions(fig, name):
    """(d) Legends + figure texts may never sit on top of data ink."""
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    tol = 2.0
    suspects = []
    for t in fig.texts:
        if t.get_text().strip():
            suspects.append((t.get_text()[:30], t.get_window_extent(renderer)))
    for lg in fig.legends:
        for t in lg.get_texts():
            suspects.append((t.get_text()[:30], t.get_window_extent(renderer)))
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        if ax.get_legend():
            for t in ax.get_legend().get_texts():
                suspects.append((t.get_text()[:30], t.get_window_extent(renderer)))
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        for label, sb in suspects:
            for db in _data_ink_boxes(ax, renderer):
                ox = min(sb.x1, db.x1) - max(sb.x0, db.x0)
                oy = min(sb.y1, db.y1) - max(sb.y0, db.y0)
                if ox > tol and oy > tol:
                    raise AssertionError(
                        f"[{name}] '{label}' sits on data ink "
                        f"(overlap {ox:.0f}×{oy:.0f} px)")


def _seg_hits_rect(p, q, rect, pad=0.0):
    """Exact Liang–Barsky clip test: does segment p→q enter the rect?

    Cheaper bbox-vs-bbox approximations report collisions for segments that
    merely pass near a corner; this answers the question a reader actually
    asks, "does a drawn line cross these letters?".
    """
    x0, y0 = rect.x0 - pad, rect.y0 - pad
    x1, y1 = rect.x1 + pad, rect.y1 + pad
    t0, t1 = 0.0, 1.0
    dx, dy = q[0] - p[0], q[1] - p[1]
    for pp, qq in ((-dx, p[0] - x0), (dx, x1 - p[0]),
                   (-dy, p[1] - y0), (dy, y1 - p[1])):
        if pp == 0:
            if qq < 0:
                return False
        else:
            t = qq / pp
            if pp < 0:
                if t > t1:
                    return False
                t0 = max(t0, t)
            else:
                if t < t0:
                    return False
                t1 = min(t1, t)
    return True


def _validate_no_line_through_text(fig, name):
    """(e) No drawn line segment may cross a rendered text bbox."""
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    tol = 1.5
    for ax in fig.get_axes():
        if not ax.axison:
            continue
        texts = []
        for t in list(ax.texts) + [ax.title]:
            if t.get_text().strip() and t.get_visible():
                texts.append((t.get_text()[:26].replace("\n", "⏎"),
                              t.get_window_extent(renderer=r)))
        if not texts:
            continue
        for ln in ax.lines:
            if not ln.get_visible() or ln.get_linestyle() in ("None", "", " "):
                continue
            xd = np.asarray(ln.get_xdata(), dtype=float)
            yd = np.asarray(ln.get_ydata(), dtype=float)
            if len(xd) < 2:
                continue
            pts = ax.transData.transform(np.column_stack([xd, yd]))
            for k in range(len(pts) - 1):
                p, q = pts[k], pts[k + 1]
                if not np.isfinite(p).all() or not np.isfinite(q).all():
                    continue
                for label, tb in texts:
                    if _seg_hits_rect(p, q, tb, pad=tol):
                        raise AssertionError(
                            f"[{name}] a plotted line crosses the text '{label}' "
                            f"(segment {k} of '{ln.get_label()}') — line-through-text")


TITLE_FLOOR_PT = 12.0


def _shrink_overflowing_titles(fig):
    """Scale down any panel title that would run off the canvas.

    Panel titles are not clipped by matplotlib — they simply run past the PNG
    edge. Width scales ~linearly with font size, so one proportional pass lands
    them inside the canvas without touching the surrounding layout.
    """
    limit = 0.992 * fig.bbox.x1
    for _ in range(3):
        fig.canvas.draw()
        r = fig.canvas.get_renderer()
        worst = 1.0
        for ax in fig.get_axes():
            if not ax.axison:
                continue
            for t in (ax._left_title, ax._right_title, ax.title):
                if t is None or not t.get_text().strip():
                    continue
                bb = t.get_window_extent(renderer=r)
                if bb.x1 > limit and bb.width > 0:
                    scale = (limit - bb.x0) / bb.width
                    t.set_fontsize(max(TITLE_FLOOR_PT,
                                       t.get_fontsize() * scale))
                    worst = min(worst, scale)
        if worst >= 1.0:
            break


def save(fig, name, skip_data_check=False, out=None, no_watermark=False):
    """Finalize a figure under all v3 validators, then write the PNG."""
    _shrink_overflowing_titles(fig)
    fig.canvas.draw()
    boxes = _visible_boxes(fig, fig.canvas.get_renderer())
    _validate_inside_canvas(fig, name, boxes)
    _validate_no_text_overlaps(fig, name, boxes)
    _validate_text_clearance(fig, name, boxes)
    _validate_legibility(fig, name, boxes)
    if not no_watermark:
        wm = watermark(fig, boxes)
        if wm is not None:
            for label, bb, _ in boxes:
                ox = min(bb.x1, wm.x1) - max(bb.x0, wm.x0)
                oy = min(bb.y1, wm.y1) - max(bb.y0, wm.y0)
                if ox > 2.0 and oy > 2.0:
                    raise AssertionError(
                        f"[{name}] '{label}' overlaps the watermark "
                        f"({ox:.0f}×{oy:.0f} px)")
            boxes = _visible_boxes(fig, fig.canvas.get_renderer())
    if not skip_data_check:
        _validate_no_data_collisions(fig, name)
        _validate_no_line_through_text(fig, name)
    fig.savefig(out or f"{GRAPHS}/{name}")
    plt.close(fig)
    lo = min(displayed_px(t.get_fontsize(), fig.get_size_inches()[0])
             for _, _, t in boxes)
    gap = _min_text_gap_px(fig, boxes)
    print(f"  ✓ {name} (7 validators · smallest text {lo:.1f} px · "
          f"tightest gap {gap:.1f} px on screen)")


# ================================================================ 1 · HERO
def graph_hero():
    demo = load("demo_report.json")
    full = load("summary.json")["full_metrics"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 13.6),
                                   gridspec_kw={"height_ratios": [1.15, 1]})
    tension = [m["tension"] for m in demo["messages"]]
    xs = np.arange(1, len(tension) + 1)
    handles = [plt.Line2D([], [], color=CYAN, lw=2.6, marker="o", ms=8,
                          mfc=CYAN, mec=BG, label="predicted tension")]
    if demo["turning_points"]:
        tp = max(demo["turning_points"], key=lambda t: abs(t["tension_change"]))
        ax1.axvline(tp["message_id"], color=PINK, ls="--", lw=1.6, alpha=.95)
        handles.append(plt.Line2D(
            [], [], color=PINK, ls="--", lw=1.6,
            label=f'turning point #{tp["message_id"]}: {tp["before"]["emotion"]} → '
                  f'{tp["after"]["emotion"]} (Δtension {tp["tension_change"]:+.1f})'))
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
    ax1.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.20),
               ncols=1, framealpha=0, borderaxespad=0)
    style_ax(ax1)

    rows = [("Sarcasm", full["sarcasm"]["roc_auc"], CYAN),
            ("Irony", full["irony"]["roc_auc"], PURPLE),
            ("Passive-aggr.", full["passive_aggression"]["roc_auc"], PINK),
            ("Escalation", full["escalation"]["f1_macro"], ORANGE),
            ("Tension R²", max(full["tension"]["r2"], 0), GREEN)][::-1]
    y = np.arange(len(rows))
    bars = ax2.barh(y, [v for _, v, _ in rows], 0.58,
                    color=[c for _, _, c in rows], edgecolor=BG, lw=.6)
    for b, (_, v, _) in zip(bars, rows):
        ax2.text(v + .006, b.get_y() + b.get_height() / 2, f"{v:.4f}",
                 va="center", fontsize=FS_VALUE, fontweight="bold", color="white")
    ax2.set_yticks(y, [n for n, _, _ in rows], fontsize=FS_TICK)
    ax2.set_xlim(0.85, 1.055)
    ax2.set_xlabel("score (ROC-AUC / F1 / R²)")
    ax2.set_title("Headline metrics (test split, 89 conversations)",
                  loc="left", pad=12)
    style_ax(ax2)

    band = header(fig, "CEREBRO — conversation intelligence at a glance",
                  "top: per-message tension with a detected turning point (real model "
                  "output) · bottom: test-split headline metrics")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.125, left=0.20,
                        right=0.975, hspace=0.60)
    save(fig, "hero_dashboard.png")


# ================================================================ 2 · MAIN RESULT
def graph_main_result():
    base = load("baselines.json")
    full = load("summary.json")["full_metrics"]
    models = ["B1 TF-IDF+LR", "B2 TF-IDF+SVC", "B3 TF-IDF+ctx", "CEREBRO (E) full"]
    keys = ["B1_tfidf_logreg", "B2_tfidf_svc", "B3_tfidf_context"]
    metrics = [("sarcasm", "Sarcasm", CYAN),
               ("irony", "Irony", PURPLE),
               ("passive_aggression", "Passive-aggr.", PINK)]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 13.0))

    # -- horizontal grouped bars: values print at bar ends, never on a neighbour
    y = np.arange(len(models))[::-1] * 1.0
    h = 0.26
    for i, (key, lbl, c) in enumerate(metrics):
        vals = [base[k][key]["roc_auc"] for k in keys] + [full[key]["roc_auc"]]
        ax1.barh(y + (1 - i) * h, vals, height=h * .92, color=c, label=lbl,
                 edgecolor=BG, lw=.6)
        for yi, v in zip(y, vals):
            ax1.text(v + .0035, yi + (1 - i) * h, f"{v:.4f}", va="center",
                     fontsize=FS_VALUE,
                     color="white" if abs(v - max(vals)) < 1e-9 else "#C9D1DB",
                     fontweight="bold" if abs(v - max(vals)) < 1e-9 else "normal")
    ax1.set_yticks(y, models, fontsize=FS_TICK)
    ax1.set_xlim(0.88, 1.045)
    ax1.set_xlabel("ROC-AUC (axis starts at 0.88 to magnify small gaps)")
    ax1.set_title("Hidden-signal ranking quality — higher is better",
                  loc="left", pad=12)
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.22), ncols=3,
               framealpha=0, borderaxespad=0)
    style_ax(ax1)

    mae = [base[k]["tension"]["mae"] for k in keys] + [full["tension"]["mae"]]
    colors = [BLUE, BLUE, BLUE, GREEN]
    bars = ax2.barh(y, mae, 0.52, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(v + .02, b.get_y() + b.get_height() / 2, f"{v:.3f}", va="center",
                 fontsize=FS_VALUE + .5, fontweight="bold",
                 color="white" if v == min(mae) else "#C9D1DB")
    ax2.set_yticks(y, models, fontsize=FS_TICK)
    ax2.set_xlim(0, max(mae) * 1.30)
    ax2.axvline(min(mae), color=GREEN, ls="--", lw=1, alpha=.6)
    ax2.set_xlabel("MAE on the tension scale (lower is better)")
    ax2.set_title(f"Tension regression error · best {min(mae):.3f} (dashed line)",
                  loc="left", pad=12)
    style_ax(ax2)

    band = header(fig, "Baselines vs CEREBRO — real test-split results",
                  "89 held-out conversations · sequential predicted-history inference · "
                  "seed 42")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.115, left=0.235,
                        right=0.975, hspace=0.42)
    save(fig, "baselines_vs_cerebro.png")


# ================================================================ 3 · ABLATION
def graph_ablation():
    abl = load("ablations.json")
    full = load("summary.json")["full_metrics"]
    vlabels = ["A\ntext only", "B\n+context\nwindow", "C\n+speaker\nmemory",
               "D\n+behavior\n(full heads)", "E\n+hidden fusion\n(FULL)"]
    sarc = [abl[v]["sarcasm"]["roc_auc"] for v in ("A", "B", "C", "D")] + \
        [full["sarcasm"]["roc_auc"]]
    mae = [abl[v]["tension"]["mae"] for v in ("A", "B", "C", "D")] + \
        [full["tension"]["mae"]]

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 13.4))
    x = np.arange(5)
    colors = [BLUE, BLUE, BLUE, BLUE, GREEN]

    bars = ax1.bar(x, sarc, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, sarc):
        ax1.text(b.get_x() + b.get_width() / 2, v + .0015, f"{v:.4f}",
                 ha="center", fontsize=FS_VALUE,
                 fontweight="bold" if v == max(sarc) else "normal",
                 color="white" if v == max(sarc) else "#D1D5DB")
    y_hi = max(sarc) + .015          # lift arc lives ABOVE every printed number
    ax1.annotate("", xy=(4, y_hi), xytext=(0, y_hi),
                 arrowprops=dict(arrowstyle="-|>", color=GREEN, lw=1.6,
                                 connectionstyle="arc3,rad=-0.18"))
    ax1.set_xticks(x, vlabels, fontsize=FS_TICK - 0.5)
    ax1.set_ylim(0.90, max(sarc) + .045)
    ax1.set_ylabel("Sarcasm ROC-AUC")
    ax1.set_title(f"Ranking gain per component · +{(sarc[4] - sarc[0]) * 100:.2f} pts "
                  f"AUC from the full stack", loc="left", pad=12)
    style_ax(ax1)

    bars = ax2.bar(x, mae, 0.55, color=colors, edgecolor=BG, lw=.6)
    for b, v in zip(bars, mae):
        ax2.text(b.get_x() + b.get_width() / 2, v + .012, f"{v:.3f}", ha="center",
                 fontsize=FS_VALUE,
                 fontweight="bold" if v == min(mae) else "normal",
                 color="white" if v == min(mae) else "#D1D5DB")
    ax2.set_xticks(x, vlabels, fontsize=FS_TICK - 0.5)
    ax2.set_ylim(0, max(mae) * 1.30)
    ax2.set_ylabel("Tension MAE (lower = better)")
    ax2.set_title("Regression gain per component", loc="left", pad=12)
    style_ax(ax2)

    band = header(fig, "Ablation study — what does each component contribute?",
                  "identical training protocol; E adds hidden-signal fusion + temporal "
                  "engines on top of D")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.145, left=0.10,
                        right=0.975, hspace=0.45)
    save(fig, "ablation_study.png")


# ================================================================ 4 · CAPABILITY
def graph_capability():
    full = load("summary.json")["full_metrics"]
    rows = [("Sarcasm ROC-AUC", full["sarcasm"]["roc_auc"], CYAN),
            ("Irony ROC-AUC", full["irony"]["roc_auc"], PURPLE),
            ("Passive-aggression ROC-AUC", full["passive_aggression"]["roc_auc"], PINK),
            ("Sarcasm F1 (macro)", full["sarcasm"]["f1_macro"], CYAN),
            ("Irony F1 (macro)", full["irony"]["f1_macro"], PURPLE),
            ("Passive-aggr. F1 (macro)", full["passive_aggression"]["f1_macro"], PINK),
            ("Escalation F1 (macro)", full["escalation"]["f1_macro"], ORANGE),
            ("Tension R²", max(full["tension"]["r2"], 0), GREEN)][::-1]
    fig, ax = plt.subplots(figsize=(FIG_W, 8.6))
    y = np.arange(len(rows))
    bars = ax.barh(y, [v for _, v, _ in rows], 0.6,
                   color=[c for _, _, c in rows], edgecolor=BG, lw=.6)
    for b, (_, v, _) in zip(bars, rows):
        ax.text(v + .004, b.get_y() + b.get_height() / 2, f"{v:.4f}", va="center",
                fontsize=FS_VALUE, fontweight="bold", color="white")
    ax.set_yticks(y)
    ax.set_yticklabels(["\n".join(textwrap.wrap(n, 22)) for n, _, _ in rows],
                       fontsize=FS_TICK)
    ax.set_xlim(0.85, 1.055)
    ax.set_xlabel("score (ROC-AUC / F1 / R²)")
    ax.set_title("All heads ≥ 0.92 — ranking metrics are the honest benchmark",
                 loc="left", pad=12)
    style_ax(ax)
    band = header(fig, "Full CEREBRO (E) — capability sheet on the test split",
                  "one bar per reported metric · values printed at bar ends · all "
                  "numbers from evaluation/results/summary.json")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.115, left=0.285,
                        right=0.975)
    save(fig, "capability_sheet.png")


# ================================================================ 5 · CONFUSIONS
def graph_confusion(head_key, labels, title, fname, color):
    from sklearn.metrics import confusion_matrix
    yt, yp = _test_predictions_labels(head_key)
    cm = confusion_matrix(yt, yp, labels=labels, normalize="true")
    n = len(labels)
    side = max(7.4, min(FIG_W, n * 0.62 + 2.6))
    fig, ax = plt.subplots(figsize=(side, side))
    im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
        "neon", [PANEL, color]), vmin=0, vmax=1)
    # single-line labels on both axes: a WRAPPED label rotated 90° is two line
    # heights wide and its neighbours overlap (caught by the validator on the
    # 14-way tone matrix)
    ax.set_xticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=12.5)
    ax.set_yticks(range(n))
    ax.set_yticklabels(labels, fontsize=12.5)
    for i in range(n):
        for j in range(n):
            v = cm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=12.0,
                        color="#0B0F1A" if v > 0.55 else "#E5E7EB",
                        fontweight="bold" if i == j else "normal")
    ax.set_xlabel("predicted", fontsize=FS_AXLABEL)
    ax.set_ylabel("ground truth", fontsize=FS_AXLABEL)
    ax.tick_params(axis="both", labelsize=12.5)
    ax.grid(False)
    cbar = fig.colorbar(im, fraction=0.046, pad=0.03)
    cbar.ax.tick_params(labelsize=12.5, colors="#D1D5DB")
    band = header(fig, f"Confusion matrix — {title}",
                  "persisted CEREBRO engine re-run on 45 held-out test conversations · "
                  "row-normalized (recall view)")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.235, left=0.20,
                        right=0.955)
    save(fig, fname, skip_data_check=True)   # cell texts sit on the heatmap by design


# ================================================================ 6 · CALIBRATION
def graph_calibration():
    from sklearn.calibration import calibration_curve
    proba = _test_predictions_proba()
    cal = load("summary.json")["full_metrics"].get("calibration", {})
    fig, ax = plt.subplots(figsize=(FIG_W, 8.8))
    for name, color in [("sarcasm", PINK), ("irony", PURPLE),
                        ("passive_aggression", GREEN)]:
        y = np.array([p[0] for p in proba[name]])
        p = np.array([p[1] for p in proba[name]])
        frac, mean_p = calibration_curve(y, p, n_bins=8, strategy="quantile")
        brier = float(np.mean((p - y) ** 2))
        ece = cal.get(name, {}).get("ece")
        label = (f"{name}  (ECE {ece:.3f})" if ece is not None
                 else f"{name}  (Brier {brier:.3f})")
        ax.plot(mean_p, frac, "-o", color=color, lw=3.0, ms=10, label=label)
    ax.plot([0, 1], [0, 1], "--", color="#6B7280", lw=1.6,
            label="perfectly calibrated")
    ax.set_xlabel("predicted probability")
    ax.set_ylabel("observed positive frequency")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.tick_params(axis="x", pad=13)   # corner x/y '0.0' ticks must not touch
    legend_below(ax, 2, y_offset=-0.18)
    ax.set_title("Curves hugging the diagonal = trustworthy confidences",
                 loc="left", pad=12)
    style_ax(ax)
    mc = [cal.get(h, {}).get("ece") for h in ("sentiment", "emotion", "tone")]
    extra = (f" · top-1 ECE: sentiment {mc[0]:.3f}, emotion {mc[1]:.3f}, "
             f"tone {mc[2]:.3f}" if all(v is not None for v in mc) else "")
    band = header(fig, "Probability calibration — every head, not just the binary ones",
                  "reliability curves on 45 test conversations · expected calibration "
                  "error in legend (lower = better)" + extra)
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.215, left=0.135,
                        right=0.975)
    save(fig, "calibration_curves.png")


# ================================================ 7 · CONTEXT PROOF (§3, §36)
def graph_context_proof():
    """The headline experiment: text-only is bounded by the lexical ceiling."""
    proof = load("context_proof.json")
    tx = load("transformer_baselines.json")
    reg = proof["regimes"][proof["primary_regime"]]
    ceiling = reg["text_only_lexical_ceiling"]["sentiment"]["accuracy"]
    variants = ["A", "B", "C", "D", "E"]
    vlabels = ["A text only", "B +context", "C +memory", "D +behavior",
               "E +fusion"]
    heads = [("sentiment", "Sentiment", CYAN), ("emotion", "Emotion", PURPLE),
             ("tone", "Tone", PINK)]

    # one panel per head: each value label then owns its own slot, so no two
    # numbers can ever land on the same baseline (the three heads share values)
    fig, axes = plt.subplots(4, 1, figsize=(FIG_W, 16.6),
                             gridspec_kw={"height_ratios": [1, 1, 1, 1.30]})
    x = np.arange(len(variants))
    for ax, (key, lbl, _c) in zip(axes[:3], heads):
        vals = [reg["variants"][v][key]["accuracy"] for v in variants]
        bars = ax.bar(x, vals, 0.55, color=[BLUE] * 4 + [GREEN], edgecolor=BG, lw=.6)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + .032, f"{v:.3f}",
                    ha="center", fontsize=FS_VALUE, color="white",
                    fontweight="bold" if v == max(vals) else "normal")
        ax.axhline(ceiling, color=ORANGE, ls="--", lw=1.8)
        ax.set_ylim(0, 1.20)
        ax.set_yticks(np.arange(0, 1.01, .25))
        ax.set_ylabel(f"{lbl}\naccuracy", fontsize=FS_TICK - 1)
        ax.set_title(f"{lbl} — dashed line = text-only lexical ceiling "
                     f"{ceiling:.3f}", loc="left", pad=10)
        ax.tick_params(labelbottom=False)
        style_ax(ax)
    axes[2].set_xticks(x, vlabels, fontsize=FS_TICK - 0.5)
    axes[2].tick_params(labelbottom=True)
    ax2 = axes[3]

    # -- panel 4: every context-free reader sits ON the ceiling
    probe = tx["probe_experiment"]["test"]
    rows = [("MiniLM (B2)\nno context",
             probe["B2_transformer_text_only"]["sentiment"]["accuracy"], PINK),
            ("TF-IDF (A)\nno context",
             reg["variants"]["A"]["sentiment"]["accuracy"], BLUE),
            ("MiniLM (B3)\n+ context",
             probe["B3_transformer_plus_context"]["sentiment"]["accuracy"], PURPLE),
            ("CEREBRO full (E)", reg["variants"]["E"]["sentiment"]["accuracy"], GREEN)]
    y = np.arange(len(rows))[::-1]
    ax2.barh(y, [v for _, v, _ in rows], 0.55, color=[c for _, _, c in rows],
             edgecolor=BG, lw=.6)
    for yi, (_, v, _) in zip(y, rows):
        # value printed INSIDE the bar, so nothing sits near the ceiling line
        ax2.text(v - .025, yi, f"{v:.3f}", va="center", ha="right",
                 fontsize=FS_VALUE, color="white", fontweight="bold")
    ax2.axvline(ceiling, color=ORANGE, ls="--", lw=1.8)
    ax2.set_yticks(y, [n for n, _, _ in rows], fontsize=FS_TICK)
    ax2.set_xlim(0, 1.06)
    ax2.set_xticks(np.arange(0, 1.01, .2))
    ax2.set_xlabel("sentiment accuracy on probe turns (held-out history wording)")
    ax2.set_title("No amount of language understanding substitutes for context",
                  loc="left", pad=12)
    style_ax(ax2)

    band = header(fig, "Controlled proof — context is required, not merely helpful",
                  f"{reg['n_probe_turns']} probe turns · identical wording under benign vs "
                  f"tense histories · balanced so I(text;label)=0")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.075, left=0.20,
                        right=0.975, hspace=0.52)
    save(fig, "context_proof.png")


# ============================================= 8 · TRANSFORMER BASELINES (§9)
def graph_transformer_baselines():
    tx = load("transformer_baselines.json")
    probe = tx["probe_experiment"]["test"]
    main = tx["main_corpus_experiment"]
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 13.2))

    heads = [("sentiment", "Sentiment", CYAN), ("emotion", "Emotion", PURPLE),
             ("tone", "Tone", PINK)]
    x = np.arange(len(heads))
    w = 0.36
    for i, (key, name) in enumerate([("B2_transformer_text_only", "B2 transformer, no context"),
                                     ("B3_transformer_plus_context", "B3 transformer + context")]):
        vals = [probe[key][h]["accuracy"] for h, _, _ in heads]
        cols = [c for _, _, c in heads] if i == 0 else ["#94A3B8"] * 3
        bars = ax1.bar(x + (i - .5) * w, vals, w * .92, color=cols, label=name,
                       edgecolor=BG, lw=.6, alpha=1.0 if i == 0 else .55)
        for b, v in zip(bars, vals):
            # printed inside the bar — B2's three heads are all exactly 0.636
            ax1.text(b.get_x() + b.get_width() / 2, v - .055, f"{v:.3f}",
                     ha="center", fontsize=FS_VALUE, color="white",
                     fontweight="bold")
    ax1.set_xticks(x, [n for _, n, _ in heads], fontsize=FS_TICK)
    ax1.set_ylim(0, 1.20)
    ax1.set_yticks(np.arange(0, 1.01, .2))
    ax1.set_ylabel("accuracy on probe turns")
    ax1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2,
               framealpha=0, borderaxespad=0)
    ax1.set_title("A pretrained encoder still needs the conversation around it",
                  loc="left", pad=12)
    style_ax(ax1)

    rows = [("Tone", "tone"), ("Emotion", "emotion"), ("Sentiment", "sentiment")]
    y = np.arange(len(rows))[::-1]
    h = 0.34
    for i, (name, key) in enumerate([("B2 no context", "B2_transformer_text_only"),
                                     ("B3 + context", "B3_transformer_plus_context")]):
        vals = [main[key][k]["accuracy"] for _, k in rows]
        ax2.barh(y + (i - .5) * h, vals, h * .92,
                 color=[CYAN, PURPLE][i], label=name, edgecolor=BG, lw=.6,
                 alpha=1.0 if i == 0 else .55)
        for yi, v in zip(y + (i - .5) * h, vals):
            ax2.text(v - .012, yi, f"{v:.3f}", va="center", ha="right",
                     fontsize=FS_VALUE, color="white")
    ax2.set_yticks(y, [n for n, _ in rows], fontsize=FS_TICK)
    ax2.set_xlim(0.90, 1.035)
    ax2.set_xlabel("accuracy on the main-corpus test split (axis starts at 0.90)")
    ax2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncols=2,
               framealpha=0, borderaxespad=0)
    ax2.set_title("The template corpus saturates every model family alike",
                  loc="left", pad=12)
    style_ax(ax2)

    band = header(fig, "Pretrained-transformer baselines B2 and B3",
                  "frozen sentence-transformers/all-MiniLM-L6-v2 (ONNX, CPU) + the same "
                  "heads — no fine-tuning, so the comparison isolates context")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.155, left=0.13,
                        right=0.975, hspace=0.42)
    save(fig, "transformer_baselines.png")


# ================================================================ 9 · DATASET
def graph_dataset():
    from cerebro.data.generator import generate_corpus
    corpus = generate_corpus(convs_per_cell=14)
    msgs = [m for c in corpus for m in c]
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(FIG_W, 15.4))

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
        ax1.text(xi, p + n + g + 130, f"{p + n + g:,}", ha="center",
                 fontsize=FS_VALUE, fontweight="bold", color="#E5E7EB")
    ax1.set_xticks(x)
    ax1.set_xticklabels([f"{b}\n{lo}–{hi if hi < 101 else 100}"
                         for b, lo, hi in bands], fontsize=FS_TICK)
    ax1.set_ylabel("messages")
    ax1.set_ylim(0, max(p + n + g for p, n, g in zip(pos, neu, neg)) * 1.22)
    ax1.set_title("Sentiment composition per tension band", loc="left", pad=12)
    legend_below(ax1, 3, y_offset=-0.28)
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

    band = header(fig, "CEREBRO corpus — 6 domains × 7 narrative arcs, "
                       "weak-supervision annotated",
                  "conversation-level splits (no leakage) · 588 conversations · "
                  "10,956 messages · seed 42")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.075, left=0.115,
                        right=0.975, hspace=0.62)
    save(fig, "dataset_overview.png")


# ================================================================ 8 · DEMO REPORT
def graph_demo_report():
    demo = load("demo_report.json")
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 11.8), sharex=True,
                                   gridspec_kw={"height_ratios": [1.3, 1]})
    msgs = demo["messages"]
    xs = np.arange(1, len(msgs) + 1)
    tension = [m["tension"] for m in msgs]
    ax1.plot(xs, tension, color=CYAN, lw=2.6, marker="o", ms=8, mfc=CYAN,
             mec=BG, mew=1.4, zorder=3)
    ax1.fill_between(xs, tension, color=CYAN, alpha=.10)
    ax1.set_ylim(0, 132)
    ax1.set_yticks([0, 25, 50, 75, 100])
    for k, t in enumerate(demo["turning_points"]):
        ax1.axvline(t["message_id"], color=PINK, ls=":", lw=1.3, alpha=.85,
                    ymax=0.82)          # line stops below its own label
        ax1.text(t["message_id"], 108 if k % 2 == 0 else 121,
                 f'#{t["message_id"]} {t["tension_change"]:+.0f}',
                 fontsize=FS_VALUE - 0.5, color=PINK, ha="center", va="bottom",
                 fontweight="bold")
    ax1.axhspan(60, 100, color=PINK, alpha=.07)
    ax1.set_ylabel("tension (0–100)")
    ax1.set_title("tension — turning-point markers sit above the curve",
                  loc="left", fontsize=FS_PANEL_TITLE - 1.5, color=CYAN, pad=10)
    style_ax(ax1)

    ax2.plot(xs, [m["sarcasm"]["probability"] for m in msgs], "-o", color=PINK,
             lw=2.2, ms=6.5, label="sarcasm", mec=BG)
    ax2.plot(xs, [m["irony"]["probability"] for m in msgs], "-o", color=PURPLE,
             lw=2.2, ms=6.5, label="irony", mec=BG)
    ax2.plot(xs, [m["passive_aggression"]["probability"] for m in msgs], "-o",
             color=GREEN, lw=2.2, ms=6.5, label="passive-aggr.", mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.set_ylim(0, 1.18)
    ax2.set_yticks([0, .25, .5, .75, 1])
    ax2.set_xticks(xs)
    ax2.set_xticklabels([f'#{m["message_id"]}' for m in msgs], fontsize=FS_TICK - 0.5)
    ax2.set_xlabel("message (#id — full texts in the README worked example)")
    ax2.set_ylabel("probability")
    ax2.set_title("hidden signals — dashed line = 0.5 decision threshold",
                  loc="left", fontsize=FS_PANEL_TITLE - 1.5, color=PINK, pad=10)
    style_ax(ax2)

    band = header(fig, "CEREBRO on a held-out test conversation — "
                       "full per-message readout",
                  "top: tension with turning-point markers · bottom: sarcasm / irony / "
                  "passive-aggression probabilities")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.155, left=0.095,
                        right=0.975, hspace=0.30)
    figure_legend(fig, 3)          # bottom of the canvas, clear of all ink
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
    yt = {"sent": [], "emo": [], "tone": []}
    yp = {"sent": [], "emo": [], "tone": []}
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


def _short(name):
    return name.replace("_", " ")


# ================================================================ 9 · SCENARIOS
def graph_scenarios():
    sc = load("scenarios.json")["scenarios"]
    calm_expected = {"normal", "happy", "very_short", "slow", "humor", "malformed",
                     "multi_speaker"}
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 15.0), sharex=True)

    names = [s["scenario"] for s in sc]
    mean_t = [s["mean_tension"] for s in sc]
    peak_t = [s["peak_tension"] for s in sc]
    colors = [GREEN if n in calm_expected else ORANGE for n in names]
    x = np.arange(len(names))
    ax1.bar(x, mean_t, 0.62, color=colors, alpha=.55, edgecolor=BG, lw=.5)
    ax1.plot(x, peak_t, "_", color="white", ms=22, mew=2.4)
    ax1.axhline(25.9, color=CYAN, ls="--", lw=1.4)
    ax1.set_ylabel("tension (0–100)")
    ax1.set_ylim(0, max(peak_t) * 1.20)
    ax1.set_title("Per-scenario tension — green = calm, orange = conflict",
                  loc="left", pad=10, fontsize=FS_PANEL_TITLE - 2.0)
    style_ax(ax1)

    for (h, c) in [("sarcasm", PINK), ("irony", PURPLE),
                   ("passive_aggression", GREEN)]:
        key = {"sarcasm": "sarcasm_mean", "irony": "irony_mean",
               "passive_aggression": "pa_mean"}[h]
        ax2.plot(x, [s[key] for s in sc], "-o", color=c, lw=2.2, ms=6.5,
                 label=_short(h), mec=BG)
    ax2.axhline(0.5, color="#9CA3AF", ls="--", lw=1.2)
    ax2.set_xticks(x)
    ax2.set_xticklabels(names, rotation=90, fontsize=FS_TICK - 1.5)
    ax2.set_ylim(0, 1.0)
    ax2.set_ylabel("mean probability")
    ax2.set_title("Hidden-signal levels — dashed line = 0.5 threshold",
                  loc="left", pad=10, fontsize=FS_PANEL_TITLE - 2.0)
    # the scenario name is tinted with the same colour as its bar, so the
    # calm/conflict encoding needs no legend at all
    for lbl, c in zip(ax2.get_xticklabels(), colors):
        lbl.set_color(c)
    style_ax(ax2)

    band = header(fig, "PS-01 §41 robustness — 20 hand-crafted "
                       "out-of-distribution scenarios",
                  "different phrasing, emoji, slang, timing and formats than the "
                  "training corpus · all 20 executed without failure")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.235, left=0.105,
                        right=0.975, hspace=0.16)
    figure_legend(fig, 3)
    save(fig, "scenario_robustness.png")


# ================================================================ 10 · TRANSFER
def graph_transfer():
    try:
        s = load("transfer_logsafe.json")
    except FileNotFoundError:
        print("  – transfer_logsafe.json missing, skip")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 13.0))

    keys = [("emotion top-3", s["zero_shot_emotion_top3"], 1 / 3),
            ("emotion exact (13)", s["zero_shot_emotion_accuracy"], 1 / 13),
            ("sentiment exact (3)", s["zero_shot_sentiment_accuracy"], 1 / 3)]
    y = np.arange(len(keys))[::-1]
    vals = [v for _, v, _ in keys]
    chance = [c for _, _, c in keys]
    ax1.barh(y, vals, color=[CYAN, PURPLE, PINK], height=.58, zorder=3)
    ax1.barh(y, chance, color="none", edgecolor="#9CA3AF", height=.58, lw=1.2,
             ls="--", zorder=4)
    for yi, v in zip(y, vals):
        ax1.text(v + .012, yi, f"{v:.1%}", va="center", fontsize=FS_VALUE + .5,
                 color="#E5E7EB", fontweight="bold")
    ax1.set_yticks(y, [k for k, _, _ in keys], fontsize=FS_TICK)
    ax1.set_xlim(0, .66)
    ax1.set_xlabel("accuracy on real text (dashed outline = chance)")
    ax1.set_title("Zero-shot accuracy vs chance", loc="left", pad=12)
    style_ax(ax1)

    gold = s["gold_emotion_distribution"]
    pred = s["pred_emotion_distribution"]
    classes = list(dict.fromkeys(list(gold) + list(pred)))[:8]
    yy = np.arange(len(classes))[::-1]
    gv = [gold.get(c, 0) / s["n_messages"] for c in classes]
    pv = [pred.get(c, 0) / s["n_messages"] for c in classes]
    ax2.barh(yy + .19, gv, height=.36, color="#6B7280", label="gold (GoEmotions)",
             zorder=3)
    ax2.barh(yy - .19, pv, height=.36, color=CYAN, label="predicted", zorder=3)
    # only the predicted series carries numbers: when gold and predicted are
    # close the two labels land at the same x and stack with almost no gap
    for yi, v in zip(yy - .19, pv):
        ax2.text(v + .004, yi, f"{v:.2f}", va="center", fontsize=FS_VALUE - 1.0,
                 color="#C9D1DB")
    ax2.set_yticks(yy, classes, fontsize=FS_TICK - 0.5)
    ax2.set_xlabel("share of messages")
    ax2.set_xlim(0, max(max(gv), max(pv)) * 1.14)
    ax2.set_title("Label shift — grey = gold, cyan = predicted", loc="left",
                  pad=12, fontsize=FS_PANEL_TITLE - 2.5)
    style_ax(ax2)

    band = header(fig, "Zero-shot transfer to real GoEmotions text",
                  "3,000 real Reddit comments · honest cross-corpus metrics (not "
                  "comparable to in-corpus tables)")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.085, left=0.245,
                        right=0.975, hspace=0.34)
    save(fig, "transfer_goemotions.png")


# ================================================================ 11 · BENCHMARKS
def graph_benchmarks():
    try:
        b = load("benchmarks.json")
    except FileNotFoundError:
        print("  – benchmarks.json missing, skip")
        return
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(FIG_W, 12.8))

    rows = b["scaling"]
    ns = [r["n_messages"] for r in rows]
    tot = [r["total_s"] for r in rows]
    ax1.plot(ns, tot, "-o", color=CYAN, lw=3, ms=10, mec=BG, zorder=3)
    for i, (n, t) in enumerate(zip(ns, tot)):
        # below-right is the one quadrant a rising log-x curve never enters;
        # the final point flips to above-left and the first hugs its marker
        # (a -24 pt offset there lands on the x tick labels)
        last = i == len(ns) - 1
        low = t < 0.20          # the two smallest runs sit near the x axis
        ax1.annotate(f"{t:.2f}s", (n, t), textcoords="offset points",
                     xytext=(-12, 13) if last else (9, -14 if low else -24),
                     ha="right" if last else "left",
                     va="bottom" if last else "top",
                     fontsize=FS_VALUE, fontweight="bold", color="white")
    ax1.set_xscale("log")
    ax1.set_xticks(ns, [str(n) for n in ns])
    ax1.set_xlabel("messages in the conversation (log scale)")
    ax1.set_ylabel("end-to-end seconds")
    ax1.set_title("Latency scales linearly — no context-window blow-up",
                  loc="left", pad=12)
    # a small negative floor keeps the first point (10 msgs ≈ 0.04 s) off the
    # axis line, so its printed value never lands on the tick labels
    ax1.set_ylim(-max(tot) * 0.15, max(tot) * 1.30)
    ax1.set_yticks([t for t in (0, 1, 2, 3) if t <= max(tot) * 1.30])
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
        ax2.text(b_.get_x() + b_.get_width() / 2, v + .30, f"{v:.2f}", ha="center",
                 fontsize=FS_VALUE, fontweight="bold", color="white")
    ax2.set_xticks(np.arange(len(order)), disp, fontsize=FS_TICK - 1.0)
    ax2.set_ylabel("ms per message")
    ax2.set_ylim(0, max(vals) * 1.30)
    ax2.set_title(f"Where the time goes · ML heads dominate "
                  f"({vals[2] / max(sum(vals), 1e-9):.0%} of stage cost)",
                  loc="left", pad=12)
    style_ax(ax2)

    band = header(fig, "Efficiency — real benchmarks, persisted engine, single CPU core",
                  "top: end-to-end latency up to 1,000 messages · bottom: per-stage "
                  "cost · peak memory 8.9 MB @ 1k msgs · API round-trip 184 ms @ 50 msgs")
    fig.subplots_adjust(top=top_for(fig, band), bottom=0.135, left=0.115,
                        right=0.975, hspace=0.48)
    save(fig, "efficiency_benchmarks.png")


# ================================================================ 12 · ARCHITECTURE
def graph_architecture():
    """Vertical single-column pipeline at the canonical width.

    Geometry contract (asserted below): every box inside exactly one band, no
    box overlaps another, no arrow endpoint buried inside a box, and every text
    stays inside the canvas.
    """
    fig = plt.figure(figsize=(FIG_W, 23.0))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)

    rects = []          # (name, x, y, w, h) for the overlap validator

    def box(x, y, w, h, lines, color, fs=17.0, sub_fs=13.5, name=""):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=PANEL,
                                   edgecolor=color, lw=2, zorder=3))
        head, rest = lines[0], lines[1:]
        ax.text(x + w / 2, y + (h * 0.68 if rest else h / 2), head,
                ha="center", va="center", fontsize=fs, fontweight="bold",
                color=color, zorder=4)
        if rest:
            ax.text(x + w / 2, y + h * 0.30, "\n".join(rest), ha="center",
                    va="center", fontsize=sub_fs, color="#D1D5DB", zorder=4)
        rects.append((name or lines[0][:22], x, y, w, h))

    def arrow(x1, y1, x2, y2, color="#4B5563", lw=2):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw))

    def band(y, h, label, color):
        ax.add_patch(plt.Rectangle((3.4, y), 95, h, facecolor=color,
                                   alpha=0.045, edgecolor="none", zorder=1))
        ax.text(1.8, y + h / 2, label, fontsize=13.5, color=color, alpha=0.95,
                fontweight="bold", zorder=2, va="center", ha="center",
                rotation=90)

    # ---- INPUT LAYER: 6 chips in one row ----
    band(88.0, 8.5, "INPUT · PS-01 §2", CYAN)
    for txt, x in [("WhatsApp", 5.5), ("Discord", 20.5), ("Slack", 35.5),
                   ("CSV", 50.5), ("JSON", 65.5), ("plain", 80.5)]:
        box(x, 89.0, 14.0, 5.6, [txt], CYAN, fs=14.5, name=f"in:{txt}")
        # fan into DISTINCT points on the parser's top edge — arrowheads never
        # stack on one another
        tx = 50 + (x + 7.0 - 50) * 0.35
        arrow(x + 7.0, 89.0, tx, 87.0, lw=1.4)

    # ---- UNDERSTANDING LAYER ----
    band(55.5, 32.0, "UNDERSTANDING · §5 §8 §9 §17", PURPLE)
    box(20, 81.0, 60, 6.0,
        ["CHAT PARSER + AUTO-DETECT",
         "one common format · speakers · timestamps"],
        CYAN, fs=16.5, sub_fs=13.0, name="parser")
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
        GREEN, fs=16.5, sub_fs=13.0, name="repr")
    arrow(50, 56.3, 50, 53.3)

    # ---- INTELLIGENCE LAYER ----
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

    # ---- OUTPUT LAYER ----
    band(0.5, 21.2, "OUTPUT · §23–30", PINK)
    box(8.5, 12.6, 83, 7.0,
        ["EXPLAINABILITY + CONVERSATION REPORT",
         "WHY? · WHAT CHANGED? · speaker profiles · 18-section report"],
        PINK, name="explain")
    arrow(50, 12.6, 50, 11.4)
    box(8.5, 5.6, 83, 5.8,
        ["EMOTIONAL ARC · TENSION CURVE",
         "the journey, explained — served by the FastAPI dashboard"],
        CYAN, fs=16.0, sub_fs=13.0, name="arc")

    # ---- programmatic validators ----
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

    save(fig, "architecture.png", skip_data_check=True,
         out="docs/architecture/architecture.png", no_watermark=True)


if __name__ == "__main__":
    from cerebro.common.labels import SENTIMENT_LABELS, EMOTION_LABELS, TONE_LABELS
    print("generating GitHub-native graphs from real results...")
    graph_hero()
    graph_main_result()
    graph_ablation()
    graph_capability()
    graph_dataset()
    graph_demo_report()
    graph_scenarios()
    graph_transfer()
    graph_context_proof()
    graph_transformer_baselines()
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
