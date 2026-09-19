"""Independent audit of the committed figures — a second opinion.

`evaluation/make_graphs.py` validates figures *in process* while it draws them
(text boxes, data ink, line segments). This script checks the PNGs that were
actually written to disk, so a bug in the validators themselves cannot hide a
defect:

  1. edge ink — bright pixels touching the outer frame mean text was clipped by
     the canvas edge (seven titles were silently clipped before this existed);
  2. canvas width — no figure may be drawn wider than the canonical 9.6 in, so
     GitHub's ~830 px README column can never scale the fonts below the target
     (see the README's "Figure legibility policy"). Narrower is fine: the
     3-way confusion matrix is deliberately 7.4 in for a 0.75x display scale.
  3. near-empty / over-inked panels — catches a collapsed layout or a runaway
     fill that still passes the geometric checks.

Run:  python scripts/audit_figures.py       (exit 1 on any failure)
"""
from __future__ import annotations

import glob
import sys

import numpy as np
from PIL import Image

GITHUB_COLUMN_PX = 830
CANONICAL_WIDTH_IN = 9.6
DPI = 150
MAX_W = int(CANONICAL_WIDTH_IN * DPI)   # 1440 px — wider means smaller text on GitHub
MIN_W = 1050                            # sanity floor for a README-width chart
EDGE_LUM = 110          # "ink" threshold against the #0B0F1A background
MIN_INK_PCT, MAX_INK_PCT = 1.0, 45.0


def paths() -> list[str]:
    return sorted(glob.glob("assets/graphs/*.png")) + \
        ["docs/architecture/architecture.png"]


def audit(path: str) -> list[str]:
    problems: list[str] = []
    img = Image.open(path)
    a = np.asarray(img.convert("RGB")).astype(float)
    h, w, _ = a.shape
    lum = 0.2126 * a[..., 0] + 0.7152 * a[..., 1] + 0.0722 * a[..., 2]

    frame = np.concatenate([lum[:2, :].ravel(), lum[-2:, :].ravel(),
                            lum[:, :2].ravel(), lum[:, -2:].ravel()])
    n_edge = int((frame > EDGE_LUM).sum())
    if n_edge:
        problems.append(f"{n_edge} ink pixels on the canvas frame (clipped text?)")

    if w > MAX_W:
        problems.append(f"width {w}px exceeds the {MAX_W}px cap "
                        f"({CANONICAL_WIDTH_IN}in @ {DPI}dpi) — text will be "
                        f"scaled below the legibility target")
    elif w < MIN_W:
        problems.append(f"width {w}px is below the {MIN_W}px floor — "
                        f"suspiciously narrow for a full-width README embed")

    ink_pct = float((np.abs(a - np.array([11, 15, 26])).sum(axis=2) > 40).mean()) * 100
    if not (MIN_INK_PCT <= ink_pct <= MAX_INK_PCT):
        problems.append(f"ink coverage {ink_pct:.1f}% outside "
                        f"[{MIN_INK_PCT}, {MAX_INK_PCT}]% (collapsed or runaway layout)")

    print(f"  {'FAIL' if problems else ' OK '} {path.split('/')[-1]:30s} "
          f"{w}x{h}  scale {GITHUB_COLUMN_PX / w:.2f}x  ink {ink_pct:5.1f}%")
    return problems


def main() -> int:
    files = paths()
    print(f"auditing {len(files)} committed figures\n")
    failures: list[str] = []
    for p in files:
        for problem in audit(p):
            failures.append(f"{p}: {problem}")

    print()
    if failures:
        for f in failures:
            print("FAIL:", f)
        print(f"\n✗ figure audit FAILED ({len(failures)} problem(s))")
        return 1
    print(f"✓ figure audit clean — {len(files)} figures, zero clipped text")
    return 0


if __name__ == "__main__":
    sys.exit(main())
