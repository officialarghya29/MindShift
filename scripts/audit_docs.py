"""Docs integrity gate: every link and anchor in the README must resolve.

The README claims a docs-integrity check in its quality-gates table, so the check
has to exist and be runnable (`python scripts/audit_docs.py`). It verifies:

  1. every local image `src` and markdown link target exists on disk;
  2. every in-page `#anchor` points at a real heading, using GitHub's own slug
     algorithm (lowercase, drop punctuation, spaces → hyphens) so the check agrees
     with what the rendered page actually does;
  3. heading slugs are unique, which is required for (2) to be deterministic;
  4. every figure referenced by the README exists in `assets/graphs/`.

Exit code 0 = clean, 1 = at least one broken reference.
"""
from __future__ import annotations

import os
import re
import sys
import unicodedata

README = "README.md"
GRAPH_DIR = "assets/graphs"
# every markdown file we ship, not just the README — a wrong relative link in a
# docs/ subpage is the easiest kind of reference error to ship unnoticed
DOCS = [README, "AGENTS.md", "docs/dataset/DATASET_CARD.md",
        "docs/methodology/CONTEXT_TEMPORAL_THEORY.md",
        "docs/methodology/PS01_WORKFLOW.md",
        "docs/methodology/FINETUNING.md",
        "docs/architecture/ARCHITECTURE.md",
        "evaluation/results/README.md",
        "docs/presentation/README.md"]

LINK_RE = re.compile(r"\]\(([^)\s]+)\)")
IMG_SRC_RE = re.compile(r'<img[^>]+src="([^"]+)"')
HEADING_RE = re.compile(r"^(#{1,6})\s+(.*?)\s*$", re.MULTILINE)
FENCE_RE = re.compile(r"```.*?```", re.DOTALL)


def slug(title: str) -> str:
    """GitHub's heading-anchor algorithm (mirrors github-slugger).

    Order matters and is easy to get subtly wrong: emoji and punctuation are
    *removed*, not replaced, and then every whitespace character becomes a
    hyphen — one per character. So `## \U0001f680 Quickstart` slugs to
    `-quickstart` (the space the emoji left behind), which is exactly how the
    rendered page links to it. Stripping first would silently break every
    emoji-headed anchor in the file.
    """
    t = re.sub(r"`([^`]*)`", r"\1", title)                 # inline code → text
    t = re.sub(r"\*\*?([^*]*)\*+", r"\1", t)               # emphasis → text
    t = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", t)         # links → link text
    t = re.sub(r"<[^>]*>", "", t)
    t = "".join(c for c in t
                if c in "-_" or not unicodedata.category(c)[0] in "PS")
    t = t.lower()
    return re.sub(r"\s", "-", t)                          # one hyphen per space


def audit(readme: str = README) -> dict:
    if not os.path.exists(readme):
        return {"readme": readme, "headings": 0, "anchors": 0,
                "local_paths_checked": 0, "figures": 0,
                "problems": [], "skipped": True, "passed": True}
    text = open(readme, encoding="utf-8").read()
    body = FENCE_RE.sub("", text)                          # ignore code blocks

    problems = []

    # 1 · headings + slugs
    headings = [m.group(2) for m in HEADING_RE.finditer(body)]
    slugs = [slug(h) for h in headings]
    seen: dict[str, int] = {}
    for h, s in zip(headings, slugs):
        seen[s] = seen.get(s, 0) + 1
    dupes = sorted(s for s, n in seen.items() if n > 1 and s)
    if dupes:
        problems.append(f"duplicate heading slugs: {dupes}")
    slug_set = set(slugs)

    base = os.path.dirname(readme)

    # 2 · in-page anchors
    anchors = [t for t in LINK_RE.findall(body) if t.startswith("#")]
    for a in anchors:
        if a.lstrip("#") not in slug_set:
            problems.append(f"broken anchor: {a}")

    # 3 · local files (links + image srcs), resolved relative to this file
    targets = LINK_RE.findall(body) + IMG_SRC_RE.findall(body)
    rel_check = 0
    for t in targets:
        if t.startswith(("http://", "https://", "#", "mailto:")):
            continue
        path = t.split("#", 1)[0]
        if not path:
            continue
        if not os.path.exists(os.path.normpath(os.path.join(base, path))):
            problems.append(f"missing file: {path}")
        else:
            rel_check += 1

    # 4 · every <img> pointing into assets/graphs must exist
    figs = [s for s in IMG_SRC_RE.findall(body) if GRAPH_DIR in s]
    for f in figs:
        if not os.path.exists(f):
            problems.append(f"missing figure: {f}")

    return {"readme": readme, "headings": len(headings), "anchors": len(anchors),
            "local_paths_checked": rel_check, "figures": len(figs),
            "problems": problems, "passed": not problems, "skipped": False}


def main() -> int:
    all_problems = []
    tot = {"headings": 0, "anchors": 0, "local_paths_checked": 0, "figures": 0}
    scanned = 0
    for doc in DOCS:
        res = audit(doc)
        if res.get("skipped"):
            continue
        scanned += 1
        for k in tot:
            tot[k] += res[k]
        all_problems += [f"{doc}: {p}" for p in res["problems"]]

    print(f"docs integrity: {scanned} files · {tot['headings']} headings · "
          f"{tot['anchors']} anchors · {tot['local_paths_checked']} local paths · "
          f"{tot['figures']} figure embeds")
    if all_problems:
        print(f"\n✗ {len(all_problems)} broken reference(s):")
        for p in all_problems:
            print("   -", p)
        return 1
    print("✓ docs integrity clean — every link, anchor and figure resolves")
    return 0


if __name__ == "__main__":
    sys.exit(main())
