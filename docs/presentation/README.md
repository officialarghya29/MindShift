# CEREBRO — presentation deck

Ten slides following PS-01 §53, aimed at a hackathon/pitch audience.

| File | What it is |
|---|---|
| `slides.html` | the deck — open it in any browser (no build step, no CDN, no network) |
| `slides.pdf` | the same deck rendered to PDF for sharing (headless Chrome) |
| `facts.json` | **every number that appears on a slide**, machine-readable |

## Why a generated deck

The blueprint's final development principle says: *do not claim results that were
not measured.* A slide deck is where that rule is easiest to break — someone
types "97% accuracy" into a text box and it never gets re-checked when the model
changes.

So the deck is not written, it is **rendered**:

```bash
python scripts/make_deck.py
```

[`scripts/make_deck.py`](../../scripts/make_deck.py) reads
`evaluation/results/*.json` — the same artifacts CI regenerates on every push —
and refuses to start if one is missing. Figures are referenced in place from
`assets/graphs/`, so regenerating a graph updates the deck automatically. There
is no path by which a slide can disagree with the evaluation that produced it,
because the slide has no independent source of numbers.

`facts.json` exists for the same reason: so a reviewer can diff the deck's claims
against the results without reading HTML.

## Rendering the PDF

```bash
google-chrome --headless=new --disable-gpu --no-sandbox --no-pdf-header-footer \
  --print-to-pdf=docs/presentation/slides.pdf \
  "file://$PWD/docs/presentation/slides.html"
```

## Slide map (PS-01 §53)

| # | Slide | Section |
|---|---|---|
| 1 | Text carries more than words | §50 problem framing |
| 2 | Message-level analysis misses the conversation | §9, §14 |
| 3 | CEREBRO — the solution | §2, §47 |
| 4 | Architecture | §5, §54 |
| 5 | Core innovation: the context proof | §11, §12, §36 |
| 6 | Evaluation design: lexical + transformer baselines | §9, §36 |
| 7 | Results | §33, §34, §38 |
| 8 | Real human text is the honest test | §35, §39 |
| 9 | Explainable, robust, fast | §28–30, §46 |
| 10 | Applications & future | §53 |
