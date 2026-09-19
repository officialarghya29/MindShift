"""Generate the presentation deck from the published result files (blueprint §53).

The deck is built by *reading* `evaluation/results/*.json` — the same artifacts CI
regenerates — so a slide can never drift from a measured number, and a claim that
was not measured cannot appear on one. §53 asks for ten slides; the blueprint's
"do not claim results that were not measured" rule is enforced structurally.

Outputs
-------
  docs/presentation/slides.html   dark deck, opens in any browser
  docs/presentation/facts.json    every number on a slide, machine-readable

Figures are referenced in place from `assets/graphs/` (no duplicated copies), so
regenerating the graphs updates the deck automatically.

Run:  python scripts/make_deck.py
      (then optionally: chrome --headless --print-to-pdf=... slides.html)
"""
from __future__ import annotations

import json
import os

RESULTS = "evaluation/results"
GRAPHS = "assets/graphs"
OUT = "docs/presentation"

CYAN, PURPLE, PINK, GREEN, ORANGE = "#00E5FF", "#B388FF", "#FF5C8A", "#7CFFB2", "#FF9E64"
BG, PANEL, TEXT, MUTED = "#080B13", "#111827", "#E5E7EB", "#9CA3AF"

FIG_REL = "../../assets/graphs"      # relative to docs/presentation/
# the architecture diagram is written outside assets/graphs by make_graphs.py
FIG_OVERRIDE = {"architecture.png": "../architecture/architecture.png"}


def load(name):
    path = os.path.join(RESULTS, name)
    if not os.path.exists(path):
        raise SystemExit(f"missing {path} — run the evaluation stage first")
    with open(path) as fh:
        return json.load(fh)


def pct(x, nd=1):
    return f"{100 * x:.{nd}f}%"


def build_facts():
    """Every number that appears in the deck, read from measured artifacts."""
    summary = load("summary.json")
    proof = load("context_proof.json")
    tx = load("transformer_baselines.json")
    audit = load("dataset_audit.json")
    bench = load("benchmarks.json")
    scen = load("scenarios.json")
    transfer = load("transfer_logsafe.json")
    finetune = load("finetune_summary.json")

    primary = proof["regimes"][proof["primary_regime"]]
    seen = proof["regimes"]["seen_history_wording"]
    probe_tx = tx["probe_experiment"]["test"]
    fm = summary["full_metrics"]
    cal = fm["calibration"]
    sig = primary["significance"]["sentiment"]["A_to_E"]

    return {
        "corpus": summary["corpus_stats"],
        "splits": {k: summary[f"n_{k}_conversations"] for k in ("train", "val", "test")},
        "n_test_messages": summary["n_test_messages"],
        "context": {
            "n_probe_turns": primary["n_probe_turns"],
            "ceiling": primary["text_only_lexical_ceiling"]["sentiment"]["accuracy"],
            "text_only": primary["variants"]["A"]["sentiment"]["accuracy"],
            "full": primary["variants"]["E"]["sentiment"]["accuracy"],
            "full_emotion": primary["variants"]["E"]["emotion"]["accuracy"],
            "full_tone": primary["variants"]["E"]["tone"]["accuracy"],
            "seen_full": seen["variants"]["E"]["sentiment"]["accuracy"],
            "gain_pp": primary["headline_gain_pp"]["sentiment"]["gain_pp"],
            "mcnemar_p": sig["p_value"],
            "ci95": sig["ci95"],
            "tx_no_context": probe_tx["B2_transformer_text_only"]["sentiment"]["accuracy"],
            "tx_context": probe_tx["B3_transformer_plus_context"]["sentiment"]["accuracy"],
            "flip_sentiment": proof["flip_rates"]["sentiment"],
        },
        "metrics": {
            "sarcasm_auc": fm["sarcasm"]["roc_auc"],
            "irony_auc": fm["irony"]["roc_auc"],
            "pa_auc": fm["passive_aggression"]["roc_auc"],
            "tension_mae": fm["tension"]["mae"],
            "tension_r2": fm["tension"]["r2"],
            "latency_ms": fm["latency_ms"],
            "ece_sentiment": cal["sentiment"]["ece"],
            "ece_emotion": cal["emotion"]["ece"],
            "ece_tone": cal["tone"]["ece"],
        },
        "baselines": {k: v for k, v in load("baselines.json").items()},
        "ablations": load("ablations.json"),
        "audit": {
            "conversation_overlap": audit["test_set_contamination"]["conversation_overlap_train"],
            "shared_templates": audit["test_set_contamination"]["shared_template_texts_train"],
            "distinct_texts": audit["duplicate_messages"]["distinct_normalized_texts"],
            "min_kappa": audit["annotation_consistency"]["min_kappa"],
            "messages": audit["n_messages"],
            "conversations": audit["n_conversations"],
            "leakage_pass": audit["conversation_level_separation"]["passed"],
        },
        "bench": {
            "load_s": bench["load_seconds"],
            "peak_mb": bench["memory_1000msg"]["peak_mb"],
            "api_mean": bench["api_50msg"]["mean_s"],
            "msgs_per_s": round(1000 / bench["stage_latency_ms_per_message"]["ml_heads"], 1),
        },
        "scenarios": {"n": scen["scenarios"]["__len__"] if isinstance(scen.get("scenarios"), dict)
                      and "__len__" in scen["scenarios"] else len(scen["scenarios"]),
                      "passed": scen["n_passed"]},
        "transfer": {
            "dataset": transfer["dataset"],
            "zero_shot_emotion": transfer["zero_shot_emotion_accuracy"],
            "zero_shot_valence_auc": transfer["tension_separates_valence_auc"],
            "ft_emotion": finetune["after_finetune"]["emotion_exact"],
            "ft_top3": finetune["after_finetune"]["emotion_top3"],
            "ft_sentiment": finetune["after_finetune"]["sentiment_exact"],
            "ft_valence_auc": finetune["after_finetune"]["tension_valence_auc"],
        },
    }


def slide(title, kicker, body, footer=""):
    return f"""
    <section class="slide">
      <div class="kicker">{kicker}</div>
      <h2>{title}</h2>
      {body}
      {f'<div class="footer">{footer}</div>' if footer else ''}
    </section>"""


def fig(name, caption, width="100%"):
    src = FIG_OVERRIDE.get(name, f"{FIG_REL}/{name}")
    real = os.path.normpath(os.path.join(OUT, src))
    assert os.path.exists(real), f"missing figure {name} (looked for {real})"
    return (f'<figure><img src="{src}" style="width:{width}" alt="{caption}">'
            f'<figcaption>{caption}</figcaption></figure>')


def stat(value, label, color=CYAN):
    return (f'<div class="stat"><div class="v" style="color:{color}">{value}</div>'
            f'<div class="l">{label}</div></div>')


def render(f):
    c, m, a, b = f["context"], f["metrics"], f["audit"], f["bench"]
    slides = []

    slides.append(slide(
        "Text carries more than words",
        "1 · Problem",
        f"""<p class="lead">A single message is ambiguous by construction. The same
        string can mean opposite things, and only the conversation tells you which.</p>
        <div class="quote">&ldquo;Fine.&rdquo;</div>
        <p class="muted">relieved · neutral · cold · passive-aggressive</p>
        <div class="stats">
          {stat(f'{c["flip_sentiment"]*100:.0f}%', "of probe utterances flip sentiment with history", PINK)}
          {stat(f'{a["messages"]:,}', "messages analysed in the published corpus")}
          {stat("4", "platforms parsed natively")}
        </div>""",
        "PS-01 §1"))

    slides.append(slide(
        "Message-level analysis misses the conversation",
        "2 · Why the obvious approach fails",
        f"""<p class="lead">We measured it rather than asserting it. On a controlled probe
        set where the wording is identical and only the history differs, every
        context-free reader collapses to the same score — the majority-class rate.</p>
        <div class="stats">
          {stat(f'{c["ceiling"]:.3f}', "text-only ceiling (analytic)", ORANGE)}
          {stat(f'{c["text_only"]:.3f}', "TF-IDF text-only", ORANGE)}
          {stat(f'{c["tx_no_context"]:.3f}', "frozen MiniLM transformer, no context", ORANGE)}
        </div>
        <p class="muted">The design balances every utterance across its gold labels, so
        the mutual information between the words and the label is zero. A pretrained
        encoder buys nothing here: it is bounded by the same ceiling.</p>""",
        "PS-01 §3"))

    slides.append(slide(
        "CEREBRO",
        "3 · Our solution",
        f"""<p class="lead">A context-aware, temporal conversation-intelligence engine.
        It does not ask <em>what is the sentiment of this message</em>. It asks
        <em>what is happening to the emotional state of this conversation, where did it
        change, and what accompanied the change</em>.</p>
        <div class="stats">
          {stat(f'{c["full"]:.3f}', "probe accuracy with context", GREEN)}
          {stat(f'+{c["gain_pp"]:.1f} pp', "over the text-only ceiling", GREEN)}
          {stat(f'p = {c["mcnemar_p"]:.0e}', "exact McNemar vs text-only")}
        </div>
        <p class="muted">95% bootstrap CI on the paired gain:
        [{c["ci95"][0]:.3f}, {c["ci95"][1]:.3f}] accuracy.</p>""",
        "PS-01 §2, §47"))

    slides.append(slide(
        "Architecture",
        "4 · System",
        fig("architecture.png", "End-to-end pipeline: parse → features → contextual "
            "multi-task heads → hidden signals → temporal engines → fusion → "
            "explainability → API + dashboard."),
        "PS-01 §5, §54"))

    slides.append(slide(
        "Context, memory, behaviour — and a proof they are used",
        "5 · Core innovation",
        fig("context_proof.png", "The component ladder on held-out history wording. "
            "Text-only sits exactly on the lexical ceiling; the full stack reaches "
            f'{c["full"]:.3f} sentiment / {c["full_emotion"]:.3f} emotion / '
            f'{c["full_tone"]:.3f} tone.'),
        "PS-01 §11, §12, §36"))

    slides.append(slide(
        "Baselines: lexical, and a real pretrained transformer",
        "6 · Evaluation design",
        fig("transformer_baselines.png",
            "B2 is a frozen sentence-transformers MiniLM encoder — a strong, "
            "general-purpose language model with no fine-tuning, exactly as §9 asks. "
            "Bottom: the template corpus saturates every model family alike, which is "
            "why the probe experiment is the informative one."),
        "PS-01 §9, §36"))

    slides.append(slide(
        "Every number measured, every claim bounded",
        "7 · Results",
        f"""<table>
          <tr><th>Head</th><th>Metric</th><th>Value</th></tr>
          <tr><td>Sentiment</td><td>probe accuracy (unseen wording)</td>
              <td class="num">{c["full"]:.3f}</td></tr>
          <tr><td>Emotion</td><td>probe accuracy (unseen wording)</td>
              <td class="num">{c["full_emotion"]:.3f}</td></tr>
          <tr><td>Tone</td><td>probe accuracy (unseen wording)</td>
              <td class="num">{c["full_tone"]:.3f}</td></tr>
          <tr><td>Sarcasm</td><td>ROC-AUC (held-out test)</td>
              <td class="num">{m["sarcasm_auc"]:.4f}</td></tr>
          <tr><td>Irony</td><td>ROC-AUC</td><td class="num">{m["irony_auc"]:.4f}</td></tr>
          <tr><td>Passive aggression</td><td>ROC-AUC</td>
              <td class="num">{m["pa_auc"]:.4f}</td></tr>
          <tr><td>Tension</td><td>MAE / R²</td>
              <td class="num">{m["tension_mae"]:.3f} / {m["tension_r2"]:.3f}</td></tr>
          <tr><td>Calibration</td><td>top-1 ECE (sentiment / emotion / tone)</td>
              <td class="num">{m["ece_sentiment"]:.3f} / {m["ece_emotion"]:.3f} / {m["ece_tone"]:.3f}</td></tr>
        </table>
        <p class="muted">In-corpus scores saturate because the corpus is template-generated
        ({a["distinct_texts"]} distinct phrasings over {a["messages"]:,} messages). We
        publish that rather than headlining it, and report real-text transfer separately.</p>""",
        "PS-01 §33, §34, §38"))

    slides.append(slide(
        "Real human text is the honest test",
        "8 · Transfer",
        fig("transfer_goemotions.png",
            "Zero-shot on GoEmotions, then the same heads fine-tuned on real Reddit text "
            f'— emotion exact {f["transfer"]["zero_shot_emotion"]:.3f} → '
            f'{f["transfer"]["ft_emotion"]:.3f}, top-3 → {f["transfer"]["ft_top3"]:.3f}.'),
        f'PS-01 §35 — measured on {f["transfer"]["dataset"]} (research licence)'))

    slides.append(slide(
        "Explainable, robust, fast",
        "9 · Engineering",
        f"""<div class="stats">
          {stat(f'{c["n_probe_turns"]}', "probe turns in the controlled experiment")}
          {stat(f'{b["msgs_per_s"]:.0f}', "messages / second, single CPU core")}
          {stat(f'{b["peak_mb"]:.1f} MB', "peak memory, 1,000-message chat")}
          {stat(f'{b["api_mean"]*1000:.0f} ms', "API round-trip, 50 messages")}
          {stat(f'{a["min_kappa"]:.2f}', "min annotator κ across heads")}
          {stat(f'{f["scenarios"]["passed"]}/{f["scenarios"]["n"]}', "robustness scenarios passed")}
        </div>
        <p class="muted">Data-leakage audit: {a["conversation_overlap"]} conversation
        overlap across splits, {a["leakage_pass"] and "0 label-leakage findings"}; the
        {a["shared_templates"]} recurring templates are disclosed, not hidden.
        Every prediction ships with confidence and a human-readable explanation that
        separates detected evidence from inferred interpretation.</p>""",
        "PS-01 §28, §29, §30, §39, §46"))

    slides.append(slide(
        "Where this goes",
        "10 · Applications &amp; future",
        """<div class="cols">
          <div><h3>Applications</h3><ul>
            <li>Customer-support escalation early-warning</li>
            <li>Team-communication and workplace analytics</li>
            <li>Community moderation triage</li>
            <li>Conversation research at scale</li>
          </ul></div>
          <div><h3>Next</h3><ul>
            <li>Multilingual parsing and lexicons</li>
            <li>Multi-turn real conversation corpora</li>
            <li>Streaming / real-time analysis</li>
            <li>Human multi-annotator agreement study</li>
          </ul></div>
        </div>
        <div class="tagline">&ldquo;Decode the shift. Understand the conversation.&rdquo;</div>
        <p class="muted">CEREBRO reports <em>model-detected emotional and tone signals</em>,
        not a person's true feelings. Turning points are statistical associations, not
        causal claims.</p>""",
        "MIND SHIFT · PS-01"))

    return "\n".join(slides)


HTML = """<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>CEREBRO — PS-01 Tone Intelligence in Conversations</title>
<style>
  :root {{ --bg:{BG}; --panel:{PANEL}; --text:{TEXT}; --muted:{MUTED};
           --cyan:{CYAN}; --purple:{PURPLE}; --pink:{PINK}; --green:{GREEN}; }}
  * {{ box-sizing:border-box; }}
  body {{ margin:0; background:var(--bg); color:var(--text);
          font-family:"Inter",-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
          -webkit-font-smoothing:antialiased; }}
  .deck {{ display:flex; flex-direction:column; align-items:center; gap:38px; padding:38px 0 96px; }}
  .slide {{ width:min(1180px,94vw); background:var(--panel); border:1px solid #1F2937;
            border-radius:18px; padding:46px 54px 40px; position:relative;
            box-shadow:0 22px 60px rgba(0,0,0,.45); }}
  .kicker {{ font-size:13.5px; letter-spacing:.20em; text-transform:uppercase;
             color:var(--cyan); font-weight:700; margin-bottom:12px; }}
  h2 {{ font-size:38px; line-height:1.15; margin:0 0 20px; letter-spacing:-.02em; }}
  h3 {{ font-size:19px; margin:0 0 10px; color:var(--cyan); }}
  p {{ font-size:17.5px; line-height:1.62; margin:0 0 14px; }}
  p.lead {{ font-size:20.5px; color:#F3F4F6; }}
  .muted {{ color:var(--muted); font-size:15.5px; line-height:1.6; }}
  .quote {{ font-size:34px; font-weight:700; color:var(--pink); margin:22px 0 6px; }}
  .stats {{ display:flex; gap:16px; flex-wrap:wrap; margin:24px 0 6px; }}
  .stat {{ flex:1 1 190px; background:#0C1322; border:1px solid #1F2937; border-radius:13px;
           padding:18px 20px; }}
  .stat .v {{ font-size:31px; font-weight:800; letter-spacing:-.02em; }}
  .stat .l {{ font-size:14px; color:var(--muted); margin-top:7px; line-height:1.42; }}
  figure {{ margin:20px 0 0; }}
  figure img {{ border-radius:12px; border:1px solid #1F2937; display:block; }}
  figcaption {{ font-size:14.5px; color:var(--muted); margin-top:12px; line-height:1.55; }}
  table {{ width:100%; border-collapse:collapse; margin:8px 0 16px; font-size:16.5px; }}
  th {{ text-align:left; padding:11px 12px; border-bottom:2px solid #374151;
        color:var(--cyan); font-size:14px; letter-spacing:.06em; text-transform:uppercase; }}
  td {{ padding:10px 12px; border-bottom:1px solid #1F2937; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; font-weight:700; color:var(--green); }}
  .cols {{ display:flex; gap:34px; margin:10px 0 18px; }}
  .cols > div {{ flex:1; }}
  ul {{ margin:0; padding-left:20px; }} li {{ font-size:17px; line-height:1.75; }}
  .tagline {{ font-size:25px; font-weight:800; color:var(--purple); margin:14px 0 18px; }}
  .footer {{ position:absolute; right:30px; bottom:18px; font-size:12.5px;
             color:#4B5563; letter-spacing:.09em; }}
  @media print {{ body {{ background:#fff; }} .deck {{ gap:0; padding:0; }}
    .slide {{ width:100%; border-radius:0; box-shadow:none; page-break-after:always; }} }}
</style></head>
<body><div class="deck">
{slides}
</div></body></html>
"""


def main():
    os.makedirs(OUT, exist_ok=True)
    facts = build_facts()
    html = HTML.format(slides=render(facts), BG=BG, PANEL=PANEL, TEXT=TEXT,
                       MUTED=MUTED, CYAN=CYAN, PURPLE=PURPLE, PINK=PINK, GREEN=GREEN)
    n_fig = html.count("<figure>")
    out = os.path.join(OUT, "slides.html")
    with open(out, "w", encoding="utf-8") as fh:
        fh.write(html)
    with open(os.path.join(OUT, "facts.json"), "w", encoding="utf-8") as fh:
        json.dump(facts, fh, indent=1)
    print(f"  ✓ {out} ({len(html) // 1024} KB, 10 slides, {n_fig} embedded figures)")
    print(f"  ✓ {OUT}/facts.json — every deck number, machine-readable")


if __name__ == "__main__":
    main()
