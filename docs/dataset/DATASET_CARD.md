# CEREBRO Dataset Card — v1.0.0

## Motivation

PS-01 §3 calls for a unified conversational corpus covering sentiment, emotion, tone,
sarcasm, irony, passive-aggression and tension with a conversation-level split.
CEREBRO v1.0 uses a **generated corpus** so that:

1. every label is exact at creation time (no crowdsourced disagreement to model away);
2. the train/val/test split is **by conversation** — zero context leakage (§3 requirement);
3. the whole corpus is reproducible from a single command with a fixed seed;
4. no private chat data is ever collected or shipped.

## Construction

- **Domains (6):** work project, college team, startup founders, flatmates,
  customer support, old friends — each with a 2-speaker cast and seed topics.
- **Arcs (7):** calm, positive, friction, escalation (+de-escalation tail),
  sarcasm, passive, mixed.
- **Turn templates:** three arousal bands (calm / strained / hot) per domain;
  the generator composes arcs from these bands with no-template-repetition
  constraints per conversation.
- **Weak supervision:** each template carries its full label tuple
  `(sentiment, emotion, tone, sarcasm, irony, passive_aggression, tension)`
  applied at generation time (§4 label space).
- **Realism noise:** tension gets gaussian noise (σ = 3.5); 1.5% of binary labels
  are flipped (controlled annotator disagreement); 25% of turns reference the
  domain seed topic in-text.
- **Full class coverage:** all **13 emotion** classes (including *surprise*) and
  all **14 tone** classes (including *humorous*) carry non-zero support, in every
  arousal band. This was not always true — those two classes originally had zero
  training examples, which made the heads structurally *incapable* of emitting
  them. The dataset audit caught it (`class_imbalance.*.classes_absent_from_corpus`)
  and the fix was to author covering turns, not to hide the dead classes.

## Exact statistics (seed 42)

| Property | Value |
|---|---|
| Conversations | 588 (6 × 7 × 14) |
| Messages | 10,956 |
| Splits (by conversation) | 411 / 88 / 89 |
| Messages per split | 7,748 / 1,560 / 1,648 |
| Sentiment | 6,110 positive · 2,474 neutral · 2,372 negative |
| Positive rates | sarcasm 3.81% · irony 3.61% · PA 3.91% |
| Tension | mean 26.1 · range 0–100 |
| Emotion classes with support | 13 / 13 |
| Tone classes with support | 14 / 14 |
| Distinct normalised message texts | 62 (reuse ×176.7 — disclosed, not hidden) |
| Annotator agreement (κ / α, design vs released) | 1.00 categorical · 0.915–0.944 hidden signals |

## Annotator agreement — what is and is not measured

Two annotators exist over every label: **A1** is the template design (the schema
intent), **A2** is the released label (A1 plus the injected 1.5% disagreement).
Cohen's κ and Krippendorff's nominal α between them are computed per head by
[`evaluation/audit_dataset.py`](../../evaluation/audit_dataset.py):

| Head | Observed agreement | Cohen's κ | Krippendorff α |
|---|---|---|---|
| sentiment · emotion · tone | 1.0000 | 1.000 | 1.000 |
| sarcasm | 0.9944 | 0.918 | 0.918 |
| irony | 0.9963 | 0.944 | 0.944 |
| passive-aggression | 0.9941 | 0.915 | 0.915 |

**Read this carefully:** A2 is *simulated*, so these numbers quantify the injected
disagreement rate and the resulting label noise — they are **not** a substitute for
a human multi-annotator study, which remains on the roadmap and is not claimed.
The categorical heads show κ = 1.00 because the generator only injects flips into
binary labels.

## Unified schema (per message)

```
conversation_id · message_id · speaker_id · timestamp · text · platform
sentiment · emotion · tone · sarcasm · irony · passive_aggression · tension
escalation (derived: tension ≥ 60) · topic
```

## Known limitations

- Template composition → classification heads saturate (see README honesty note);
  the discriminative benchmarks are the hidden-signal heads, tension regression
  and calibration metrics. PS-01 §3's context question is answered by the
  controlled probe corpus instead (`cerebro/data/context_probes.py`) — this corpus
  *cannot* answer it, because every utterance carries a fixed label there.
- Annotation agreement is measured against a *simulated* second annotator.
- Two-speaker conversations only in v1 (multi-party is on the roadmap).
- English only.

## Extension path with public datasets

All of the following can be mapped onto the unified schema above. **Check each
license before redistribution; the loaders below are provided as adapters only.**

| Dataset | Covers | Notes |
|---|---|---|
| GoEmotions (Google Research) | 27 emotions | Reddit comments; per-message emotion labels; map to the 13-class space |
| SARC (Khodak et al.) | sarcasm | self-labeled /-s marks on Reddit; train sarcasm head |
| iCas / Sarcasm Corpus v2 (ORCA) | sarcasm, irony | broadcast news + Twitter; check UIMA licence terms |
| DailyDialog (LRDC) | dialogue act, emotion, sentiment | multi-turn English dialogs; conversation-level splits supported |
| EmpatheticDialogues (Facebook) | emotion in dialogue | 25 emotion labels; map to the 13-class space |

Adapter contract: implement `load() -> list[list[dict]]` matching the schema
above and pass it to `split_conversations()`; the whole training/evaluation
stack accepts any schema-compliant corpus.
