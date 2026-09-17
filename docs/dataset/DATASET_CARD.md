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

## Exact statistics (seed 42)

| Property | Value |
|---|---|
| Conversations | 588 (6 × 7 × 14) |
| Messages | 10,956 |
| Splits (by conversation) | 411 / 88 / 89 |
| Messages per split | 7,748 / 1,560 / 1,648 |
| Sentiment | 5,670 positive · 2,615 neutral · 2,671 negative |
| Positive rates | sarcasm 2.82% · irony 3.39% · PA 4.06% |
| Tension | mean 25.9 · range 0–100 |

## Unified schema (per message)

```
conversation_id · message_id · speaker_id · timestamp · text · platform
sentiment · emotion · tone · sarcasm · irony · passive_aggression · tension
```

## Known limitations

- Template composition → classification heads saturate (see README honesty note);
  the discriminative benchmarks are the hidden-signal heads, tension regression
  and calibration metrics.
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
