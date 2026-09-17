# Methodology: Contextual & Temporal Theory in CEREBRO

This document formalizes the five commitments described in the README and maps
each to its implementation.

## 1. Orthogonal projection heads

For message $m_i$ we predict three distinct quantities:

- **Sentiment** $s_i \in \{\text{pos}, \text{neu}, \text{neg}\}$ — surface valence.
- **Emotion** $e_i \in \mathcal{E}$, $|\mathcal{E}| = 13$ — appraisal state.
- **Tone** $t_i \in \mathcal{T}$, $|\mathcal{T}| = 14$ — social presentation.

These are *not* redundant: the sarcastic message is literally positive,
appraised as angry, and presented as mocking. Collapsing them into one label
destroys exactly the signal the hidden-tone engines need. Implementation:
independent heads over one shared contextual representation (PS-01 §10).

## 2. Sarcasm as contradiction (Contrast + Echoic theory)

Per Raskin (1985), irony requires a **contrast** between a positive evaluation
and a failed expectation; per Sperber & Wilson (1981), the utterance *echoes*
an expectation while dissociating from it. CEREBRO computes:

```
contradiction(m_i) = 0.30 · 1[lit_pos(m_i)] · ctx_neg(C_i)
                   + 0.20 · 1[lit_pos(m_i)] · 1[excl(m_i) ≥ 1]
                   + 0.15 · 1[lit_pos(m_i)] · min(|markers(m_i)|/2, 1)
```

where `lit_pos` is P(positive) − P(negative) > 0.15 from the sentiment head and
`ctx_neg` counts negative-context words in the preceding window. A tension
spike **after** the message adds trajectory corroboration. Evidence streams are
combined by **noisy-OR** (independence assumption): the probability that no
source signals sarcasm is the product of the per-source null probabilities.
Passive aggression additionally requires corroboration (coldness markers or
rising trajectory) — a phrase alone is never a verdict (PS-01 §16).

## 3. Conversation as a non-stationary time series

- **Emotional arc**: intensity $y_i = I(e_i) \in [0,1]$ with a fixed
  emotion→intensity map; stability $= 1 - \min(\sigma_y / 0.4,\ 1)$.
- **Transitions**: a first-order Markov chain over $\mathcal{E}$; a transition
  is *significant* iff $|I(e_i) - I(e_{i-1})| > 0.22$ or $|\Delta \text{tension}| > 12$.
- **Turning points**: robust z-scores on first differences
  $d_i = \tau_i - \tau_{i-1}$: $z_i = (d_i - \mathrm{med}(d)) / (1.4826\,\mathrm{MAD}(d))$,
  flagged at $|z| > 1.6$ or $|d_i| > 15$, with a degenerate-scale fallback
  ($\mathrm{std}(d)$, floored at 1.0) and a no-change guard. Interpretation:
  association, not causation.
- **Escalation**: trajectory classification by phase means (terciles) and
  OLS slope; onset = first 4-run of strictly increasing tension above the
  baseline mean + 8.

## 4. Validation-tuned fusion

Six evidence streams (text · context · memory · behavior · temporal · hidden)
are combined with weights $w_s \propto 1/\hat{\epsilon}_s$ where
$\hat{\epsilon}_s$ is a leave-one-stream-out error estimate on validation
behavior — never hand-picked constants (PS-01 §24). Binary heads are
**Platt-calibrated** (sigmoid over held-out folds) and evaluated with Brier
scores; calibration plots ship in the README.

## 5. Behavioral features as thermometers

The 16-dim behavioral vector (word count, exclamations, CAPS ratio, emoji
polarity, response gap, short-response flag, ...) enters the model as *evidence
channels* and exits the explanation engine as *supporting signals* — never as
proof of an emotional state. The message-length collapse pattern around
escalations (24 → 7 words in PS-01 §17's example) is captured by the
short-response flag and the length dim jointly.

## Teacher forcing vs. predicted history

Training matrices are built with **gold history** (teacher forcing). Test and
production inference re-feed the model's **own predictions** into speaker
memory (predicted history). This gap is deliberate: it measures deployment
reality rather than an oracle setting, and it is why memory-enabled variants
show their value in trajectory quality rather than raw per-message accuracy.

## Reproducibility

Seed 42 everywhere; corpus generation, featurization, training, evaluation and
plotting are all deterministic. Run `python -m evaluation.run_full fit` then
`eval`, and `python -m evaluation.make_graphs` to regenerate every artifact.
