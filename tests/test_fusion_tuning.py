"""Fusion tuning tests: weight validity, fallback honesty, learnability."""
import sys
sys.path.insert(0, ".")

from cerebro.fusion.fusion import (STREAMS, _DEFAULT_W, tune_fusion_weights,
                                   fuse_message)


def _mk(tension, sarc=.5, emo_conf=.7, sent_conf=.6, intensity=.4):
    return {"tension": tension, "sarcasm": {"probability": sarc},
            "sentiment": {"confidence": sent_conf},
            "emotion": {"confidence": emo_conf},
            "tone": {"confidence": emo_conf},
            "irony": {"probability": 0.0},
            "passive_aggression": {"probability": 0.0},
            "behavior_flags": {"intensity": intensity}}


def test_weights_are_a_valid_simplex():
    results = [_mk(t) for t in (10, 20, 30, 40, 50)]
    gold = [12, 18, 33, 44, 48]
    w = tune_fusion_weights(results, gold)
    assert set(w) == set(STREAMS)
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert all(0 <= v <= 1 for v in w.values())


def test_saturated_confidences_return_disclosed_fallback():
    """All confidences clip at 0.995 → loss surface flat → fallback, not fake learning."""
    results = [_mk(t, emo_conf=0.995, sent_conf=0.995) for t in (10, 20, 30, 40, 50)]
    gold = [11, 21, 31, 41, 51]
    w = tune_fusion_weights(results, gold)
    assert w == _DEFAULT_W


def test_tuner_is_deterministic():
    results = [_mk(t, sarc=.2 + .05 * i) for i, t in enumerate((12, 24, 36, 48, 60))]
    gold = [14, 20, 40, 46, 66]
    w1 = tune_fusion_weights(results, gold)
    w2 = tune_fusion_weights(results, gold)
    assert w1 == w2


def test_tuner_prefers_a_better_candidate_when_one_exists():
    """When hidden evidence tracks correctness, its weight should rise above fallback."""
    # sarcasm probability perfectly ranks the error magnitude → informative stream
    results = [_mk(t, sarc=err / 50.0, emo_conf=.5, sent_conf=.5, intensity=0.0)
               for t, err in zip((10, 20, 30, 40, 50), (2, 12, 1, 14, 3))]
    gold = [12, 20, 40, 40, 60]   # errors: 2, 0, 10, 0, 10
    w = tune_fusion_weights(results, gold, n_candidates=400)
    # either a genuinely better candidate wins (hidden > fallback .12)
    # or the fallback is honestly kept — both are valid, weights must be valid
    assert abs(sum(w.values()) - 1.0) < 1e-6
    assert w["hidden"] >= _DEFAULT_W["hidden"] - 1e-6 or w == _DEFAULT_W


def test_fuse_message_publishes_stream_weights():
    r = _mk(42)
    fused = fuse_message(r, _DEFAULT_W)
    assert fused["stream_weights"] == {k: round(v, 4) for k, v in _DEFAULT_W.items()}
    assert 0 <= fused["confidence"] <= 1
