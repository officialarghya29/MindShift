"""Tests for the context-dependence experiment, calibration metrics and the
pretrained-transformer baseline's tokenizer (blueprint §3, §7, §9, §28)."""
import numpy as np
import pytest

from cerebro.common.metrics import (calibration_metrics, cohen_kappa,
                                    krippendorff_alpha_nominal)
from cerebro.data.context_probes import (ALL_PROBES, BENIGN_HISTORIES,
                                         CONTEXT_DEPENDENT_PROBES, TENSE_HISTORIES,
                                         build_probe_corpus, probe_stats,
                                         split_probes)


# ---------------------------------------------------------------- probe corpus
def test_probe_corpus_shape_and_flags():
    corpus = build_probe_corpus()
    expected = len(ALL_PROBES) * 2 * len(BENIGN_HISTORIES) * 4
    assert len(corpus) == expected
    for conv in corpus[:40]:
        assert len(conv) == len(BENIGN_HISTORIES[0]) + 1
        assert sum(1 for m in conv if m.get("is_probe")) == 1
        assert conv[-1]["is_probe"] is True
        # timestamps strictly increase — behavioural features depend on it
        stamps = [m["timestamp"] for m in conv]
        assert stamps == sorted(stamps)
        for m in conv:
            assert 0.0 <= m["tension"] <= 100.0
            assert m["escalation"] == int(m["tension"] >= 60.0)


def test_every_probe_is_label_balanced_across_conditions():
    """I(text; label) must be 0 — the property the whole proof rests on."""
    corpus = build_probe_corpus()
    from collections import defaultdict
    seen = defaultdict(list)
    for conv in corpus:
        seen[conv[-1]["probe_id"]].append(conv[-1]["probe_condition"])
    assert seen, "probe corpus is empty"
    for pid, conds in seen.items():
        assert sorted(set(conds)) == ["benign", "tense"], pid
        assert conds.count("benign") == conds.count("tense"), pid


def test_split_has_no_conversation_overlap_and_holds_out_history_wording():
    corpus = build_probe_corpus()
    splits = split_probes(corpus)
    ids = {k: {c[0]["conversation_id"] for c in v} for k, v in splits.items()}
    assert not (ids["train"] & ids["test"])
    assert not (ids["train"] & ids["val"])
    assert not (ids["test"] & ids["test_seen"])

    train_scripts = {c[-1]["history_index"] for c in splits["train"]}
    test_scripts = {c[-1]["history_index"] for c in splits["test"]}
    assert test_scripts and not (test_scripts & train_scripts), \
        "held-out test must use history wording unseen in training"
    # both conditions land in every split, else the comparison is unbalanced
    for name in ("train", "val", "test"):
        conds = {c[-1]["probe_condition"] for c in splits[name]}
        assert conds == {"benign", "tense"}, name


def test_probe_stats_reports_balance():
    stats = probe_stats(build_probe_corpus())
    assert stats["context_dependent_probes"] == len(CONTEXT_DEPENDENT_PROBES)
    assert stats["history_scripts_per_condition"] == len(TENSE_HISTORIES)
    assert stats["utterance_labels_balanced"] is True


# ------------------------------------------------------------ calibration (§28)
def test_calibration_perfect_predictions_have_zero_ece():
    # 50% confident and right half the time; 100% confident and always right
    conf = np.array([0.5] * 100 + [1.0] * 100)
    correct = np.array([1, 0] * 50 + [1] * 100)
    m = calibration_metrics(correct, conf)
    assert m["ece"] == 0.0
    assert m["mce"] == 0.0
    assert m["n"] == 200


def test_calibration_keeps_points_on_the_lowest_bin_edge():
    """Regression: the edge point must be scored, not silently dropped."""
    conf = np.array([0.5] * 10)
    correct = np.zeros(10, int)              # 50% confident, never right
    m = calibration_metrics(correct, conf)
    assert m["n"] == 10
    assert m["ece"] == pytest.approx(0.5, abs=1e-9)


def test_calibration_is_small_for_a_calibrated_predictor():
    rng = np.random.default_rng(0)
    conf = rng.uniform(0.5, 1.0, 4000)
    correct = (rng.uniform(size=4000) < conf).astype(int)
    m = calibration_metrics(correct, conf)
    assert m["ece"] < 0.05, m
    assert abs(m["mean_confidence"] - m["accuracy"]) < 0.05


def test_calibration_overconfident_is_penalised():
    n = 400
    conf = np.full(n, 0.99)
    correct = np.zeros(n, int)               # 99% confident, never right
    m = calibration_metrics(correct, conf)
    assert m["ece"] > 0.9
    assert m["mce"] > 0.9


def test_calibration_empty_is_safe():
    m = calibration_metrics([], [])
    assert m["ece"] is None and m["n"] == 0


# ------------------------------------------------------------- agreement (§7)
def test_cohen_kappa_bounds():
    assert cohen_kappa(["a", "b", "a"], ["a", "b", "a"])["kappa"] == 1.0
    assert cohen_kappa(["a", "b"], ["b", "a"])["kappa"] == -1.0
    # chance-level agreement with balanced marginals sits near 0
    k = cohen_kappa(["a", "b"] * 50, ["a"] * 50 + ["b"] * 50)["kappa"]
    assert abs(k) < 1e-9


def test_krippendorff_alpha_matches_kappa_when_coders_symmetric():
    a = ["x", "y", "x", "y", "x"]
    b = ["x", "x", "x", "y", "y"]
    ka = cohen_kappa(a, b)["kappa"]
    al = krippendorff_alpha_nominal(a, b)["alpha"]
    # both use the pooled/own marginals here; they must agree in direction
    assert al == pytest.approx(ka, abs=0.25)
    assert -1.0 <= al <= 1.0


# ------------------------------------------------- transformer tokenizer (§9)
def _tiny_vocab(tmp_path):
    tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "fine", ".", "don", "##t",
              "i", "'", "ll", "do", "it", "tonight", "okay"]
    path = tmp_path / "vocab.txt"
    path.write_text("\n".join(tokens) + "\n", encoding="utf-8")
    return str(path)


def test_wordpiece_tokenizer_handles_continuations_and_unknowns(tmp_path):
    from cerebro.models.transformer_baseline import WordPieceTokenizer
    tok = WordPieceTokenizer(_tiny_vocab(tmp_path))
    ids, mask = tok.encode("Fine.")
    inv = {v: k for k, v in tok.vocab.items()}
    assert [inv[i] for i in ids] == ["[CLS]", "fine", ".", "[SEP]"]
    assert mask == [1] * len(ids)
    # greedy longest-match: "dont" -> "don" + "##t"
    ids, _ = tok.encode("dont")
    assert [inv[i] for i in ids][1:3] == ["don", "##t"]
    # out-of-vocabulary collapses to [UNK]
    ids, _ = tok.encode("zzzz")
    assert tok.unk_id in ids
    # empty input still yields a well-formed pair
    ids, mask = tok.encode("")
    assert ids[0] == tok.cls_id and ids[-1] == tok.sep_id and len(ids) == len(mask)


def test_wordpiece_respects_max_length(tmp_path):
    from cerebro.models.transformer_baseline import MAX_LEN, WordPieceTokenizer
    tok = WordPieceTokenizer(_tiny_vocab(tmp_path))
    ids, mask = tok.encode(" ".join(["fine"] * 500))
    assert len(ids) <= MAX_LEN
    assert len(ids) == len(mask)
    assert ids[-1] == tok.sep_id
