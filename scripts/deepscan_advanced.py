"""Advanced deepscan: adversarial robustness of the full pipeline.

1. Fuzz parsers (random bytes, truncations, unicode attacks, injection shapes)
2. Fuzz the analysis pipeline itself (adversarial messages)
3. State-leak check: repeated analyze() on the same engine must be independent
4. Numeric bounds: every probability in [0,1], tension in [0,100], no NaN/inf
5. Result-schema validation of a full report against the documented shape
6. Adapter schema round-trip with the fuzzed conversations

Exit code 0 = all scans clean.
"""
import random
import string
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "backend")

FAIL = []


def check(name, cond, detail=""):
    status = "✓" if cond else "✗"
    print(f"  {status} {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAIL.append(name)


# ============================================================ 1 · parser fuzz
print("═ 1 · parser fuzz (500 adversarial payloads)")
from cerebro.parsers.platforms import auto_parse

random.seed(42)
alphabet = string.printable + "éüजनह्😄😂🎉\u200b\ufeff\r"
ATTACKS = [
    "", " ", "\n\n\n", "\x00\x00", "\r\n" * 100,
    "1/15/25, 10:02 - " + "A" * 10000 + ": " + "B" * 10000,
    "1/15/25, 10:02 - A: " + "\ud83d" * 5 if False else "1/15/25, 10:02 - A: 🙂" * 500,
    "1/15/25, 10:02 - A: <Media omitted>\n" * 300,
    "13/13/25, 25:61 - ??? :: :: ::",
    "1/15/25, 10:02 - A: 🙃" + "😀" * 200 + "!?" * 100,
    "{\"json\": true}", "[1,2,3]", "<html><body>x</body></html>",
    "2025-01-15T10:02:00Z", "Message", "-", ":", "::",
]
payloads = ATTACKS[:]
for _ in range(480):
    n = random.randint(0, 400)
    s = "".join(random.choice(alphabet) for _ in range(n))
    if random.random() < 0.3 and s:
        cut = random.randint(0, len(s))
        s = s[:cut]                     # random truncation
    payloads.append(s)

crashes = 0
for p in payloads:
    try:
        auto_parse(p)
    except Exception:
        crashes += 1
check("parser fuzz: zero uncaught exceptions", crashes == 0,
      f"{crashes} crashes across {len(payloads)} payloads")

# =========================================================== 2 · pipeline fuzz
print("═ 2 · analysis fuzz (adversarial conversations)")
from cerebro.models.pipeline import CerebroPipeline
from cerebro.models.engines import MultiTaskEngine

eng = MultiTaskEngine().load("models/saved/cerebro_engine")
pipe = CerebroPipeline.from_trained(eng)

ADV = [
    [],                                   # empty conversation
    [{"message_id": 1, "speaker_id": "", "timestamp": None, "text": ""}],
    [{"message_id": 1, "speaker_id": "A" * 5000, "timestamp": None,
      "text": "x" * 50000}],
    [{"message_id": 1, "speaker_id": "A", "timestamp": None, "text": "🙂" * 3000}],
    [{"message_id": i, "speaker_id": "AB"[i % 2], "timestamp": None,
      "text": random.choice(["Fine.", "Whatever.", "Sure.", "Great.", "Why?!",
                             "OK then.", "no problem", "PERFECT!!!", "cool cool"])
      if i else "start"} for i in range(1, 60)],
    [{"message_id": 1, "speaker_id": "A", "timestamp": None,
      "text": "WHY IS THIS BROKEN AGAIN!!! 😡😡"},
     {"message_id": 2, "speaker_id": "B", "timestamp": None,
      "text": "sure. whatever. do what you want."},
     {"message_id": 3, "speaker_id": "A", "timestamp": None,
      "text": "Wow. Great. Just great. 👏"}],
]
crashes = 0
for conv in ADV:
    try:
        rep = pipe.analyze(conv)
        _ = rep["messages"], rep["turning_points"], rep["summary"]
    except Exception as e:
        crashes += 1
        print("      crash:", type(e).__name__, str(e)[:80])
check("pipeline fuzz: zero crashes on adversarial conversations",
      crashes == 0, f"{crashes} crashes")

# ======================================================== 3 · state-leak check
print("═ 3 · state-leak independence")
probe = [{"message_id": 1, "speaker_id": "A", "timestamp": None,
          "text": "Why is this still broken?!"},
         {"message_id": 2, "speaker_id": "B", "timestamp": None,
          "text": "Fine. Do what you want then."}]
base = pipe.analyze(probe)
leaks = 0
for _ in range(4):
    hot = [{"message_id": 1, "speaker_id": "X", "timestamp": None,
            "text": "I HATE THIS SO MUCH!!! 😡"}]
    pipe.analyze(hot)                     # contaminate engine state
    again = pipe.analyze(probe)
    for a, b in zip(base["messages"], again["messages"]):
        if (a["tension"] != b["tension"]
                or a["emotion"]["label"] != b["emotion"]["label"]
                or a["sarcasm"]["probability"] != b["sarcasm"]["probability"]):
            leaks += 1
check("repeated analyze() is state-independent", leaks == 0,
      f"{leaks} divergent fields after contamination runs")

# ======================================================= 4 · numeric bounds
print("═ 4 · numeric bounds over all probe reports")
import math

bounds_ok = True
reports = [base] + [pipe.analyze(c) for c in ADV if c]
for rep in reports:
    for m in rep["messages"]:
        vals = [m["sentiment"]["confidence"],
                m["emotion"]["confidence"], m["tone"]["confidence"],
                m["sarcasm"]["probability"], m["irony"]["probability"],
                m["passive_aggression"]["probability"], m["tension"]]
        for v in vals:
            if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
                bounds_ok = False
        probs = list(m["sentiment"]["probabilities"].values()) + \
            list(m["emotion"]["probabilities"].values()) + \
            list(m["tone"]["probabilities"].values())
        if not all(-1e-9 <= p <= 1 + 1e-9 for p in probs):
            bounds_ok = False
        if not 0.0 <= m["tension"] <= 100.0:
            bounds_ok = False
check("all probabilities ∈ [0,1], tension ∈ [0,100], no NaN/inf", bounds_ok)

sum_ok = True
for rep in reports:
    for m in rep["messages"]:
        s = sum(m["sentiment"]["probabilities"].values())
        if not 0.98 <= s <= 1.02:
            sum_ok = False
check("sentiment probabilities sum to 1", sum_ok)

# ================================================== 5 · report schema validation
print("═ 5 · full report schema")
KEY_MSG = {"message_id", "speaker_id", "text", "sentiment", "emotion", "tone",
           "sarcasm", "irony", "passive_aggression", "tension", "context_text",
           "signals"}
KEY_BIN = {"probability", "confidence", "supporting_signals"}
KEY_SUM = {"n_messages", "trajectory", "mean_tension"}
schema_ok = True
for rep in reports:
    if not KEY_SUM <= set(rep["summary"]):
        schema_ok = False
    if "turning_points" not in rep:
        schema_ok = False
    for m in rep["messages"]:
        if not KEY_MSG <= set(m):
            schema_ok = False
        for h in ("sarcasm", "irony", "passive_aggression"):
            if not KEY_BIN <= set(m[h]) or "learned_p" not in m[h]:
                schema_ok = False
check("report schema fields present on every message/summary", schema_ok)

# ================================================ 6 · adapter round-trip fuzz
print("═ 6 · adapter round-trip on fuzzed conversations")
from cerebro.data.adapters import validate_conversation, _mk

fuzz_convs = []
for c in ADV:
    if not c:
        continue
    fuzz_convs.append([_mk("fz", i + 1, m["speaker_id"][:20] or "A",
                           m["text"][:200] or "ok", "neutral", "casual",
                           0, 0, 0) for i, m in enumerate(c)])
try:
    validate_conversation(fuzz_convs)
    check("fuzzed conversations pass unified schema", True)
except Exception as e:
    check("fuzzed conversations pass unified schema", False, str(e)[:90])

print()
if FAIL:
    print("✗ DEEPSCAN FAILURES:", FAIL)
    sys.exit(1)
print("✓ ADVANCED DEEPSCAN CLEAN — all scans passed")
