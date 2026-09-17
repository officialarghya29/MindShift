"""Deepscan: exercise every API endpoint end-to-end with the real trained engine."""
import io
import sys
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from fastapi.testclient import TestClient
from backend.app import main

CHAT = (
    b"Aarav: Hey! Did you finish the project?\n"
    b"Meera: Yeah I'll do it tonight.\n"
    b"Aarav: Perfect, thanks!\n"
    b"Meera: You said that yesterday too.\n"
    b"Aarav: Fine. Do what you want then.\n"
    b"Meera: Wow. Great. Just great.\n"
    b"Aarav: I'm sorry, I really mean it this time.\n"
    b"Meera: ...okay. Let's just fix it tomorrow."
)

client = TestClient(main.app)
passed, failed = 0, []


def check(name, cond, extra=""):
    global passed
    if cond:
        passed += 1
        print(f"  OK   {name} {extra}")
    else:
        failed.append(name)
        print(f"  FAIL {name} {extra}")


# 1 health (engine lazy-loads on first /analyze, so model_loaded may be False here)
r = client.get("/healthz")
check("GET /healthz (pre-load)", r.status_code == 200 and r.json()["status"] == "ok")

# 2 upload
r = client.post("/upload", files={"file": ("chat.txt", io.BytesIO(CHAT), "text/plain")})
check("POST /upload", r.status_code == 200 and r.json()["n_messages"] == 8)
up = r.json()

# 3 parse
r = client.post("/parse", files={"file": ("chat.txt", io.BytesIO(CHAT), "text/plain")})
check("POST /parse", r.status_code == 200 and r.json()["n_messages"] == 8)

# 4 analyze
r = client.post("/analyze", files={"file": ("chat.txt", io.BytesIO(CHAT), "text/plain")})
check("POST /analyze", r.status_code == 200 and r.json()["summary"]["n_messages"] == 8)
rep = r.json()
conv = rep["summary"]["conversation_id"]

# 4b health after lazy load
r = client.get("/healthz")
check("GET /healthz (post-load)", r.status_code == 200
      and r.json()["model_loaded"] is True and r.json()["stored_conversations"] >= 1)

# 5 conversation store
r = client.get(f"/conversation/{conv}")
check("GET /conversation/{id}", r.status_code == 200 and len(r.json()["messages"]) == 8)

# 6 timeline
r = client.get(f"/conversation/{conv}/timeline")
tl = r.json()["timeline"]
check("GET /timeline", r.status_code == 200 and len(tl) == 8
      and {"message_id", "emotion", "tension", "sarcasm"} <= set(tl[0]))

# 7 turning points
r = client.get(f"/conversation/{conv}/turning-points")
check("GET /turning-points", r.status_code == 200 and len(r.json()["turning_points"]) >= 1)

# 8 speakers
r = client.get(f"/conversation/{conv}/speakers")
spk = r.json()["speaker_profiles"]
check("GET /speakers", r.status_code == 200 and set(spk) == {"Aarav", "Meera"})

# 9 report
r = client.get(f"/conversation/{conv}/report")
check("GET /report", r.status_code == 200 and "emotional_arc" in r.json())

# 10 topics
r = client.get(f"/conversation/{conv}/topics")
check("GET /topics", r.status_code == 200 and len(r.json()["topics"]) >= 1)

# 11 why
r = client.get(f"/why/{conv}/5")
why = r.json()
check("GET /why/5", r.status_code == 200
      and "detected_evidence" in why and "inferred_interpretation" in why)

# 12 what-changed
r = client.get(f"/what-changed/{conv}/5")
wc = r.json()
check("GET /what-changed/5", r.status_code == 200 and "change" in wc)

# negative paths
r = client.get("/conversation/nope")
check("404 unknown conversation", r.status_code == 404)
r = client.post("/upload", files={"file": ("x.txt", io.BytesIO(b""), "text/plain")})
check("422 empty upload", r.status_code == 422)
r = client.get("/why/nope/1")
check("404 why on unknown conv", r.status_code == 404)

# schema contract: required message fields present
m = rep["messages"][0]
required = {"message_id", "speaker_id", "text", "sentiment", "emotion", "tone",
            "tension", "sarcasm", "irony", "passive_aggression", "confidence"}
check("message schema contract", required <= set(m))

print(f"\n{passed} checks passed, {len(failed)} failed"
      + (f" -> {failed}" if failed else ""))
sys.exit(1 if failed else 0)
