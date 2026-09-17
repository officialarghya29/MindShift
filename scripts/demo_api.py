"""Live demo: analyze a sample chat through the real FastAPI app + trained model."""
import io
import json
import sys
import time
import warnings

warnings.filterwarnings("ignore")
sys.path.insert(0, ".")
sys.path.insert(0, "backend")

from fastapi.testclient import TestClient
from backend.app import main

SAMPLE = (
    b"Aarav: Hey! Did you finish the project?\n"
    b"Meera: Yeah I'll do it tonight.\n"
    b"Aarav: Perfect, thanks!\n"
    b"Meera: You said that yesterday too.\n"
    b"Aarav: Fine. Do what you want then.\n"
    b"Meera: Wow. Great. Just great.\n"
    b"Aarav: I'm sorry, I really mean it this time.\n"
    b"Meera: ...okay. Let's just fix it tomorrow."
)

t0 = time.time()
client = TestClient(main.app)   # real engine lazy-loads on first /analyze
r = client.post("/analyze", files={
    "file": ("chat.txt", io.BytesIO(SAMPLE), "text/plain")})
report = r.json()
print("status:", r.status_code, "| took %.1fs" % (time.time() - t0))
print("conversation:", report["summary"]["conversation_id"],
      "| msgs:", report["summary"]["n_messages"])
print("trajectory:", report["summary"]["trajectory"],
      "| mean tension:", report["summary"]["mean_tension"])
for m in report["messages"]:
    print("  #%d %-6s | %-11s | tension %5.1f | sarc %.2f | PA %.2f | %r" % (
        m["message_id"], m["speaker_id"], m["emotion"]["label"], m["tension"],
        m["sarcasm"]["probability"], m["passive_aggression"]["probability"],
        m["text"][:38]))
print("turning points:", [
    (t["message_id"], t["before"]["emotion"], "->", t["after"]["emotion"],
     t["tension_change"]) for t in report["turning_points"]])
conv = report["summary"]["conversation_id"]
wc = client.get("/what-changed/%s/5" % conv).json()
print("WHAT CHANGED @5:", json.dumps(wc.get("change", wc), indent=1))
