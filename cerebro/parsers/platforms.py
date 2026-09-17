"""Platform parsers → one common message format (PS-01 §2).

Common record:
  conversation_id, message_id, speaker_id, timestamp(iso|None), text, platform
"""
from __future__ import annotations

import json
import re
from datetime import datetime

WA_RE = re.compile(
    r"^\s*([\d]{1,2})/([\d]{1,2})/([\d]{2,4}),?\s+([\d]{1,2}):([\d]{2})(?::([\d]{2}))?\s*"
    r"(?:([APap])\.?[Mm]\.?)?\s*[-–]\s*([^:]+?):\s(.*)$")
SYS_MARKERS = ("<Media omitted>", "<This message was edited>",
               "Messages and calls are end-to-end encrypted",
               "created group", "added you", "changed the subject",
               "joined using this group")


def _norm_speaker(name: str) -> str:
    return name.strip().strip("\u200f\u200e")


def _detect_dayfirst(lines: list[str]) -> bool:
    """Decide DD/MM vs MM/DD from unambiguous timestamps in the file.
    Falls back to day-first (the WhatsApp default in most locales)."""
    for line in lines:
        m = WA_RE.match(line)
        if m:
            a, b = int(m.group(1)), int(m.group(2))
            if a > 12:
                return True       # first component must be the day
            if b > 12:
                return False      # second component must be the day
    return True


def parse_whatsapp(raw_text: str, conversation_id: str = "conv_001") -> list[dict]:
    """WhatsApp .txt export → records. Handles 12/24h clocks, DD/MM vs MM/DD
    (auto-detected from unambiguous timestamps) and continuation lines."""
    lines = raw_text.splitlines()
    day_first = _detect_dayfirst(lines)
    msgs = []
    pending = None
    for line in lines:
        m = WA_RE.match(line)
        if m:
            if pending:
                msgs.append(pending)
            a, b, y, hh, mm, ss, ap, spk, txt = m.groups()
            d, mo = (int(a), int(b)) if day_first else (int(b), int(a))
            y = int(y)
            if y < 100:
                y += 2000
            hh = int(hh)
            if ap:
                ap = ap.upper()          # regex captures 'P'/'A' ('M' consumed separately)
                hh = hh % 12 + (12 if ap == "P" else 0)
            ts = None
            try:
                ts = datetime(y, mo, d, hh, int(mm), int(ss or 0)).isoformat()
            except ValueError:
                pass
            pending = {"conversation_id": conversation_id, "message_id": len(msgs) + 1,
                       "speaker_id": _norm_speaker(spk), "timestamp": ts,
                       "text": txt.strip(), "platform": "whatsapp"}
        elif pending and line.strip() and not any(s in line for s in SYS_MARKERS):
            pending["text"] += "\n" + line.strip()   # continuation of previous message
    if pending:
        msgs.append(pending)
    return msgs


def parse_discord(raw_text: str, conversation_id: str = "conv_001") -> list[dict]:
    """Discord JSON export or '[hh:mm] Author' plain text."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, list) and data and "author" in data[0]:
            return [{"conversation_id": conversation_id, "message_id": i + 1,
                     "speaker_id": _norm_speaker(str(m.get("author", {}).get("username",
                                                                    m.get("author")))),
                     "timestamp": m.get("timestamp"),
                     "text": (m.get("content") or "").strip(),
                     "platform": "discord"} for i, m in enumerate(data)]
    except (json.JSONDecodeError, TypeError, KeyError, AttributeError):
        pass
    pat = re.compile(r"^\[?([\d]{1,2}:[\d]{2}(?::[\d]{2})?)\]?\s+([^:]+):\s(.*)$")
    msgs, pending = [], None
    for line in raw_text.splitlines():
        m = pat.match(line)
        if m:
            if pending:
                msgs.append(pending)
            ts, spk, txt = m.groups()
            pending = {"conversation_id": conversation_id, "message_id": len(msgs) + 1,
                       "speaker_id": _norm_speaker(spk), "timestamp": ts, "text": txt.strip(),
                       "platform": "discord"}
        elif pending and line.strip():
            pending["text"] += "\n" + line.strip()
    if pending:
        msgs.append(pending)
    return msgs


def parse_slack(raw_text: str, conversation_id: str = "conv_001") -> list[dict]:
    """Slack JSON export (list of {user,text,ts} objects)."""
    try:
        data = json.loads(raw_text)
        if isinstance(data, list) and data and ("user" in data[0] or "text" in data[0]):
            out = []
            for i, m in enumerate(data):
                if m.get("subtype") or not (m.get("text") or "").strip():
                    continue
                ts = None
                if m.get("ts"):
                    try:
                        ts = datetime.fromtimestamp(float(m["ts"])).isoformat()
                    except (ValueError, OSError, OverflowError):
                        ts = None
                out.append({"conversation_id": conversation_id, "message_id": len(out) + 1,
                            "speaker_id": _norm_speaker(str(m.get("user", m.get("username",
                                                                                 "unknown")))),
                            "timestamp": ts, "text": m["text"].strip(), "platform": "slack"})
            return out
    except (json.JSONDecodeError, TypeError, KeyError):
        pass
    return parse_generic(raw_text, conversation_id, platform="slack")


def parse_generic(raw_text: str, conversation_id: str = "conv_001",
                  platform: str = "generic") -> list[dict]:
    """'Speaker: text' per line (one message per line)."""
    msgs = []
    for line in raw_text.splitlines():
        line = line.strip()
        if not line:
            continue
        m = re.match(r"^([^:]{1,32}):\s(.*)$", line)
        if m and not re.match(r"^(https?|www)", m.group(1)):
            spk, txt = m.groups()
        else:
            spk, txt = "user", line
        msgs.append({"conversation_id": conversation_id, "message_id": len(msgs) + 1,
                     "speaker_id": _norm_speaker(spk), "timestamp": None,
                     "text": txt.strip(), "platform": platform})
    return msgs


def auto_parse(raw_text: str, conversation_id: str = "conv_001",
               platform: str | None = None) -> list[dict]:
    """Detect platform automatically, then dispatch."""
    if platform == "whatsapp" or (platform is None and
                                  any(WA_RE.match(l)
                                      for l in raw_text.splitlines()[:5])):
        return parse_whatsapp(raw_text, conversation_id)
    stripped = raw_text.lstrip()
    if platform == "slack" or (platform is None and stripped.startswith("[")
                               and ("\"user\"" in raw_text or "\"text\"" in raw_text)):
        slack = parse_slack(raw_text, conversation_id)
        if slack:
            return slack
    if platform == "discord" or (platform is None and
                                 re.search(r"^\[?[\d]{1,2}:[\d]{2}", raw_text, re.M)):
        disc = parse_discord(raw_text, conversation_id)
        if disc:
            return disc
    return parse_generic(raw_text, conversation_id)
