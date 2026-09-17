"""Parser tests: all four platforms + edge cases (PS-01 §41 case 20)."""
from cerebro.parsers.platforms import (parse_whatsapp, parse_discord, parse_slack,
                                       parse_generic, auto_parse)

WA_SAMPLE = """1/15/25, 10:02 - Aarav: Hey, did you finish the project?
1/15/25, 10:04 - Meera: Yeah I'll do it tonight.
1/15/25, 10:05 - Aarav: Perfect
thanks!
1/16/25, 9:10 PM - Meera: Morning!
<Media omitted>
"""


def test_whatsapp_basic_and_continuation():
    msgs = parse_whatsapp(WA_SAMPLE, "c1")
    assert len(msgs) == 4                      # media line dropped, continuation merged
    assert msgs[0]["speaker_id"] == "Aarav"
    assert msgs[1]["text"].startswith("Yeah")
    assert msgs[2]["text"].endswith("thanks!")  # multi-line message preserved
    assert msgs[3]["platform"] == "whatsapp"
    assert msgs[0]["timestamp"].startswith("2025-01-15T10:02")


def test_whatsapp_pm_clock():
    # 3/25 forces day-first interpretation (25 can only be a day)
    msgs = parse_whatsapp("3/25/24, 9:10 PM - Riya: Evening!", "c2")
    assert msgs[0]["timestamp"].startswith("2024-03-25T21:10")


def test_discord_json_and_text():
    js = '[{"timestamp": "2025-01-15T10:00:00Z", "author": {"username": "Dev"}, "content": "hello"}]'
    out = parse_discord(js, "c3")
    assert out[0]["speaker_id"] == "Dev" and out[0]["text"] == "hello"
    txt = "[10:02] Kabir: starting now\n[10:04] Ananya: on my way"
    out2 = parse_discord(txt, "c4")
    assert len(out2) == 2 and out2[1]["speaker_id"] == "Ananya"


def test_slack_json():
    js = '[{"user": "U1", "text": "deploy done", "ts": "1736942400.000000"}, {"subtype": "bot_help", "text": "skip"}]'
    out = parse_slack(js, "c5")
    assert len(out) == 1 and out[0]["speaker_id"] == "U1"


def test_generic_and_auto():
    g = "Aarav: hi\nMeera: hello there"
    out = parse_generic(g, "c6")
    assert [m["speaker_id"] for m in out] == ["Aarav", "Meera"]
    assert auto_parse(WA_SAMPLE, "c7")[0]["platform"] == "whatsapp"


def test_malformed_input_does_not_crash():
    for junk in ("", "\n\n\n", "::::::", "[]", "{}", "no speaker here"):
        out = auto_parse(junk, "cx")
        assert isinstance(out, list)
