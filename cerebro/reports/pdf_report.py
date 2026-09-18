"""PDF conversation report (PS-01 §30, P2 roadmap item).

Renders a stored CEREBRO report dict into a compact, dark-accent PDF:
summary metrics → tension curve (matplotlib, no text leakage) → speaker
profiles → turning points → per-message table → methodology disclaimer.

fpdf2 is an optional dependency; import lazily and fail with a clear
message. Run standalone:  PYTHONPATH=. python -m cerebro.reports.pdf_report \
    evaluation/results/demo_report.json out.pdf
"""
from __future__ import annotations

import io
import json
import sys


def _asc(text) -> str:
    """Core fonts are latin-1; map the few typographic chars we use and
    drop the rest so real chat text never crashes the export."""
    replacements = {"\u2014": "-", "\u2013": "-", "\u2018": "'",
                    "\u2019": "'", "\u201c": '"', "\u201d": '"',
                    "\u2192": "->", "\u00b7": "·"}
    out = []
    for ch in str(text):
        if ch in replacements:
            out.append(replacements[ch])
        elif ord(ch) < 256:
            out.append(ch)
        else:
            out.append("?")
    return "".join(out)


def build_pdf(report: dict, out_path: str) -> str:
    try:
        from fpdf import FPDF
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "fpdf2 is required for PDF export: pip install fpdf2") from e

    _A = _asc
    INK = (15, 23, 42)
    CYAN = (0, 150, 199)
    MUTED = (100, 116, 139)

    s = report["summary"]
    msgs = report["messages"]

    class PDF(FPDF):
        def header(self):
            self.set_font("helvetica", "B", 13)
            self.set_text_color(*INK)
            self.cell(0, 8, _A("CEREBRO — Conversation Intelligence Report"),
                      new_x="LMARGIN", new_y="NEXT")
            self.set_font("helvetica", "", 8.5)
            self.set_text_color(*MUTED)
            self.cell(0, 5,
                      f"conversation {s.get('conversation_id', '?')} · "
                      f"{s['n_messages']} messages · {s.get('n_speakers', '?')} speakers "
                      f"· generated {__import__('datetime').datetime.now():%Y-%m-%d %H:%M}",
                      new_x="LMARGIN", new_y="NEXT")
            self.set_draw_color(*CYAN)
            self.set_line_width(.5)
            self.line(10, 24, 200, 24)
            self.ln(3)

    pdf = PDF(format="A4")
    pdf.set_margins(10, 12, 10)
    pdf.set_auto_page_break(True, margin=16)
    pdf.add_page()

    # ---- summary band -----------------------------------------------------
    pdf.set_font("helvetica", "B", 10.5)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, "1 · Overall dynamics", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 9)
    dom = ", ".join(f"{lbl} {int(100 * share)}%"
                    for lbl, share in (s.get("dominant_emotion") or [])[:3]) or "—"
    dist = s.get("sentiment_distribution") or {}
    sent = ", ".join(f"{k} {int(100 * v / max(sum(dist.values()), 1))}%"
                     for k, v in dist.items()) or "—"
    for line in (
            _A(f"Trajectory: {s.get('trajectory', '?')}   ·   "
               f"Mean tension: {s.get('mean_tension', '?')}   ·   "
               f"Peak tension: {s.get('peak_tension', '?')}"),
            _A(f"Dominant emotions: {dom}"),
            _A(f"Sentiment mix: {sent}"),
            _A(f"Sarcasm {s.get('sarcasm_level', 0):.2f} · "
               f"Irony {s.get('irony_level', 0):.2f} · "
               f"Passive-aggression {s.get('passive_aggression_level', 0):.2f} (mean)"),
    ):
        pdf.cell(0, 5, line, new_x="LMARGIN", new_y="NEXT")
    pdf.ln(2)

    # ---- tension curve ----------------------------------------------------
    pdf.set_font("helvetica", "B", 10.5)
    pdf.cell(0, 6, "2 · Tension curve & turning points", new_x="LMARGIN",
             new_y="NEXT")
    tps = {tp["message_id"]: tp for tp in report.get("turning_points", [])}
    curve = [m["tension"] for m in msgs]
    buf = io.BytesIO()
    _tension_png(curve, tps, buf)
    buf.seek(0)
    pdf.image(buf, w=pdf.epw)
    pdf.ln(1)

    # ---- speakers ----------------------------------------------------------
    pdf.set_font("helvetica", "B", 10.5)
    pdf.cell(0, 6, "3 · Speaker profiles", new_x="LMARGIN", new_y="NEXT")
    for spk, p in (report.get("speaker_profiles") or {}).items():
        pdf.set_font("helvetica", "B", 9)
        pdf.set_text_color(*INK)
        top_emo = ", ".join(f"{k} {int(100 * v)}%"
                            for k, v in list(p.get("dominant_emotions", {})
                                             .items())[:3]) or "—"
        pdf.cell(0, 5, _A(f"{spk} — {p.get('messages', '?')} messages · "
                          f"avg tension {p.get('avg_tension', '?')}"),
                 new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("helvetica", "", 8.5)
        pdf.set_text_color(*MUTED)
        pdf.cell(0, 4.5, _A(f"emotions: {top_emo}"), new_x="LMARGIN", new_y="NEXT")

    pdf.ln(2)

    # ---- turning points ----------------------------------------------------
    pdf.set_font("helvetica", "B", 10.5)
    pdf.set_text_color(*INK)
    pdf.cell(0, 6, "4 · Turning points (model-estimated)", new_x="LMARGIN",
             new_y="NEXT")
    if tps:
        for tp in report.get("turning_points", []):
            b, a = tp.get("before", {}), tp.get("after", {})
            pdf.set_font("helvetica", "", 8.5)
            pdf.cell(0, 4.6,
                     _A(f"#{tp['message_id']}  {b.get('emotion', '?')} -> {a.get('emotion', '?')}"
                        f"   tension {b.get('tension', '?')} -> {a.get('tension', '?')}"
                        f"  (d{tp.get('tension_change', 0):+.1f}, z={tp.get('robust_z', 0):.2f})"),
                     new_x="LMARGIN", new_y="NEXT")
            pdf.set_text_color(*MUTED)
            pdf.multi_cell(0, 4.2, _A(f"trigger: \u201c{tp.get('trigger_text', '')[:110]}\u201d"))
            pdf.set_text_color(*INK)
    else:
        pdf.set_font("helvetica", "", 9)
        pdf.cell(0, 5, "no significant shifts detected", new_x="LMARGIN",
                 new_y="NEXT")
    pdf.ln(2)

    # ---- per-message table --------------------------------------------------
    pdf.set_font("helvetica", "B", 10.5)
    pdf.cell(0, 6, "5 · Message explorer", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("helvetica", "", 7.6)
    headers = ["#", "speaker", "emotion", "tens", "sarc", "PA", "text"]
    widths = [8, 20, 24, 12, 12, 12, 112]
    pdf.set_fill_color(238, 244, 248)
    for h, w in zip(headers, widths):
        pdf.cell(w, 5.4, h, border=.2, fill=True)
    pdf.ln()
    for m in msgs:
        if pdf.get_y() > 262:
            pdf.add_page()
            for h, w in zip(headers, widths):
                pdf.cell(w, 5.4, h, border=.2, fill=True)
            pdf.ln()
        esc = _asc(str(m.get("text", "")).replace("\n", " "))
        pdf.cell(widths[0], 5, str(m["message_id"]), border=.2)
        pdf.cell(widths[1], 5, str(m["speaker_id"])[:12], border=.2)
        pdf.cell(widths[2], 5, m["emotion"]["label"][:14], border=.2)
        pdf.cell(widths[3], 5, f"{m['tension']:.0f}", border=.2)
        pdf.cell(widths[4], 5, f"{m['sarcasm']['probability']:.2f}", border=.2)
        pdf.cell(widths[5], 5, f"{m['passive_aggression']['probability']:.2f}",
                 border=.2)
        pdf.cell(widths[6], 5, esc[:70], border=.2)
        pdf.ln()

    # ---- disclaimer ----------------------------------------------------------
    pdf.set_y(-24)
    pdf.set_font("helvetica", "I", 7.5)
    pdf.set_text_color(*MUTED)
    pdf.multi_cell(0, 4,
                   "All outputs are model-estimated with calibrated confidence. "
                   "Turning points are statistical associations, not causal claims. "
                   "Speaker profiles describe language patterns, not psychological "
                   "diagnoses (PS-01 §29, §40).")

    pdf.output(out_path)
    return out_path


def _tension_png(curve: list[float], tps: dict, buf: io.BytesIO) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7.4, 2.1))
    x = list(range(1, len(curve) + 1))
    ax.plot(x, curve, color="#0097C7", lw=1.8, zorder=3)
    ax.fill_between(x, curve, color="#0097C7", alpha=.10, zorder=2)
    ax.axhline(60, color="#F59E0B", lw=.9, ls="--")
    ax.text(len(curve) + .3, 60, "escalation", fontsize=7, color="#B45309",
            va="center")
    for mid, tp in tps.items():
        if 1 <= mid <= len(curve):
            ax.scatter([mid], [curve[mid - 1]], color="#E11D48", zorder=4, s=22)
            ax.annotate(f"#{mid}", (mid, curve[mid - 1]), textcoords="offset points",
                        xytext=(0, 7), fontsize=7.5, color="#9F1239",
                        ha="center")
    ax.set_xlim(.6, len(curve) + 2.6)
    ax.set_ylim(0, 100)
    ax.set_xlabel("message", fontsize=8)
    ax.set_ylabel("tension", fontsize=8)
    ax.tick_params(labelsize=7.5)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    src, dst = sys.argv[1], sys.argv[2]
    with open(src) as f:
        rep = json.load(f)
    build_pdf(rep, dst)
    print(f"PDF written → {dst}")
