"""Graph generation for the README — every plot is built from *executed* results
(evaluation/results/*.json + the persisted engine re-run on the test split).

Style: dark futuristic, logo watermarked.
Run:  python -m evaluation.make_graphs
"""
from __future__ import annotations

import json
import warnings
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from PIL import Image

warnings.filterwarnings("ignore")

RESULTS = "evaluation/results"
GRAPHS = "assets/graphs"
LOGO = "assets/logo/cerebro_logo.png"
LOGO_SMALL = "assets/logo/cerebro_logo_128.png"

# futuristic palette
CYAN, PURPLE, PINK, GREEN, YELLOW, ORANGE = ("#00E5FF", "#B388FF", "#FF5C8A",
                                             "#7CFFB2", "#FFD166", "#FF9E64")
BG = "#0B0F1A"
PANEL = "#111827"

plt.rcParams.update({
    "figure.facecolor": BG, "axes.facecolor": PANEL, "savefig.facecolor": BG,
    "axes.edgecolor": "#374151", "axes.labelcolor": "#E5E7EB",
    "xtick.color": "#9CA3AF", "ytick.color": "#9CA3AF",
    "text.color": "#F3F4F6", "grid.color": "#1F2937",
    "font.family": "DejaVu Sans", "axes.grid": True, "grid.alpha": .5,
})


def load(name):
    with open(f"{RESULTS}/{name}", encoding="utf-8") as f:
        return json.load(f)


def watermark(fig, alpha=0.10, zoom=0.28, pos=(0.78, 0.04)):
    """Place the CEREBRO logo as a subtle watermark on a figure."""
    logo = Image.open(LOGO).convert("RGBA")
    logo.thumbnail((360, 360))
    ax_img = fig.add_axes([pos[0], pos[1], zoom, zoom * logo.height / logo.width],
                          zorder=-1)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo), alpha=alpha)


def hero_header(fig, title, subtitle):
    """Small logo + title header on a figure."""
    logo = Image.open(LOGO_SMALL)
    ax_img = fig.add_axes([0.012, 0.90, 0.075, 0.085 * logo.height / logo.width], zorder=5)
    ax_img.axis("off")
    ax_img.imshow(np.asarray(logo))
    fig.text(0.095, 0.955, title, fontsize=17, fontweight="bold", color=CYAN,
             ha="left", va="center")
    fig.text(0.095, 0.915, subtitle, fontsize=9, color="#9CA3AF", ha="left", va="center")


def _cv(d, k):
    return d[k] if d and k in d else None


# ---------------------------------------------------------------- graph 1
def graph_hero():
    base = load("baselines.json")
    abl = load("ablations.json")
    summ = load("summary.json")
    demo = load("demo_report.json")
    full = summ["full_metrics"]

    fig = plt.figure(figsize=(14.5, 9.5))
    hero_header(fig, "CEREBRO — Conversation Intelligence Dashboard",
                "588 synthetic-annotated conversations · 10,956 messages · seed 42 · evaluated on held-out test conversations")

    # -- panel 1: tension curve of the demo conversation (real model output)
    ax1 = fig.add_axes([0.06, 0.58, 0.40, 0.26])
    tension = [m["tension"] for m in demo["messages"]]
    xs = list(range(1, len(tension) + 1))
    ax1.plot(xs, tension, color=CYAN, lw=2.2, marker="o", ms=3.5,
             mfc=CYAN, mec=BG, label="tension (predicted)")
    if demo["turning_points"]:
        tp = demo["turning_points"][0]
        ax1.axvline(tp["message_id"], color=PINK, ls="--", lw=1.4, alpha=.9)
        ax1.annotate(f'turning point\n{tp["before"]["emotion"]} → {tp["after"]["emotion"]}',
                     xy=(tp["message_id"], tp["after"]["tension"]),
                     xytext=(tp["message_id"] + 0.5, tp["after"]["tension"] - 22),
                     fontsize=7.5, color=PINK,
                     arrowprops=dict(arrowstyle="->", color=PINK, lw=1))
    ax1.axhspan(60, 100, color=PINK, alpha=.06)
    ax1.set_title("Emotional arc — sample test conversation (Model E output)",
                  fontsize=10, loc="left", color="#E5E7EB")
    ax1.set_xlabel("message #"); ax1.set_ylabel("tension (0–100)")
    ax1.set_ylim(0, 100)

    # -- panel 2: baselines vs CEREBRO (sentiment / emotion / tone macro-F1)
    ax2 = fig.add_axes([0.55, 0.58, 0.40, 0.26])
    names = ["B1 TF-IDF+LR", "B2 TF-IDF+SVC", "B3 +context", "CEREBRO (E)"]
    keys = ["B1_tfidf_logreg", "B2_tfidf_svc", "B3_tfidf_context"]
    sent = [base[k]["sentiment"]["f1_macro"] for k in keys] + [full["sentiment"]["f1_macro"]]
    emo = [base[k]["emotion"]["f1_macro"] for k in keys] + [full["emotion"]["f1_macro"]]
    tone = [base[k]["tone"]["f1_macro"] for k in keys] + [full["tone"]["f1_macro"]]
    x = np.arange(4); w = 0.26
    ax2.bar(x - w, sent, w, color=CYAN, label="sentiment")
    ax2.bar(x, emo, w, color=PURPLE, label="emotion")
    ax2.bar(x + w, tone, w, color=PINK, label="tone")
    for xi, vals in zip(x, zip(sent, emo, tone)):
        for dx, v in zip((-w, 0, w), vals):
            ax2.text(xi + dx, v + .01, f"{v:.2f}", ha="center", fontsize=6.3, color="#D1D5DB")
    ax2.set_xticks(x); ax2.set_xticklabels(names, fontsize=7.5)
    ax2.set_ylim(0, 1.1)
    ax2.set_title("Macro-F1: baselines vs full CEREBRO (test split)",
                  fontsize=10, loc="left", color="#E5E7EB")
    ax2.legend(fontsize=7, framealpha=0, loc="lower right")

    # -- panel 3: ablation progression
    ax3 = fig.add_axes([0.06, 0.09, 0.40, 0.36])
    variants = ["A", "B", "C", "D"]
    labels = ["A\ntext", "B\n+context", "C\n+memory", "D\n+behavior"]
    s_f1 = [abl[v]["sentiment"]["f1_macro"] for v in variants]
    e_f1 = [abl[v]["emotion"]["f1_macro"] for v in variants]
    t_mae = [abl[v]["tension"]["mae"] for v in variants]
    x = np.arange(4)
    ax3.plot(x, s_f1, "-o", color=CYAN, lw=2, label="sentiment F1 (macro)")
    ax3.plot(x, e_f1, "-s", color=PURPLE, lw=2, label="emotion F1 (macro)")
    for xi, v in zip(x, s_f1):
        ax3.text(xi, v + .015, f"{v:.3f}", ha="center", fontsize=7, color=CYAN)
    for xi, v in zip(x, e_f1):
        ax3.text(xi, v - .045, f"{v:.3f}", ha="center", fontsize=7, color=PURPLE)
    ax3.set_xticks(x); ax3.set_xticklabels(labels, fontsize=8)
    ax3.set_ylim(0.5, 1.05)
    ax3.set_ylabel("macro-F1", color="#E5E7EB")
    ax3b = ax3.twinx()
    ax3b.plot(x, t_mae, "--^", color=ORANGE, lw=1.6, label="tension MAE")
    ax3b.set_ylabel("tension MAE (lower better)", color=ORANGE)
    ax3b.grid(False)
    ax3.set_title("Ablation study — component contribution (PS-01 §38)",
                  fontsize=10, loc="left", color="#E5E7EB")
    h1, l1 = ax3.get_legend_handles_labels(); h2, l2 = ax3b.get_legend_handles_labels()
    ax3.legend(h1 + h2, l1 + l2, fontsize=7, framealpha=0, loc="lower right")

    # -- panel 4: radar of full-system capability
    ax4 = fig.add_axes([0.55, 0.07, 0.40, 0.38], polar=True)
    metrics = [
        ("sentiment F1", full["sentiment"]["f1_macro"]),
        ("emotion F1", full["emotion"]["f1_macro"]),
        ("tone F1", full["tone"]["f1_macro"]),
        ("sarcasm AUC", full["sarcasm"]["roc_auc"] or 0),
        ("irony AUC", full["irony"]["roc_auc"] or 0),
        ("PA AUC", full["passive_aggression"]["roc_auc"] or 0),
        ("escalation F1", full["escalation"]["f1_macro"]),
        ("tension R²", max(full["tension"]["r2"], 0)),
    ]
    N = len(metrics)
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False).tolist()
    vals = [v for _, v in metrics] + [metrics[0][1]]
    angles += angles[:1]
    ax4.plot(angles, vals, color=CYAN, lw=2)
    ax4.fill(angles, vals, color=CYAN, alpha=.14)
    ax4.set_xticks(angles[:-1])
    ax4.set_xticklabels([m for m, _ in metrics], fontsize=7.5, color="#E5E7EB")
    ax4.set_ylim(0, 1); ax4.set_yticks([.25, .5, .75, 1])
    ax4.set_yticklabels([".25", ".50", ".75", "1.0"], fontsize=6, color="#6B7280")
    ax4.set_title("Full CEREBRO (E) — test-split capability profile",
                  fontsize=10, loc="left", color="#E5E7EB", pad=18)

    watermark(fig, alpha=0.08, zoom=0.34, pos=(0.80, 0.32))
    fig.savefig(f"{GRAPHS}/hero_dashboard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ hero_dashboard.png")


# ---------------------------------------------------------------- graph 2
def graph_ablation():
    abl = load("ablations.json")
    full = load("summary.json")["full_metrics"]
    fig = plt.figure(figsize=(11.5, 6.2))
    hero_header(fig, "Ablation Study — what does each component contribute?",
                "All variants share the same training protocol; test split = 89 held-out conversations, sequential predicted-history inference")
    variants = ["A", "B", "C", "D"]
    x = np.arange(4); w = 0.27
    heads = [("sentiment", CYAN), ("emotion", PURPLE), ("tone", PINK)]
    for i, (h, c) in enumerate(heads):
        vals = [abl[v][h]["f1_macro"] for v in variants] + [full[h]["f1_macro"]]
        bars = plt.bar(x + (i - 1) * w, vals[:4], w, color=c, label=f"{h} macro-F1")
        plt.bar([3 + (i - 1) * w], [vals[4]], w, color=c, alpha=.35, hatch="//")
        for xi, v in zip(x + (i - 1) * w, vals):
            plt.text(xi, v + .012, f"{v:.3f}", ha="center", fontsize=7, color="#D1D5DB")
    plt.xticks(list(x) + [3], ["A\ntext only", "B\n+context", "C\n+speaker mem",
                               "D\n+behavior\n(full heads)", "E\nfull CEREBRO\n(+hidden-temporal fusion)"])
    plt.ylim(0, 1.14); plt.ylabel("macro-F1")
    plt.legend(fontsize=8, framealpha=0, loc="lower right", ncols=3)
    plt.title("Hatched bars = full system (engine D + hidden-signal fusion + temporal engines)",
              fontsize=9, color="#9CA3AF", loc="left")
    watermark(fig, alpha=0.07, zoom=0.30)
    fig.savefig(f"{GRAPHS}/ablation_study.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ ablation_study.png")


# ---------------------------------------------------------------- graph 3
def graph_baselines():
    base = load("baselines.json")
    full = load("summary.json")["full_metrics"]
    fig = plt.figure(figsize=(11.5, 6.2))
    hero_header(fig, "Baselines vs CEREBRO — every head, real test numbers",
                "B1: TF-IDF+LogisticRegression · B2: TF-IDF+LinearSVC · B3: TF-IDF+context · E: full CEREBRO")
    names = ["B1 TF-IDF+LR", "B2 TF-IDF+SVC", "B3 +context", "CEREBRO (E)"]
    keys = ["B1_tfidf_logreg", "B2_tfidf_svc", "B3_tfidf_context"]
    rows = [
        ("sentiment F1", [base[k]["sentiment"]["f1_macro"] for k in keys] + [full["sentiment"]["f1_macro"]], CYAN),
        ("emotion F1", [base[k]["emotion"]["f1_macro"] for k in keys] + [full["emotion"]["f1_macro"]], PURPLE),
        ("sarcasm ROC-AUC", [base[k]["sarcasm"]["roc_auc"] for k in keys] + [full["sarcasm"]["roc_auc"]], PINK),
        ("tension MAE ↓", [base[k]["tension"]["mae"] for k in keys] + [full["tension"]["mae"]], ORANGE),
    ]
    x = np.arange(4); w = 0.2
    for i, (lbl, vals, c) in enumerate(rows):
        bars = plt.bar(x + (i - 1.5) * w, vals, w, color=c, label=lbl)
        for xi, v in zip(x + (i - 1.5) * w, vals):
            plt.text(xi, v + .02, f"{v:.2f}", ha="center", fontsize=7, color="#D1D5DB")
    plt.xticks(x, names, fontsize=9)
    plt.ylabel("score (MAE in tension units)"); plt.ylim(0, 12)
    plt.legend(fontsize=8, framealpha=0, ncols=4, loc="upper left")
    watermark(fig, alpha=0.07, zoom=0.30)
    fig.savefig(f"{GRAPHS}/baselines_vs_cerebro.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ baselines_vs_cerebro.png")


# ---------------------------------------------------------------- post-hoc
def _test_predictions(n_convs=45):
    """Re-run the persisted engine on test conversations (same protocol)."""
    from cerebro.data.generator import generate_corpus, split_conversations
    from cerebro.models.engines import MultiTaskEngine
    from cerebro.models.pipeline import _add_behavior_flags
    from cerebro.models.hidden_signals import apply_hidden_signals

    corpus = generate_corpus(convs_per_cell=14)
    test = split_conversations(corpus)["test"][:n_convs]
    eng = MultiTaskEngine(seed=42).load("models/saved/cerebro_engine")
    yt, yp, proba = {"sent": [], "emo": [], "tone": []}, {"sent": [], "emo": [], "tone": []}, \
        {"sarcasm": [], "irony": [], "passive_aggression": []}
    yb, pb = [], []
    for c in test:
        results = eng.predict_conversation(c)
        apply_hidden_signals(results)
        _add_behavior_flags(results)
        for r, m in zip(results, c):
            yt["sent"].append(m["sentiment"]); yp["sent"].append(r["sentiment"]["label"])
            yt["emo"].append(m["emotion"]); yp["emo"].append(r["emotion"]["label"])
            yt["tone"].append(m["tone"]); yp["tone"].append(r["tone"]["label"])
            for h in proba:
                proba[h].append((int(m[h]), r[h]["probability"]))
            yb.append(int(float(m["tension"]) >= 60))
            pb.append(int(r["tension"] >= 60))
    return yt, yp, proba, yb, pb


def graph_confusions():
    from sklearn.metrics import confusion_matrix
    from cerebro.common.labels import SENTIMENT_LABELS, EMOTION_LABELS, TONE_LABELS
    yt, yp, _, _, _ = _test_predictions()
    fig = plt.figure(figsize=(14.5, 5.0))
    hero_header(fig, "Confusion structure — sentiment · emotion · tone (test split)",
                "Persisted CEREBRO engine re-run on 45 held-out conversations (sequential inference)")
    for k, (title, labels, color) in enumerate([
            ("Sentiment", SENTIMENT_LABELS, CYAN),
            ("Emotion (13-way)", EMOTION_LABELS, PURPLE),
            ("Tone (14-way)", TONE_LABELS, PINK)]):
        ax = fig.add_axes([0.06 + k * 0.32, 0.08, 0.27, 0.72])
        key = {"Sentiment": "sent", "Emotion (13-way)": "emo", "Tone (14-way)": "tone"}[title]
        cm = confusion_matrix(yt[key], yp[key], labels=labels, normalize="true")
        im = ax.imshow(cm, cmap=matplotlib.colors.LinearSegmentedColormap.from_list(
            "neon", [BG, color]), vmin=0, vmax=1)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=6)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels if k == 0 else [], fontsize=6)
        thr = cm.max() * 0.55
        for i in range(len(labels)):
            for j in range(len(labels)):
                if cm[i, j] >= 0.005:
                    ax.text(j, i, f"{cm[i,j]:.2f}", ha="center", va="center",
                            fontsize=5.2, color="white" if cm[i, j] > thr else "#9CA3AF")
        ax.set_title(title, fontsize=10, loc="left", color=color)
        ax.grid(False)
    fig.colorbar(im, ax=fig.axes[-1], fraction=.046, pad=.03)
    watermark(fig, alpha=0.06, zoom=0.26, pos=(0.855, 0.06))
    fig.savefig(f"{GRAPHS}/confusion_heatmaps.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ confusion_heatmaps.png")


def graph_calibration():
    from sklearn.calibration import calibration_curve
    _, _, proba, _, _ = _test_predictions()
    fig = plt.figure(figsize=(11.5, 5.6))
    hero_header(fig, "Probability calibration — hidden-signal heads",
                "Reliability diagrams on the test split; Brier score annotated (lower = better calibrated)")
    for k, (name, color) in enumerate([("sarcasm", PINK), ("irony", PURPLE),
                                       ("passive_aggression", GREEN)]):
        ax = fig.add_axes([0.07 + k * 0.31, 0.14, 0.25, 0.66])
        y = np.array([p[0] for p in proba[name]])
        p = np.array([p[1] for p in proba[name]])
        frac_pos, mean_p = calibration_curve(y, p, n_bins=8, strategy="quantile")
        ax.plot(mean_p, frac_pos, "-o", color=color, lw=2, ms=5)
        ax.plot([0, 1], [0, 1], "--", color="#4B5563", lw=1, label="perfectly calibrated")
        brier = float(np.mean((p - y) ** 2))
        ax.set_title(f"{name}  ·  Brier {brier:.3f}", fontsize=10, loc="left", color=color)
        ax.set_xlabel("predicted probability"); ax.set_ylabel("observed frequency")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.legend(fontsize=7, framealpha=0, loc="upper left")
    watermark(fig, alpha=0.07, zoom=0.28)
    fig.savefig(f"{GRAPHS}/calibration_curves.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ calibration_curves.png")


def graph_dataset():
    from cerebro.data.generator import generate_corpus, split_conversations
    corpus = generate_corpus(convs_per_cell=14)
    splits = split_conversations(corpus)
    msgs = [m for c in corpus for m in c]
    fig = plt.figure(figsize=(12.5, 5.6))
    hero_header(fig, "CEREBRO corpus — 6 domains × 7 narrative arcs",
                "Weak-supervision annotated at generation time; conversation-level splits (no context leakage)")

    # sentiment distribution per arc
    arcs = ["calm", "positive", "friction", "escalation", "sarcasm", "passive", "mixed"]
    ax = fig.add_axes([0.06, 0.13, 0.40, 0.66])
    by_arc = {a: Counter() for a in arcs}
    for c in corpus:
        arc = None
        for a in arcs:
            if a in c[0]["conversation_id"] or True:
                pass
        # recover arc from tension profile signature
    # simpler: tension histogram colored by sentiment
    for m in msgs:
        arc_guess = ("calm" if m["tension"] < 25 else
                     "friction" if m["tension"] < 55 else
                     "escalation" if m["tension"] < 85 else "peak")
        by_arc[arc_guess] = by_arc.get(arc_guess, Counter())
        by_arc[arc_guess][m["sentiment"]] += 1
    bands = ["calm", "friction", "escalation", "peak"]
    pos = [by_arc[b]["positive"] for b in bands]
    neu = [by_arc[b]["neutral"] for b in bands]
    neg = [by_arc[b]["negative"] for b in bands]
    x = np.arange(len(bands))
    ax.bar(x, pos, .55, color=GREEN, label="positive")
    ax.bar(x, neu, .55, bottom=pos, color="#60A5FA", label="neutral")
    ax.bar(x, neg, .55, bottom=[p + n for p, n in zip(pos, neu)], color=PINK, label="negative")
    totals = [p + n + g for p, n, g in zip(pos, neu, neg)]
    for xi, t in zip(x, totals):
        ax.text(xi, t + 150, f"{t:,}", ha="center", fontsize=7.5, color="#9CA3AF")
    ax.set_xticks(x); ax.set_xticklabels([f"{b}\n(tension band)" for b in bands], fontsize=8)
    ax.set_ylabel("messages"); ax.legend(fontsize=8, framealpha=0)
    ax.set_title("Sentiment composition across tension bands", fontsize=10, loc="left")

    # messages per conversation + tension histogram
    ax2 = fig.add_axes([0.56, 0.13, 0.38, 0.66])
    lens = [len(c) for c in corpus]
    tens = [m["tension"] for m in msgs]
    ax2.hist(lens, bins=18, color=CYAN, alpha=.85, label="messages / conversation")
    ax3 = ax2.twiny()
    ax3.hist(tens, bins=30, color=PURPLE, alpha=.55, label="tension")
    ax2.set_xlabel("messages per conversation", color=CYAN)
    ax3.set_xlabel("tension value", color=PURPLE)
    ax2.set_ylabel("conversations", color=CYAN); ax3.grid(False)
    ax2.set_title("Conversation length & tension distributions", fontsize=10, loc="left")
    watermark(fig, alpha=0.07, zoom=0.30)
    fig.savefig(f"{GRAPHS}/dataset_overview.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ dataset_overview.png")


def graph_architecture():
    """Architecture diagram drawn natively (no external assets)."""
    fig = plt.figure(figsize=(14, 8.6))
    hero_header(fig, "CEREBRO system architecture",
                "Raw chat → parsers → context/temporal intelligence → explainable report (PS-01 §46)")
    ax = fig.add_axes([0, 0, 1, 1]); ax.axis("off"); ax.set_xlim(0, 100); ax.set_ylim(0, 100)

    def box(x, y, w, h, text, color, fs=8):
        ax.add_patch(plt.Rectangle((x, y), w, h, fill=True, facecolor=PANEL,
                                   edgecolor=color, lw=1.6, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color="#E5E7EB", zorder=3)

    def arrow(x1, y1, x2, y2, color="#374151"):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=1.4))

    L1 = [
        ("WhatsApp\n.txt", 6), ("Discord", 22), ("Slack", 36), ("CSV / JSON", 50), ("generic", 64),
    ]
    for txt, x in L1:
        box(x, 88, 13, 8, txt, CYAN)
        arrow(x + 6.5, 88, 41.5, 82)
    box(35, 74, 30, 8, "CHAT PARSER  →  common format", CYAN, 9)
    arrow(50, 74, 50, 69)
    box(30, 61, 40, 8, "PREPROCESSING  +  BEHAVIORAL FEATURES (16-dim)", PURPLE, 8.5)
    arrow(50, 61, 50, 56)
    box(12, 46, 34, 10, "CONTEXT ENGINE\nsliding window + long-range summary", GREEN, 8.5)
    box(54, 46, 34, 10, "SPEAKER MEMORY\nper-speaker emotional state", GREEN, 8.5)
    arrow(40, 56, 29, 50); arrow(60, 56, 71, 50)
    arrow(29, 46, 44, 40); arrow(71, 46, 56, 40)
    box(24, 30, 52, 10, "MULTI-TASK NLP ENGINE  ·  shared representation → 7 heads\n"
                        "sentiment · emotion · tone · tension · sarcasm · irony · passive-aggression",
        YELLOW, 8.5)
    arrow(50, 30, 50, 25)
    box(8, 15, 24, 10, "TEMPORAL ENGINES\narc · transitions · turning\npoints · escalation", PINK, 8)
    box(38, 15, 24, 10, "HIDDEN-SIGNAL FUSION\ncontradiction + trajectory\nevidence", PINK, 8)
    box(68, 15, 24, 10, "MODEL FUSION\nvalidation-tuned streams\n+ calibration", PINK, 8)
    arrow(20, 15, 42, 8); arrow(50, 15, 52, 8); arrow(80, 15, 58, 8)
    box(28, 1, 44, 7, "EXPLAINABILITY ENGINE → REPORT → DASHBOARD", ORANGE, 9)
    watermark(fig, alpha=0.05, zoom=0.26, pos=(0.86, 0.42))
    fig.savefig("docs/architecture/architecture.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("  ✓ architecture.png")


if __name__ == "__main__":
    print("generating graphs from real results...")
    graph_hero()
    graph_ablation()
    graph_baselines()
    graph_dataset()
    graph_architecture()
    try:
        graph_confusions()
        graph_calibration()
    except FileNotFoundError:
        print("  ⚠ engine not saved yet — skipping confusion/calibration graphs")
    print("done → assets/graphs/")
