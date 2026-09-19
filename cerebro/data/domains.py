"""Domain dialogue templates for the CEREBRO corpus.

Each domain defines a cast, topics, and a library of turn templates in three
arousal bands (calm / strained / hot). The generator composes them into 7 arc
patterns (PS-01 §3). Names are unisex-neutral to avoid gender confounds.
"""

# turn template: (text, sentiment, emotion, tone, sarcasm, irony, pa, tension)
# tension values are the domain-neutral "heat" of the turn: 0-100.

CALM = [
    ("Hey! Did you get a chance to look at the plan?", "neutral", "neutral", "friendly", 0, 0, 0, 12),
    ("Yeah I saw it last night, looks solid overall.", "positive", "relief", "casual", 0, 0, 0, 10),
    ("Awesome. I think the deadline is Friday, right?", "neutral", "neutral", "casual", 0, 0, 0, 14),
    ("Friday it is. I'll finish my part tonight.", "positive", "neutral", "friendly", 0, 0, 0, 12),
    ("Perfect, thanks! I'll review in the morning.", "positive", "joy", "friendly", 0, 0, 0, 10),
    ("Cool cool. Ping me if anything is unclear.", "positive", "neutral", "supportive", 0, 0, 0, 8),
    ("Will do. The weather is actually nice today.", "positive", "joy", "casual", 0, 0, 0, 6),
    ("Right? First decent day all week.", "positive", "joy", "friendly", 0, 0, 0, 6),
    ("How did your exam go btw?", "neutral", "concerned", "friendly", 0, 0, 0, 10),
    ("Honestly better than I expected, I think I passed.", "positive", "relief", "casual", 0, 0, 0, 10),
    ("That's great, congrats!", "positive", "joy", "friendly", 0, 0, 0, 8),
    ("Thanks! Celebrating this weekend.", "positive", "excitement", "casual", 0, 0, 0, 8),
    ("Just finished the setup, everything works now.", "positive", "relief", "casual", 0, 0, 0, 12),
    ("Good. I'll send the summary in a bit.", "neutral", "neutral", "professional", 0, 0, 0, 12),
    ("Thanks, that was fast.", "positive", "affection", "friendly", 0, 0, 0, 8),
    ("No worries, happy to help.", "positive", "affection", "supportive", 0, 0, 0, 6),
    ("Morning! Ready for the call at 10?", "neutral", "neutral", "professional", 0, 0, 0, 12),
    ("Yep, joined five minutes early.", "positive", "neutral", "professional", 0, 0, 0, 10),
    # surprise + humorous coverage (blueprint §4): without these the two classes
    # have zero support and the heads can never emit them
    ("Wait, you finished the whole thing already? That's amazing!", "positive", "surprise", "friendly", 0, 0, 0, 16),
    ("No way! I was sure the deadline was next month.", "positive", "surprise", "casual", 0, 0, 0, 18),
    ("Okay that diagram is genuinely too funny.", "positive", "joy", "humorous", 0, 0, 0, 8),
    ("Ha! You should put that line on a slide.", "positive", "joy", "humorous", 0, 0, 0, 8),
]

STRAINED = [
    ("You said you'd send it yesterday. It's not here.", "negative", "frustration", "frustrated", 0, 0, 0, 46),
    ("I know, I'm sorry. The day got completely away from me.", "negative", "sadness", "apologetic", 0, 0, 0, 38),
    ("That's the second time this week.", "negative", "frustration", "cold", 0, 0, 0, 52),
    ("I understand. It won't happen again.", "neutral", "anxiety", "apologetic", 0, 0, 0, 40),
    ("We keep going in circles on this.", "negative", "frustration", "dismissive", 0, 0, 0, 55),
    ("Because nobody actually reads what I write.", "negative", "sadness", "defensive", 0, 0, 0, 50),
    ("That's not fair, I read everything.", "negative", "anger", "defensive", 0, 0, 0, 58),
    ("Fine. Do it your way then.", "negative", "frustration", "passive-aggressive", 0, 0, 1, 62),
    ("What is that supposed to mean?", "negative", "confusion", "defensive", 0, 0, 0, 60),
    ("Nothing. Forget it.", "negative", "sadness", "cold", 0, 0, 1, 64),
    ("I don't want to fight about a slideshow.", "neutral", "anxiety", "concerned", 0, 0, 0, 44),
    ("Neither do I, but the deadline is real.", "negative", "anxiety", "professional", 0, 0, 0, 48),
    ("Okay. I'll take the evening slot and redo it.", "neutral", "relief", "professional", 0, 0, 0, 34),
    ("Thank you. Seriously, I appreciate it.", "positive", "relief", "apologetic", 0, 0, 0, 26),
    ("You said that yesterday too.", "negative", "frustration", "frustrated", 0, 0, 0, 56),
    ("And I meant it. Today things just slipped.", "negative", "anxiety", "apologetic", 0, 0, 0, 46),
    ("The Wifi died mid-call, of all days.", "negative", "frustration", "casual", 0, 0, 0, 42),
    ("Of course it did. This week, huh.", "neutral", "neutral", "casual", 0, 0, 0, 40),
    ("Huh. So the file was never attached at all? That explains it.", "neutral", "surprise", "concerned", 0, 0, 0, 44),
    ("Honestly, the way this broke is almost funny.", "neutral", "frustration", "humorous", 0, 0, 0, 42),
]

HOT = [
    ("Oh great, perfect, this is EXACTLY what I needed today.", "positive", "anger", "sarcastic", 1, 1, 0, 82),
    ("Wow. Relax. It's one file.", "negative", "disgust", "dismissive", 0, 1, 0, 78),
    ("One file? ONE FILE? It's the whole submission!", "negative", "anger", "aggressive", 0, 0, 0, 90),
    ("There's no need to shout at me.", "negative", "fear", "defensive", 0, 0, 0, 74),
    ("I'm not shouting, I'm just done being polite about this.", "negative", "anger", "aggressive", 0, 0, 0, 88),
    ("Done being polite. Wow, okay, that's rich.", "positive", "disgust", "sarcastic", 1, 1, 0, 86),
    ("You know what, do whatever you want. Like you always do.", "negative", "anger", "passive-aggressive", 0, 0, 1, 92),
    ("Every single time. You never change.", "negative", "anger", "aggressive", 0, 0, 0, 90),
    ("That's a lie and you know it.", "negative", "anger", "aggressive", 0, 0, 0, 94),
    ("Right, and I'm the villain of course.", "positive", "sadness", "sarcastic", 1, 1, 0, 88),
    ("This is unbelievable. Absolutely unbelievable.", "negative", "anger", "aggressive", 0, 0, 0, 92),
    ("I literally cannot talk to you right now.", "negative", "frustration", "cold", 0, 0, 0, 90),
    ("Perfect, walk away. It's what you're best at.", "positive", "anger", "sarcastic", 1, 1, 1, 96),
    ("You always do this. EVERY single time.", "negative", "anger", "aggressive", 0, 0, 0, 92),
    ("Stop. Both of you. This is getting us nowhere.", "negative", "anxiety", "concerned", 0, 0, 0, 80),
    ("...I didn't mean to blow up like that. I'm sorry.", "negative", "sadness", "apologetic", 0, 0, 0, 48),
    ("Me neither. This week just broke me a bit.", "negative", "sadness", "apologetic", 0, 0, 0, 40),
    ("Alright. Deep breath. Let's just fix the file.", "neutral", "relief", "supportive", 0, 0, 0, 28),
    ("Oh, we're doing this again tonight? Wonderful.", "positive", "surprise", "sarcastic", 1, 0, 0, 85),
    ("Great, so now we're laughing about the deadline. Incredible.", "positive", "disgust", "humorous", 1, 1, 0, 84),
]

DOMAINS = {
    "work_project": {
        "cast": ["Aarav", "Meera"],
        "seed_topics": ["project deadline", "the client demo", "the shared slide deck"],
        "bands": (CALM, STRAINED, HOT),
        "surface": [
            "about the {topic}", "regarding {topic}", "for the {topic}",
        ],
    },
    "college_team": {
        "cast": ["Riya", "Dev"],
        "seed_topics": ["the lab report", "the professor's feedback", "the group assignment"],
        "bands": (CALM, STRAINED, HOT),
        "surface": ["with {topic}", "on {topic}", "about {topic}"],
    },
    "startup_founders": {
        "cast": ["Kabir", "Ananya"],
        "seed_topics": ["the investor update", "runway", "the hiring plan"],
        "bands": (CALM, STRAINED, HOT),
        "surface": ["on {topic}", "for {topic}", "around {topic}"],
    },
    "flatmates": {
        "cast": ["Nikhil", "Sara"],
        "seed_topics": ["the electricity bill", "dishes in the sink", "the wifi router"],
        "bands": (CALM, STRAINED, HOT),
        "surface": ["about {topic}", "with {topic}", "over {topic}"],
    },
    "customer_support": {
        "cast": ["Customer", "Support"],
        "seed_topics": ["the refund", "the delayed order", "the billing error"],
        "bands": (CALM, STRAINED, HOT),
        "surface": ["regarding {topic}", "for {topic}", "about {topic}"],
    },
    "old_friends": {
        "cast": ["Vikram", "Tara"],
        "seed_topics": ["the trip plan", "the reunion date", "the booking"],
        "bands": (CALM, STRAINED, HOT),
        "surface": ["for {topic}", "about {topic}", "on {topic}"],
    },
}
