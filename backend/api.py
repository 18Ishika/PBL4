from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer, util
import yt_dlp
import spacy
import re

app = FastAPI(title="Advanced Clickbait Detector")

# ===============================
# Models
# ===============================
classifier = pipeline("text-classification", model="Stremie/bert-base-uncased-clickbait")
embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm")


class VideoData(BaseModel):
    url: str


CLICKBAIT_WORDS = [
    "shocking", "unbelievable", "secret", "exposed", "you won't believe",
    "insane", "trick", "overnight", "millionaire", "scam", "warning",
    "must watch", "gone wrong", "leaked"
]

EMOTIONAL_WORDS = [
    "shocking", "heartbreaking", "terrifying", "incredible", "destroyed",
    "exposed", "crying", "brutal", "explosive", "devastating", "outraged",
    "furious", "insane", "unreal", "mind-blowing"
]

# ===============================
# Utils
# ===============================

def extract_id(url: str):
    patterns = [r"v=([^&]+)", r"youtu\.be/([^?]+)", r"shorts/([^?]+)"]
    for p in patterns:
        match = re.search(p, url)
        if match:
            return match.group(1)
    return None


def get_video_info(url):
    ydl_opts = {
        "quiet": True,
        "skip_download": True,
        "noplaylist": True,
        "extractor_args": {"youtube": {"js_runtimes": ["nodejs"]}}
    }
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title"), info.get("description")
    except Exception as e:
        print(f"[VideoInfo] yt-dlp error: {e}")
        return None, None


def get_transcript(video_id):
    try:
        ytt_api = YouTubeTranscriptApi()

        english_codes = ['en', 'en-US', 'en-GB', 'en-AU', 'en-CA']
        try:
            fetched = ytt_api.fetch(video_id, languages=english_codes)
            text = " ".join(snippet.text for snippet in fetched)
            if text.strip():
                print(f"[Transcript] Strategy 1 success ({len(text)} chars)")
                return text
        except Exception as e:
            print(f"[Transcript] Strategy 1 failed: {e}")

        try:
            transcript_list = ytt_api.list(video_id)
            for transcript in transcript_list:
                try:
                    fetched = transcript.fetch()
                    text = " ".join(snippet.text for snippet in fetched)
                    if text.strip():
                        print(f"[Transcript] Strategy 2 success — {transcript.language_code} ({len(text)} chars)")
                        return text
                except Exception as inner_e:
                    print(f"[Transcript] Could not fetch {transcript.language_code}: {inner_e}")
                    continue
        except Exception as e:
            print(f"[Transcript] Strategy 2 failed: {e}")

        return None
    except Exception as e:
        print(f"[Transcript] Fatal error for {video_id}: {e}")
        return None


def similarity_score(title, content):
    if not content or not content.strip():
        return 0
    content = content[:3000]
    emb1 = embedder.encode(title, convert_to_tensor=True)
    emb2 = embedder.encode(content, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2)
    return round(float(sim[0][0]) * 100, 2)


# ===============================
# NEW: Triggered clickbait words
# ===============================
def get_triggered_words(title: str):
    t_lower = title.lower()
    return [word for word in CLICKBAIT_WORDS if word in t_lower]


# ===============================
# NEW: Emotion/sensationalism score
# ===============================
def get_emotion_score(title: str):
    t_lower = title.lower()
    hits = [w for w in EMOTIONAL_WORDS if w in t_lower]
    score = min(len(hits) * 20, 100)
    return {"score": score, "words": hits}


# ===============================
# NEW: Named entity mismatch
# ===============================
def get_entity_mismatch(title: str, transcript: str):
    doc = nlp(title)
    title_entities = {ent.text.lower() for ent in doc.ents if ent.label_ in ("PERSON", "ORG", "GPE", "PRODUCT", "EVENT")}
    if not title_entities or not transcript:
        return {"score": 0, "missing_entities": [], "total_entities": 0}
    transcript_lower = transcript.lower()
    missing = [e for e in title_entities if e not in transcript_lower]
    score = round((len(missing) / len(title_entities)) * 100, 2)
    return {
        "score": score,
        "missing_entities": missing,
        "total_entities": len(title_entities)
    }


# ===============================
# NEW: Human-readable explanation
# ===============================
def generate_explanation(triggered_words, emotion_data, entity_data, ai_score, sim_score):
    reasons = []
    if triggered_words:
        reasons.append(f"Contains clickbait phrases: {', '.join(triggered_words)}")
    if emotion_data["words"]:
        reasons.append(f"Uses emotionally charged words: {', '.join(emotion_data['words'])}")
    if entity_data["missing_entities"]:
        reasons.append(f"Title mentions {', '.join(entity_data['missing_entities'])} but video doesn't cover it")
    if ai_score > 60:
        reasons.append("AI model flagged title as highly sensationalist")
    if sim_score < 30:
        reasons.append("Title poorly matches actual video content")
    return reasons if reasons else ["No strong clickbait signals detected"]


# ===============================
# API Routes
# ===============================

@app.get("/")
def root():
    return {
        "message": "Advanced Clickbait Detector API is running.",
        "usage": "POST to /predict with JSON body: {\"url\": \"https://youtube.com/watch?v=...\"}"
    }


@app.post("/predict")
def predict(data: VideoData):
    # Step 1: Extract video ID
    video_id = extract_id(data.url)
    if not video_id:
        return {"error": "Invalid YouTube URL. Supported formats: watch?v=, youtu.be/, shorts/"}

    # Step 2: Fetch title and description
    title, desc = get_video_info(data.url)
    if not title:
        return {"error": "Could not fetch video info. Check the URL or try again later."}

    print(f"\n{'='*60}")
    print(f"[Predict] Title    : {title}")
    print(f"[Predict] Video ID : {video_id}")

    # Step 3: Fetch transcript
    transcript = get_transcript(video_id)
    used_transcript = transcript is not None

    if used_transcript:
        print(f"[Predict] Transcript : YES — {len(transcript)} chars")
    else:
        print(f"[Predict] Transcript : NO — using fallback weighting")

    # Step 4: Build full comparison text
    full_text = (desc or "") + " " + (transcript or "")

    # -------------------------------------------------------
    # Score 1: AI Score
    # -------------------------------------------------------
    ai_result = classifier(title)[0]
    label = ai_result["label"].upper()
    if label in ("NOT_CLICKBAIT", "LABEL_0"):
        ai_score = round((1 - ai_result["score"]) * 100, 2)
    else:
        ai_score = round(ai_result["score"] * 100, 2)

    # -------------------------------------------------------
    # Score 2: Rule-based Score
    # -------------------------------------------------------
    rule_score = 0
    t_lower = title.lower()
    for word in CLICKBAIT_WORDS:
        if word in t_lower:
            rule_score += 8
    rule_score += min(title.count("!") * 2, 10)
    rule_score += min(title.count("?") * 2, 10)
    if title.isupper() and len(title) > 10:
        rule_score += 10
    if re.search(r"\d+", title):
        rule_score += 5
    rule_score = min(rule_score, 40)

    # -------------------------------------------------------
    # Score 3: Similarity Score
    # -------------------------------------------------------
    sim_score = similarity_score(title, full_text)

    # -------------------------------------------------------
    # NEW Score 4: Entity Mismatch Score
    # -------------------------------------------------------
    entity_data = get_entity_mismatch(title, transcript)

    # -------------------------------------------------------
    # NEW: Emotion Score
    # -------------------------------------------------------
    emotion_data = get_emotion_score(title)

    # -------------------------------------------------------
    # NEW: Triggered clickbait words
    # -------------------------------------------------------
    triggered_words = get_triggered_words(title)

    # -------------------------------------------------------
    # Final Score: Dynamic Weighting
    # With transcript    → AI(40%) + Rule(20%) + Sim(25%) + Entity(15%)
    # Without transcript → AI(60%) + Rule(40%)
    # -------------------------------------------------------
    if used_transcript:
        final = (
            (ai_score * 0.40) +
            (rule_score * 0.20) +
            ((100 - sim_score) * 0.25) +
            (entity_data["score"] * 0.15)
        )
    else:
        final = (ai_score * 0.60) + (rule_score * 0.40)

    final = round(final, 2)

    # Classification Level
    if final >= 75:
        level = "Very High"
    elif final >= 60:
        level = "High"
    elif final >= 45:
        level = "Medium"
    else:
        level = "Low"

    # NEW: Explanation
    explanation = generate_explanation(triggered_words, emotion_data, entity_data, ai_score, sim_score)

    print(f"[Predict] AI={ai_score}  Rule={rule_score}  Sim={sim_score}  Entity={entity_data['score']}  Final={final}  Level={level}")
    print(f"{'='*60}\n")

    return {
        "title": title,
        "video_id": video_id,

        # --- Scores ---
        "ai_score": ai_score,
        "rule_score": rule_score,
        "similarity_score": sim_score,
        "entity_mismatch_score": entity_data["score"],
        "emotion_score": emotion_data["score"],
        "final_score": final,
        "risk_level": level,
        "clickbait": final >= 55,

        # --- Insights (NEW — shown in extension) ---
        "triggered_words": triggered_words,
        "emotion_words": emotion_data["words"],
        "missing_entities": entity_data["missing_entities"],
        "explanation": explanation,

        # --- Meta ---
        "transcript_used": used_transcript,
        "transcript_length": len(transcript) if transcript else 0,
    }