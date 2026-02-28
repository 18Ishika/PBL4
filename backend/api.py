from fastapi import FastAPI
from pydantic import BaseModel
from transformers import pipeline
from youtube_transcript_api import YouTubeTranscriptApi
from sentence_transformers import SentenceTransformer, util
import yt_dlp
import re

app = FastAPI(title="Advanced Clickbait Detector")

# ===============================
# Models
# ===============================
# Loads the clickbait classifier
classifier = pipeline("text-classification", model="Stremie/bert-base-uncased-clickbait")
# Loads the similarity embedder
embedder = SentenceTransformer("all-MiniLM-L6-v2")


class VideoData(BaseModel):
    url: str


CLICKBAIT_WORDS = [
    "shocking", "unbelievable", "secret", "exposed", "you won't believe",
    "insane", "trick", "overnight", "millionaire", "scam", "warning",
    "must watch", "gone wrong", "leaked"
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
        # Suppresses JS runtime warning — uses nodejs if available
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
    """
    Compatible with youtube-transcript-api v1.0+
    Falls back through multiple strategies to maximize transcript coverage.

    Strategy 1: Direct fetch with English language codes
    Strategy 2: List all transcripts and fetch any available one
    """
    try:
        ytt_api = YouTubeTranscriptApi()

        # Strategy 1: Try fetching English directly
        english_codes = ['en', 'en-US', 'en-GB', 'en-AU', 'en-CA']
        try:
            fetched = ytt_api.fetch(video_id, languages=english_codes)
            text = " ".join(snippet.text for snippet in fetched)
            if text.strip():
                print(f"[Transcript] Strategy 1 success — English transcript fetched ({len(text)} chars)")
                return text
        except Exception as e:
            print(f"[Transcript] Strategy 1 failed (English fetch): {e}")

        # Strategy 2: List all available transcripts and try each one
        try:
            transcript_list = ytt_api.list(video_id)
            for transcript in transcript_list:
                try:
                    fetched = transcript.fetch()
                    text = " ".join(snippet.text for snippet in fetched)
                    if text.strip():
                        print(f"[Transcript] Strategy 2 success — language: {transcript.language_code} ({len(text)} chars)")
                        return text
                except Exception as inner_e:
                    print(f"[Transcript] Could not fetch language {transcript.language_code}: {inner_e}")
                    continue
        except Exception as e:
            print(f"[Transcript] Strategy 2 failed (list transcripts): {e}")

        print(f"[Transcript] All strategies exhausted for video: {video_id}")
        return None

    except Exception as e:
        print(f"[Transcript] Fatal error for {video_id}: {e}")
        return None


def similarity_score(title, content):
    if not content or not content.strip():
        return 0
    # Process only the first 3000 chars for speed/context
    content = content[:3000]
    emb1 = embedder.encode(title, convert_to_tensor=True)
    emb2 = embedder.encode(content, convert_to_tensor=True)
    sim = util.cos_sim(emb1, emb2)
    return round(float(sim[0][0]) * 100, 2)


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
    # Model predicts CLICKBAIT or NOT_CLICKBAIT
    # We normalize so ai_score always = clickbait probability
    # -------------------------------------------------------
    ai_result = classifier(title)[0]
    label = ai_result["label"].upper()
    if label in ("NOT_CLICKBAIT", "LABEL_0"):
        ai_score = round((1 - ai_result["score"]) * 100, 2)
    else:
        ai_score = round(ai_result["score"] * 100, 2)

    # -------------------------------------------------------
    # Score 2: Rule-based Score (keyword + formatting checks)
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
    rule_score = min(rule_score, 40)  # cap at 40

    # -------------------------------------------------------
    # Score 3: Similarity Score
    # High similarity = title matches content = less clickbait
    # -------------------------------------------------------
    sim_score = similarity_score(title, full_text)

    # -------------------------------------------------------
    # Final Score: Dynamic Weighting
    # With transcript    → AI(45%) + Rule(25%) + Mismatch(30%)
    # Without transcript → AI(60%) + Rule(40%)
    # -------------------------------------------------------
    if used_transcript:
        final = (ai_score * 0.45) + (rule_score * 0.25) + ((100 - sim_score) * 0.30)
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

    print(f"[Predict] AI={ai_score}  Rule={rule_score}  Sim={sim_score}  Final={final}  Level={level}")
    print(f"{'='*60}\n")

    return {
        "title": title,
        "video_id": video_id,
        "ai_score": ai_score,
        "rule_score": rule_score,
        "similarity_score": sim_score,
        "transcript_used": used_transcript,
        "transcript_length": len(transcript) if transcript else 0,
        "final_score": final,
        "risk_level": level,
        "clickbait": final >= 55
    }