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
        if match: return match.group(1)
    return None

def get_video_info(url):
    ydl_opts = {"quiet": True, "skip_download": True, "noplaylist": True}
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            return info.get("title"), info.get("description")
    except Exception as e:
        print(f"yt-dlp error: {e}")
        return None, None

def get_transcript(video_id):
    try:
        # Improved: Look for manual OR auto-generated English
        transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
        transcript = transcript_list.find_transcript(['en'])
        data = transcript.fetch()
        return " ".join(i["text"] for i in data)
    except Exception as e:
        print(f"Transcript fetch failed for {video_id}: {e}")
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
# API Route
# ===============================

@app.post("/predict")
def predict(data: VideoData):
    video_id = extract_id(data.url)
    if not video_id:
        return {"error": "Invalid YouTube URL"}

    title, desc = get_video_info(data.url)
    if not title:
        return {"error": "Could not fetch video info"}

    # Fetch transcript with the new improved method
    transcript = get_transcript(video_id)
    used_transcript = transcript is not None
    
    # Create comparison text (Description + Transcript)
    full_text = (desc or "") + " " + (transcript or "")

    # 1. AI Score (Weight: 45%)
    ai_result = classifier(title)[0]
    ai_score = round(ai_result["score"] * 100, 2)

    # 2. Rule Score (Weight: 25%)
    rule_score = 0
    t_lower = title.lower()
    for word in CLICKBAIT_WORDS:
        if word in t_lower: rule_score += 8
    rule_score += min(title.count("!") * 2, 10) + min(title.count("?") * 2, 10)
    if title.isupper() and len(title) > 10: rule_score += 10
    if re.search(r"\d+", title): rule_score += 5
    rule_score = min(rule_score, 40)

    # 3. Similarity Score (Weight: 30%)
    sim_score = similarity_score(title, full_text)

    # ----------------------------
    # DYNAMIC WEIGHTING LOGIC
    # ----------------------------
    if used_transcript:
        # Full data available
        final = (ai_score * 0.45) + (rule_score * 0.25) + ((100 - sim_score) * 0.30)
    else:
        # Shift weight away from similarity if transcript is missing
        # This prevents "punishing" the video for a 0 similarity score
        final = (ai_score * 0.60) + (rule_score * 0.40)

    final = round(final, 2)
    
    # Classification Level
    if final >= 75: level = "Very High"
    elif final >= 60: level = "High"
    elif final >= 45: level = "Medium"
    else: level = "Low"

    return {
        "title": title,
        "ai_score": ai_score,
        "rule_score": rule_score,
        "similarity_score": sim_score,
        "transcript_used": used_transcript,
        "final_score": final,
        "risk_level": level,
        "clickbait": final >= 55
    }