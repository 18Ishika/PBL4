// popup.js — handles API call and UI rendering

const API_URL = "http://127.0.0.1:8000/predict";

const states = {
  idle:    document.getElementById("state-idle"),
  loading: document.getElementById("state-loading"),
  result:  document.getElementById("state-result"),
  error:   document.getElementById("state-error"),
};

const btn = document.getElementById("analyse-btn");

// ── Show a state ──────────────────────────────────────────
function showState(name) {
  Object.values(states).forEach(el => el.classList.remove("active"));
  states[name].classList.add("active");
}

// ── Risk level config ─────────────────────────────────────
const RISK_CONFIG = {
  "Very High": { cls: "very-high", icon: "🚨", barColor: "#ff3b3b" },
  "High":      { cls: "high",      icon: "⚠️",  barColor: "#ff8c00" },
  "Medium":    { cls: "medium",    icon: "🟡",  barColor: "#ffd60a" },
  "Low":       { cls: "low",       icon: "✅",  barColor: "#00e676" },
};

// ── Render result ─────────────────────────────────────────
function renderResult(data) {
  const cfg = RISK_CONFIG[data.risk_level] || RISK_CONFIG["Low"];

  // Banner
  const banner = document.getElementById("risk-banner");
  banner.className = `risk-banner ${cfg.cls}`;
  document.getElementById("risk-icon").textContent = cfg.icon;
  document.getElementById("risk-level-text").textContent = data.risk_level;

  // Score bar
  const bar = document.getElementById("score-bar");
  bar.style.width = `${Math.min(data.final_score, 100)}%`;
  bar.style.background = cfg.barColor;
  document.getElementById("score-num").textContent = data.final_score.toFixed(1);
  document.getElementById("score-num").style.color = cfg.barColor;

  // Title
  document.getElementById("result-title").textContent = data.title;

  // ── Breakdown cards (original 3 + 2 new) ─────────────────
  document.getElementById("card-ai").textContent            = data.ai_score.toFixed(1);
  document.getElementById("card-rule").textContent          = data.rule_score;
  document.getElementById("card-sim").textContent           = data.similarity_score.toFixed(1);
  document.getElementById("card-entity").textContent        = data.entity_mismatch_score.toFixed(1);
  document.getElementById("card-emotion").textContent       = data.emotion_score;

  // Color the AI card based on score
  const aiCard = document.getElementById("card-ai");
  aiCard.style.color = data.ai_score >= 60 ? "#ff3b3b"
                     : data.ai_score >= 40 ? "#ffd60a"
                     : "#00e676";

  // Color entity card
  const entityCard = document.getElementById("card-entity");
  entityCard.style.color = data.entity_mismatch_score >= 60 ? "#ff3b3b"
                          : data.entity_mismatch_score >= 30 ? "#ffd60a"
                          : "#00e676";

  // ── Triggered words chips ─────────────────────────────────
  const triggeredSection = document.getElementById("triggered-section");
  const triggeredContainer = document.getElementById("triggered-words");
  triggeredContainer.innerHTML = "";

  if (data.triggered_words && data.triggered_words.length > 0) {
    triggeredSection.style.display = "block";
    data.triggered_words.forEach(word => {
      const chip = document.createElement("span");
      chip.className = "chip chip-red";
      chip.textContent = word;
      triggeredContainer.appendChild(chip);
    });
  } else {
    triggeredSection.style.display = "none";
  }

  // ── Emotion words chips ───────────────────────────────────
  const emotionSection = document.getElementById("emotion-section");
  const emotionContainer = document.getElementById("emotion-words");
  emotionContainer.innerHTML = "";

  if (data.emotion_words && data.emotion_words.length > 0) {
    emotionSection.style.display = "block";
    data.emotion_words.forEach(word => {
      const chip = document.createElement("span");
      chip.className = "chip chip-orange";
      chip.textContent = word;
      emotionContainer.appendChild(chip);
    });
  } else {
    emotionSection.style.display = "none";
  }

  // ── Missing entities chips ────────────────────────────────
  const entitySection = document.getElementById("entity-section");
  const entityContainer = document.getElementById("missing-entities");
  entityContainer.innerHTML = "";

  if (data.missing_entities && data.missing_entities.length > 0) {
    entitySection.style.display = "block";
    data.missing_entities.forEach(entity => {
      const chip = document.createElement("span");
      chip.className = "chip chip-yellow";
      chip.textContent = entity;
      entityContainer.appendChild(chip);
    });
  } else {
    entitySection.style.display = "none";
  }

  // ── Why flagged explanation bullets ──────────────────────
  const explanationSection = document.getElementById("explanation-section");
  const explanationList = document.getElementById("explanation-list");
  explanationList.innerHTML = "";

  if (data.explanation && data.explanation.length > 0) {
    explanationSection.style.display = "block";
    data.explanation.forEach(reason => {
      const li = document.createElement("li");
      li.textContent = reason;
      explanationList.appendChild(li);
    });
  } else {
    explanationSection.style.display = "none";
  }

  // ── Meta tags ─────────────────────────────────────────────
  const tagT = document.getElementById("tag-transcript");
  tagT.textContent = data.transcript_used ? "✓ Transcript used" : "✗ No transcript";
  tagT.className   = `tag ${data.transcript_used ? "yes" : "no"}`;

  const tagL = document.getElementById("tag-length");
  tagL.textContent = data.transcript_used
    ? `${(data.transcript_length / 1000).toFixed(1)}k chars`
    : "Fallback scoring";
  tagL.className = "tag";

  showState("result");
}

// ── Main analyse function ─────────────────────────────────
async function analyse() {
  btn.disabled = true;
  btn.textContent = "Analysing…";

  let tabUrl = "";
  try {
    const [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
    tabUrl = tab.url || "";
  } catch (e) {
    showState("error");
    document.getElementById("err-msg").textContent = "Could not read tab URL.";
    resetBtn();
    return;
  }

  if (!tabUrl.includes("youtube.com/watch")) {
    showState("idle");
    resetBtn();
    return;
  }

  document.getElementById("loading-title").textContent = "";
  showState("loading");

  try {
    const response = await fetch(API_URL, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url: tabUrl }),
    });

    if (!response.ok) throw new Error(`API returned ${response.status}`);

    const data = await response.json();
    if (data.error) throw new Error(data.error);

    renderResult(data);

  } catch (err) {
    showState("error");
    let msg = err.message || "Unknown error";
    if (msg.includes("fetch") || msg.includes("Failed")) {
      msg = "Cannot reach API at localhost:8000. Is your FastAPI server running?";
    }
    document.getElementById("err-msg").textContent = msg;
  }

  resetBtn();
}

function resetBtn() {
  btn.disabled = false;
  btn.textContent = "🔍 Analyse This Video";
}

// ── On popup open: check if we're on a YT video ───────────
chrome.tabs.query({ active: true, currentWindow: true }, ([tab]) => {
  const url = tab?.url || "";
  if (!url.includes("youtube.com/watch")) showState("idle");
});

btn.addEventListener("click", analyse);