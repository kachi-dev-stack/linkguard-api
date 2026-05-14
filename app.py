from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import numpy as np
import pandas as pd
from scipy.sparse import hstack
import joblib
import shap
import scipy.sparse as sp
from ollama import Client
import os

app = Flask(__name__)
CORS(app)

# -----------------------------
# Lazy load models - only when needed
# -----------------------------
model = None
vectorizer = None
scaler = None
shap_explainer = None


def get_model():
    global model
    if model is None:
        model = joblib.load("model_small.pkl")
    return model

def get_vectorizer():
    global vectorizer
    if vectorizer is None:
        vectorizer = joblib.load("vectorizer_small.pkl")
    return vectorizer

def get_scaler():
    global scaler
    if scaler is None:
        scaler = joblib.load("scaler.pkl")
    return scaler

def get_shap_explainer():
    global shap_explainer
    if shap_explainer is None:
        shap_explainer = shap.TreeExplainer(get_model())
    return shap_explainer

# -----------------------------
# OLLAMA API key
# -----------------------------
OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
if not OLLAMA_API_KEY:
    print("⚠️  WARNING: OLLAMA_API_KEY not set. AI insights will use fallback mode.")

client = Client(
    host="https://api.ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
)

# -----------------------------
# Feature columns
# -----------------------------
feature_cols = [
    "url_length", "dot_count", "digit_count", "has_https",
    "has_suspicious_words", "subdomain_count",
    "has_suspicious_tld", "has_ip_address",
    "has_at_symbol", "hyphen_count", "has_long_domain",
    "is_known_legitimate",
]

# -----------------------------
# Known legitimate domain whitelist
# -----------------------------
KNOWN_LEGITIMATE_DOMAINS = {
    "google.com", "youtube.com", "facebook.com", "amazon.com",
    "wikipedia.org", "twitter.com", "instagram.com", "linkedin.com",
    "microsoft.com", "apple.com", "netflix.com", "github.com",
    "stackoverflow.com", "reddit.com", "spotify.com", "chatgpt.com",
    "openai.com", "tailwindcss.com", "tradingview.com", "anthropic.com",
    "whatsapp.com", "tiktok.com", "zoom.us", "dropbox.com", "slack.com",
    "notion.so", "figma.com", "vercel.com", "netlify.com", "heroku.com",
    "cloudflare.com", "mouau.edu.ng",
}

def get_root_domain(url: str) -> str:
    domain = url.split("//")[-1].split("/")[0].lower()
    parts = domain.split(".")
    if len(parts) >= 2:
        return f"{parts[-2]}.{parts[-1]}"
    return domain

def is_known_legitimate(url: str) -> bool:
    return get_root_domain(url) in KNOWN_LEGITIMATE_DOMAINS

# -----------------------------
# Extract features (single URL)
# -----------------------------
def extract_features_single(url):
    domain = url.split("//")[-1].split("/")[0]
    features = {
        "url_length": len(url),
        "dot_count": url.count("."),
        "digit_count": sum(c.isdigit() for c in url),
        "has_https": 1 if url.startswith("https") else 0,
        "has_suspicious_words": 1 if any(word in url for word in [
            "login", "verify", "update", "secure", "account",
            "bank", "confirm", "signin", "billing", "alert"
        ]) else 0,
        "subdomain_count": max(0, domain.count(".") - 1),
        "has_suspicious_tld": 1 if any(tld in url for tld in [
            ".ru", ".xyz", ".click", ".info", ".top"
        ]) else 0,
        "has_ip_address": 1 if re.search(r"\d+\.\d+\.\d+\.\d+", url) else 0,
        "has_at_symbol": 1 if "@" in url else 0,
        "hyphen_count": url.count("-"),
        "has_long_domain": 1 if len(domain) > 25 else 0,
        "is_known_legitimate": int(get_root_domain(url) in KNOWN_LEGITIMATE_DOMAINS),
    }
    df = pd.DataFrame([features])
    return df[feature_cols].values

# -----------------------------
# Core analysis logic
# -----------------------------
def run_analysis(url):
    vec = get_vectorizer()
    mdl = get_model()
    sc = get_scaler()

    X_tfidf = vec.transform([url])
    X_manual_raw = extract_features_single(url)
    X_manual = sc.transform(X_manual_raw)

    X_final = hstack([X_tfidf, X_manual])

    if is_known_legitimate(url):
        pred = 0
        probs = np.array([0.97, 0.03])
        result = "Legitimate"
        confidence = 97.0
        whitelisted = True
    else:
        pred = mdl.predict(X_final)[0]
        probs = mdl.predict_proba(X_final)[0]
        result = "Phishing" if pred == 1 else "Legitimate"
        confidence = probs[pred] * 100
        whitelisted = False

    return X_final, X_tfidf, X_manual, pred, probs, result, confidence, whitelisted

# -----------------------------
# Real SHAP word factors — full densify, identical to analyze.py
# -----------------------------
def get_top_tfidf_words(url, vectorizer, shap_values_for_url, num_words=10):
    """
    Extract which specific words from the URL influenced the prediction.
    shap_val > 0  →  pushes toward phishing
    shap_val < 0  →  pushes toward legitimate
    """
    feature_names = vectorizer.get_feature_names_out()

    # Non-zero indices = words actually present in this URL
    url_vector = vectorizer.transform([url])
    nonzero_indices = url_vector.nonzero()[1]

    word_factors = []
    for idx in nonzero_indices:
        word = feature_names[idx]
        if word.strip().isdigit():
            continue
        shap_val = float(shap_values_for_url[idx])
        direction = "phishing" if shap_val > 0 else "legitimate"
        word_factors.append({
            "word": word,
            "impact": round(shap_val, 8),
            "direction": direction
        })

    word_factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return word_factors[:num_words]

# -----------------------------
# Generate AI Insight
# -----------------------------
def generate_ai_insight(url, prediction, confidence, tfidf_ratio, manual_ratio,
                         top_word_factors, top_structural):
    risk_words = [w["word"] for w in top_word_factors if w["direction"] == "phishing"][:5]
    safe_words = [w["word"] for w in top_word_factors if w["direction"] == "legitimate"][:5]
    bad_signals = [s["label"] for s in top_structural if not s["good"]]
    good_signals = [s["label"] for s in top_structural if s["good"]]

    if not OLLAMA_API_KEY:
        return generate_fallback_insight(prediction, confidence, risk_words, bad_signals)

    prompt = f"""
    You are a cybersecurity expert. Explain this URL analysis result in 2-3 clear, simple sentences.
    URL: {url}
    Prediction: {prediction}
    Confidence: {confidence}
    Risk patterns detected: {', '.join(risk_words) if risk_words else 'None'}
    Safe patterns detected: {', '.join(safe_words) if safe_words else 'None'}
    Structural issues: {', '.join(bad_signals) if bad_signals else 'None'}
    Safe indicators: {', '.join(good_signals) if good_signals else 'None'}
    Model reliance: {tfidf_ratio}% text patterns, {manual_ratio}% structural features.
    Explain why this URL was classified as {prediction}. Be direct and helpful. Do NOT mention technical terms like 'SHAP', 'TF-IDF', 'features', or 'n-grams'.
    """

    try:
        response = client.chat(
            model="qwen3-coder:480b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"⚠️ Ollama error: {e}")
        return generate_fallback_insight(prediction, confidence, risk_words, bad_signals)

# -----------------------------
# Fallback Insight
# -----------------------------
def generate_fallback_insight(prediction, confidence, risk_words, bad_signals):
    if "Phishing" in prediction:
        reasons = []
        if bad_signals:
            reasons.append(f"it has {', '.join(bad_signals[:3]).lower()}")
        if risk_words:
            reasons.append(f"it contains suspicious patterns like '{', '.join(risk_words[:3])}'")
        if reasons:
            return f"This URL is likely phishing because {' and '.join(reasons)}. Exercise caution with this link."
        else:
            return f"This URL shows phishing characteristics with {confidence} confidence. Avoid clicking this link."
    else:
        if bad_signals:
            return f"This URL appears legitimate with {confidence} confidence, though it has some minor issues. Use normal caution."
        else:
            return f"This URL appears safe with {confidence} confidence. No significant threats were detected."

# -----------------------------
# Build structural signals
# -----------------------------
def build_structural_signals(X_manual_raw):
    manual_values = X_manual_raw[0]
    top_structural = []
    rules = {
        "url_length": lambda x: ("URL Length", f"{int(x)} chars", x < 54),
        "has_https": lambda x: ("HTTPS Protocol", "Secure (HTTPS)" if x == 1 else "Insecure (HTTP)", x == 1),
        "has_suspicious_words": lambda x: ("Suspicious Keywords", "Found" if x == 1 else "None Detected", x == 0),
        "has_ip_address": lambda x: ("IP Address in URL", "Detected" if x == 1 else "Not Found", x == 0),
        "has_suspicious_tld": lambda x: ("Domain Extension", "Suspicious" if x == 1 else "Safe", x == 0),
    }
    for name, value in zip(feature_cols, manual_values):
        if name in rules:
            label, display_value, good = rules[name](value)
            top_structural.append({"feature": name, "label": label, "value": display_value, "good": bool(good)})
    return top_structural

# -----------------------------
# Calculate contribution percentages
# -----------------------------
def calculate_contributions(mdl, num_tfidf_features):
    if hasattr(mdl, 'feature_importances_'):
        importances = mdl.feature_importances_
        tfidf_importance = np.sum(importances[:num_tfidf_features])
        structural_importance = np.sum(importances[num_tfidf_features:])
        total_importance = tfidf_importance + structural_importance
        if total_importance > 0:
            tfidf_ratio = (tfidf_importance / total_importance) * 100
            manual_ratio = (structural_importance / total_importance) * 100
        else:
            tfidf_ratio, manual_ratio = 50, 50
    else:
        tfidf_ratio, manual_ratio = 50, 50
    return tfidf_ratio, manual_ratio

# =============================
# ROUTES
# =============================

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "LinkGuardAI"}), 200

# 1. Quick Scan (Home page - fast, no SHAP)
@app.route("/scan", methods=["POST"])
def quick_scan():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    url = data["url"]
    try:
        _, _, _, _, _, result, confidence, whitelisted = run_analysis(url)
        return jsonify({
            "url": url,
            "prediction": result,
            "confidence": f"{confidence:.0f}%",
            "whitelisted": whitelisted
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 2. Full Analysis — real SHAP, full densify, exactly like analyze.py
@app.route("/analysis", methods=["POST"])
def analysis():
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400
    url = data["url"]
    try:
        X_final, X_tfidf, X_manual, pred, probs, result, confidence, whitelisted = run_analysis(url)

        vec = get_vectorizer()
        mdl = get_model()
        num_tfidf_features = X_tfidf.shape[1]

        tfidf_ratio, manual_ratio = calculate_contributions(mdl, num_tfidf_features)

        # Full densify — identical to analyze.py
        X_dense = X_final.toarray()
        explainer = get_shap_explainer()
        shap_values = explainer.shap_values(X_dense)

        # For binary classification, shap_values is a list of two arrays
        if isinstance(shap_values, list):
            shap_values_for_pred = shap_values[pred][0]
        else:
            shap_values_for_pred = shap_values[0]

        shap_values_for_pred = np.array(shap_values_for_pred).flatten()

        # Slice TF-IDF SHAP values only
        shap_tfidf = shap_values_for_pred[:num_tfidf_features]

        # Get top word factors using real SHAP
        top_word_factors = get_top_tfidf_words(url, vec, shap_tfidf, num_words=10)

        # Raw (unscaled) values for human-readable structural display
        X_manual_raw = extract_features_single(url)
        top_structural = build_structural_signals(X_manual_raw)

        return jsonify({
            "url": url,
            "prediction": result,
            "confidence": f"{confidence:.0f}%",
            "whitelisted": whitelisted,
            "contribution": {
                "text_patterns": f"{tfidf_ratio:.0f}%",
                "structural_features": f"{manual_ratio:.0f}%"
            },
            "profiling": {
                "top_word_factors": top_word_factors,
                "structural_signals": top_structural
            }
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 3. AI Insight only (loads after everything else)
@app.route("/insight", methods=["POST"])
def insight():
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing data"}), 400

    risk_words = [w["word"] for w in data.get("top_word_factors", []) if w.get("direction") == "phishing"][:5]
    bad_signals = [s["label"] for s in data.get("structural_signals", []) if not s.get("good")]

    try:
        insight_text = generate_ai_insight(
            data.get("url"),
            data.get("prediction"),
            data.get("confidence"),
            float(str(data.get("text_patterns", "0")).rstrip("%")),
            float(str(data.get("structural_features", "0")).rstrip("%")),
            data.get("top_word_factors", []),
            data.get("structural_signals", [])
        )
        return jsonify({"insight": insight_text})
    except Exception as e:
        return jsonify({
            "insight": generate_fallback_insight(
                data.get("prediction", ""),
                data.get("confidence", ""),
                risk_words,
                bad_signals
            )
        })

# =============================
# START SERVER
# =============================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))