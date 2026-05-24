from flask import Flask, request, jsonify
from flask_cors import CORS
import re
import os
import numpy as np
import pandas as pd
import scipy.sparse as sp
from scipy.sparse import hstack
import joblib
import shap
from ollama import Client

app = Flask(__name__)
CORS(app)


# ─────────────────────────────────────────
# Model loading  (lazy — loads on first request)
# ─────────────────────────────────────────

_model          = None
_vectorizer     = None
_scaler         = None
_threshold      = None
_shap_explainer = None

def get_model():
    global _model
    if _model is None:
        _model = joblib.load("model.pkl")
    return _model

def get_vectorizer():
    global _vectorizer
    if _vectorizer is None:
        _vectorizer = joblib.load("vectorizer.pkl")
    return _vectorizer

def get_scaler():
    global _scaler
    if _scaler is None:
        _scaler = joblib.load("scaler.pkl")
    return _scaler

def get_threshold():
    global _threshold
    if _threshold is None:
        try:
            _threshold = float(joblib.load("threshold.pkl"))
        except Exception:
            _threshold = 0.5
    return _threshold

def get_shap_explainer():
    global _shap_explainer
    if _shap_explainer is None:
        _shap_explainer = shap.TreeExplainer(get_model())
    return _shap_explainer


# ─────────────────────────────────────────
# Ollama AI client
# ─────────────────────────────────────────

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY")
if not OLLAMA_API_KEY:
    print("WARNING: OLLAMA_API_KEY not set — AI insight will use fallback text.")

ollama_client = Client(
    host="https://api.ollama.com",
    headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"} if OLLAMA_API_KEY else {}
)


# ─────────────────────────────────────────
# Feature columns — must match train2.ipynb exactly
# 13 structural features, no has_www
# ─────────────────────────────────────────

FEATURE_COLS = [
    "url_length", "dot_count", "digit_count", "has_https",
    "has_suspicious_words", "subdomain_count", "has_suspicious_tld",
    "has_ip_address", "has_at_symbol", "hyphen_count", "has_long_domain",
    "is_bare_domain", "domain_name_length",
]


# ─────────────────────────────────────────
# URL normalisation
#
# The model was trained on URLs that were predominantly in
# https://www.domain.com format. Adding www. before inference
# ensures https://mouau.edu.ng and https://www.mouau.edu.ng
# both produce identical predictions.
#
# The original URL is NEVER modified in API responses —
# the user always sees exactly what they typed.
# ─────────────────────────────────────────

def normalise_to_www(url: str) -> str:
    """Add www. if not already present — for model input only."""
    if "//" in url and "//www." not in url:
        url = url.replace("//", "//www.", 1)
    return url


# ─────────────────────────────────────────
# Feature extraction — mirrors extract_features_single() in train2.ipynb
# ─────────────────────────────────────────

def extract_features(url: str) -> np.ndarray:
    """
    Extract 13 structural features from a URL.
    url should already be the www-normalised version so features
    are consistent with what the model was trained on.
    """
    hostname = url.split("//")[-1].split("/")[0]
    parts    = hostname.split(".")
    after    = url.split("//")[-1][len(hostname):]
    sld      = parts[-2] if len(parts) >= 2 else hostname

    features = {
        "url_length"          : len(url),
        "dot_count"           : url.count("."),
        "digit_count"         : sum(c.isdigit() for c in url),
        "has_https"           : int(url.startswith("https")),
        "has_suspicious_words": int(any(w in url.lower() for w in [
                                    "login", "verify", "update", "secure", "account",
                                    "bank", "confirm", "signin", "billing", "alert"])),
        "subdomain_count"     : max(0, len(parts) - 2),
        "has_suspicious_tld"  : int(parts[-1] in {
                                    "ru", "xyz", "click", "info", "top", "tk", "ml", "ga", "cf"}),
        "has_ip_address"      : int(bool(re.search(r"\d+\.\d+\.\d+\.\d+", url))),
        "has_at_symbol"       : int("@" in url),
        "hyphen_count"        : url.count("-"),
        "has_long_domain"     : int(len(hostname) > 25),
        "is_bare_domain"      : int(after in ("", "/")),
        "domain_name_length"  : len(sld),
    }

    return pd.DataFrame([features])[FEATURE_COLS].values


# ─────────────────────────────────────────
# Core prediction
# ─────────────────────────────────────────

def run_analysis(url_original: str) -> dict:
    """
    Normalise → vectorise → scale → combine → classify.

    url_original  — what the user typed, returned in all API responses
    url_for_model — www-normalised version, used for model input only
    """
    vec = get_vectorizer()
    mdl = get_model()
    sc  = get_scaler()
    thr = get_threshold()

    url_for_model = normalise_to_www(url_original)

    X_tfidf      = vec.transform([url_for_model])
    X_manual_raw = extract_features(url_for_model)
    X_manual     = sc.transform(X_manual_raw)
    X_final      = hstack([X_tfidf, X_manual])

    probs         = mdl.predict_proba(X_final)[0]
    prob_phishing = probs[1]

    if prob_phishing >= thr:
        pred       = 1
        result     = "Phishing"
        confidence = prob_phishing * 100
    else:
        pred       = 0
        result     = "Legitimate"
        confidence = probs[0] * 100

    return {
        "X_final"      : X_final,
        "X_tfidf"      : X_tfidf,
        "X_manual_raw" : X_manual_raw,
        "pred"         : pred,
        "probs"        : probs,
        "result"       : result,
        "confidence"   : confidence,
        "url_for_model": url_for_model,
    }


# ─────────────────────────────────────────
# Word factors — real SHAP on full dense matrix
#
# shap_val > 0  → n-gram pushes toward phishing
# shap_val < 0  → n-gram pushes toward legitimate
# ─────────────────────────────────────────

def get_word_factors(url_for_model: str, shap_values_for_pred: np.ndarray,
                     num_words: int = 10) -> list:
    vec           = get_vectorizer()
    feature_names = vec.get_feature_names_out()
    url_vector    = vec.transform([url_for_model])
    nonzero_cols  = url_vector.nonzero()[1]

    word_factors = []
    for idx in nonzero_cols:
        word = feature_names[idx]
        if word.strip().isdigit():
            continue
        shap_val  = float(shap_values_for_pred[idx])
        word_factors.append({
            "word"     : word,
            "impact"   : round(shap_val, 8),
            "direction": "phishing" if shap_val > 0 else "legitimate",
        })

    word_factors.sort(key=lambda x: abs(x["impact"]), reverse=True)
    return word_factors[:num_words]


# ─────────────────────────────────────────
# Structural signals
# ─────────────────────────────────────────

def get_structural_signals(X_manual_raw: np.ndarray) -> list:
    values  = X_manual_raw[0]
    signals = []

    rules = {
        "url_length"          : lambda x: ("URL Length",          f"{int(x)} chars",                        x < 54),
        "has_https"           : lambda x: ("HTTPS Protocol",      "Secure (HTTPS)" if x else "Insecure (HTTP)", bool(x)),
        "has_suspicious_words": lambda x: ("Suspicious Keywords", "Found" if x else "None Detected",         not bool(x)),
        "has_ip_address"      : lambda x: ("IP Address in URL",   "Detected" if x else "Not Found",          not bool(x)),
        "has_suspicious_tld"  : lambda x: ("Domain Extension",    "Suspicious" if x else "Safe",             not bool(x)),
    }

    for name, value in zip(FEATURE_COLS, values):
        if name in rules:
            label, display_value, good = rules[name](value)
            signals.append({
                "feature": name,
                "label"  : label,
                "value"  : display_value,
                "good"   : bool(good),
            })

    return signals


# ─────────────────────────────────────────
# Feature contribution percentages
# ─────────────────────────────────────────

def get_contributions(mdl, num_tfidf_features: int) -> tuple:
    importances    = mdl.feature_importances_
    tfidf_imp      = np.sum(importances[:num_tfidf_features])
    structural_imp = np.sum(importances[num_tfidf_features:])
    total          = tfidf_imp + structural_imp
    if total == 0:
        return 50.0, 50.0
    return (tfidf_imp / total) * 100, (structural_imp / total) * 100


# ─────────────────────────────────────────
# AI Insight
# ─────────────────────────────────────────

def generate_insight(url, result, confidence, tfidf_pct, structural_pct,
                     word_factors, structural_signals) -> str:

    risk_words  = [w["word"]  for w in word_factors       if w["direction"] == "phishing"][:5]
    safe_words  = [w["word"]  for w in word_factors       if w["direction"] == "legitimate"][:5]
    bad_signals = [s["label"] for s in structural_signals if not s["good"]]
    ok_signals  = [s["label"] for s in structural_signals if s["good"]]

    if not OLLAMA_API_KEY:
        return _fallback_insight(result, confidence, risk_words, bad_signals)

    prompt = (
        f"You are a cybersecurity expert. Explain this URL analysis in 2-3 clear, simple sentences.\n"
        f"URL: {url}\n"
        f"Prediction: {result} (confidence: {confidence:.0f}%)\n"
        f"Risk patterns: {', '.join(risk_words) or 'None'}\n"
        f"Safe patterns: {', '.join(safe_words) or 'None'}\n"
        f"Structural issues: {', '.join(bad_signals) or 'None'}\n"
        f"Safe indicators: {', '.join(ok_signals) or 'None'}\n"
        f"Model relied {tfidf_pct:.0f}% on text patterns and {structural_pct:.0f}% on structural features.\n"
        f"Be direct. Do NOT mention SHAP, TF-IDF, features, or n-grams."
    )

    try:
        response = ollama_client.chat(
            model="qwen3-coder:480b-cloud",
            messages=[{"role": "user", "content": prompt}]
        )
        return response.message.content.strip()
    except Exception as e:
        print(f"Ollama error: {e}")
        return _fallback_insight(result, confidence, risk_words, bad_signals)


def _fallback_insight(result, confidence, risk_words, bad_signals) -> str:
    if result == "Phishing":
        reasons = []
        if bad_signals:
            reasons.append(f"it has {', '.join(bad_signals[:3]).lower()}")
        if risk_words:
            reasons.append(f"it contains suspicious patterns like '{', '.join(risk_words[:3])}'")
        if reasons:
            return f"This URL is likely phishing because {' and '.join(reasons)}. Exercise caution."
        return f"This URL shows phishing characteristics with {confidence:.0f}% confidence. Do not click it."
    else:
        if bad_signals:
            return f"This URL appears legitimate ({confidence:.0f}% confidence) but has minor issues. Use normal caution."
        return f"This URL appears safe with {confidence:.0f}% confidence. No significant threats were detected."


# ─────────────────────────────────────────
# API Routes
# ─────────────────────────────────────────

@app.route("/", methods=["GET"])
def health_check():
    return jsonify({"status": "healthy", "service": "LinkGuard AI"}), 200


@app.route("/scan", methods=["POST"])
def scan():
    """Quick scan — prediction and confidence only. Used by the home page."""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    url_original = data["url"]
    try:
        p = run_analysis(url_original)
        return jsonify({
            "url"       : url_original,
            "prediction": p["result"],
            "confidence": f"{p['confidence']:.0f}%",
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/analysis", methods=["POST"])
def analysis():
    """Full profiling — real SHAP, word factors, structural signals, contributions."""
    data = request.get_json()
    if not data or "url" not in data:
        return jsonify({"error": "Missing 'url' in request body"}), 400

    url_original = data["url"]
    try:
        p   = run_analysis(url_original)
        mdl = get_model()

        tfidf_pct, structural_pct = get_contributions(mdl, p["X_tfidf"].shape[1])

        # Full SHAP densify on single row
        X_dense      = p["X_final"].toarray()
        explainer    = get_shap_explainer()
        shap_values  = explainer.shap_values(X_dense)

        if isinstance(shap_values, list):
            shap_for_pred = shap_values[p["pred"]][0]
        else:
            shap_for_pred = shap_values[0]

        shap_for_pred = np.array(shap_for_pred).flatten()
        shap_tfidf    = shap_for_pred[:p["X_tfidf"].shape[1]]

        word_factors       = get_word_factors(p["url_for_model"], shap_tfidf, num_words=10)
        structural_signals = get_structural_signals(p["X_manual_raw"])

        return jsonify({
            "url"         : url_original,
            "prediction"  : p["result"],
            "confidence"  : f"{p['confidence']:.0f}%",
            "contribution": {
                "text_patterns"      : f"{tfidf_pct:.0f}%",
                "structural_features": f"{structural_pct:.0f}%",
            },
            "profiling": {
                "top_word_factors"  : word_factors,
                "structural_signals": structural_signals,
            },
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/insight", methods=["POST"])
def insight():
    """AI explanation — called after /analysis has already rendered."""
    data = request.get_json()
    if not data:
        return jsonify({"error": "Missing request body"}), 400

    risk_words  = [w["word"]  for w in data.get("top_word_factors",  []) if w.get("direction") == "phishing"][:5]
    bad_signals = [s["label"] for s in data.get("structural_signals", []) if not s.get("good")]

    try:
        text = generate_insight(
            url                = data.get("url", ""),
            result             = data.get("prediction", ""),
            confidence         = float(str(data.get("confidence", "0")).rstrip("%")),
            tfidf_pct          = float(str(data.get("text_patterns", "0")).rstrip("%")),
            structural_pct     = float(str(data.get("structural_features", "0")).rstrip("%")),
            word_factors       = data.get("top_word_factors", []),
            structural_signals = data.get("structural_signals", []),
        )
        return jsonify({"insight": text})
    except Exception as e:
        return jsonify({
            "insight": _fallback_insight(
                data.get("prediction", ""),
                float(str(data.get("confidence", "0")).rstrip("%")),
                risk_words,
                bad_signals,
            )
        })


# ─────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        debug=True,
        host="0.0.0.0",
        port=int(os.environ.get("PORT", 5000))
    )
