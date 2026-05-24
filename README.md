# LinkGuard AI — Phishing URL Detection API

> A machine learning API that detects and profiles phishing URLs using Random Forest classification, SHAP explainability, and AI-generated insights.

Built as a Final Year Project · Computer Science · Michael Okpara University of Agriculture, Umudike (MOUAU)

---

## Overview

LinkGuard AI analyses any URL and returns:

- **Detection** — whether the URL is phishing or legitimate, with a confidence score
- **Profiling** — which character patterns and structural features drove the prediction
- **AI Insight** — a plain-English explanation of why the URL was flagged

The system combines two feature types:

| Feature Type | Description | Count |
|---|---|---|
| Lexical patterns | Character-level TF-IDF n-grams (2–4 chars) extracted from the URL string | 10,000 |
| Structural features | Hand-crafted URL properties (length, HTTPS, suspicious keywords, TLD, etc.) | 13 |

---

## API Endpoints

### `GET /`
Health check.

**Response**
```json
{
  "status": "healthy",
  "service": "LinkGuard AI"
}
```

---

### `POST /scan`
Quick scan — fast prediction with no explainability overhead. Used by the home page.

**Request**
```json
{
  "url": "http://paypal-login-security-update.com"
}
```

**Response**
```json
{
  "url": "http://paypal-login-security-update.com",
  "prediction": "Phishing",
  "confidence": "95%"
}
```

---

### `POST /analysis`
Full profiling — returns SHAP-based word factors, structural signals, and feature contribution breakdown. No AI insight (that loads separately via `/insight`).

**Request**
```json
{
  "url": "http://paypal-login-security-update.com"
}
```

**Response**
```json
{
  "url": "http://paypal-login-security-update.com",
  "prediction": "Phishing",
  "confidence": "95%",
  "contribution": {
    "text_patterns": "72%",
    "structural_features": "28%"
  },
  "profiling": {
    "top_word_factors": [
      { "word": "login", "impact": 0.00312, "direction": "phishing" },
      { "word": "pay",   "impact": 0.00289, "direction": "phishing" }
    ],
    "structural_signals": [
      { "feature": "has_https",            "label": "HTTPS Protocol",      "value": "Insecure (HTTP)", "good": false },
      { "feature": "has_suspicious_words", "label": "Suspicious Keywords", "value": "Found",           "good": false },
      { "feature": "has_suspicious_tld",   "label": "Domain Extension",    "value": "Safe",            "good": true  }
    ]
  }
}
```

---

### `POST /insight`
AI-generated plain-English explanation. Called after `/analysis` has rendered so the UI feels fast.

**Request**
```json
{
  "url": "http://paypal-login-security-update.com",
  "prediction": "Phishing",
  "confidence": "95%",
  "text_patterns": "72%",
  "structural_features": "28%",
  "top_word_factors": [...],
  "structural_signals": [...]
}
```

**Response**
```json
{
  "insight": "This URL is likely phishing because it uses an insecure HTTP connection and contains suspicious keywords like 'login' and 'paypal' in a misleading domain name. Legitimate PayPal pages always use https://www.paypal.com."
}
```

---

## Model

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Training data | 6 Kaggle datasets merged → 1,332,195 URLs after cleaning |
| Train / test split | 80% / 20% stratified |
| TF-IDF features | Character n-grams (2–4), max 10,000 features |
| Structural features | 13 hand-crafted URL properties |
| Total features | 10,013 per URL |
| `n_estimators` | 100 |
| `max_depth` | 20 |
| `class_weight` | balanced |
| Confidence threshold | Tuned via F1 optimisation (0.56) |
| Accuracy | ~95% |
| Explainability | SHAP TreeExplainer |

### Confusion Matrix

```
                 Predicted Legitimate   Predicted Phishing
True Legitimate       165,075                6,419
True Phishing           7,475               87,470
```

---

## Project Structure

```
linkguardai-api/
│
├── app.py                  # Flask application — all routes and logic
├── requirements.txt        # Python dependencies
│
├── model_small.pkl         # Trained Random Forest model    (Git LFS)
├── vectorizer_small.pkl    # Fitted TF-IDF vectorizer       (Git LFS)
├── scaler.pkl              # Fitted StandardScaler          (Git LFS)
├── threshold.pkl           # Tuned probability threshold    (Git LFS)
│
├── train.ipynb             # Full training notebook
│   ├── Section 1 — Imports & Dataset
│   ├── Section 2 — Feature Engineering & Training
│   ├── Section 3 — Evaluation & Confusion Matrix
│   ├── Section 4 — Single URL Test
│   ├── Section 5 — Batch URL Test
│   └── Section 6 — Save Artifacts
│
└── dataset/
    └── final_feature_dataset.csv   # Merged, cleaned training data
```

---

## Local Setup

**Requirements:** Python 3.10+

```bash
# 1. Clone the repository
git clone https://github.com/your-username/linkguardai-api.git
cd linkguardai-api

# 2. Install dependencies
pip install -r requirements.txt

# 3. Set your Ollama API key (optional — fallback insight used if not set)
export OLLAMA_API_KEY=your_key_here

# 4. Run the server
python app.py
```

The API will be available at `http://localhost:5000`.

---

## Dependencies

```
flask
flask-cors
numpy
pandas
scikit-learn
scipy
joblib
shap
ollama
```

Install all with:
```bash
pip install flask flask-cors numpy pandas scikit-learn scipy joblib shap ollama
```

---

## Deployment

The API is deployed on **Render** (free tier).

Model artifacts (`.pkl` files) are stored via **Git LFS** and pulled automatically during deployment.

**Environment variable required on Render:**
```
OLLAMA_API_KEY = your_ollama_api_key
```

If the key is not set, the `/insight` endpoint falls back to a rule-based text response automatically — the other two endpoints are unaffected.

---

## Design Decisions

**Why Random Forest?**
Interpretable, fast at inference, and works well with mixed sparse (TF-IDF) and dense (structural) feature matrices via `scipy.sparse.hstack`.

**Why character-level TF-IDF?**
Phishing URLs manipulate domain names at the character level — `paypa1.com`, `arnazon.com`. Character n-grams catch these patterns without needing word tokenisation.

**Why SHAP?**
SHAP (SHapley Additive exPlanations) provides theoretically grounded, per-prediction feature attribution. It answers not just *what* the model predicted but *why* — which is the core of the profiling requirement.

**Why a tuned threshold?**
The default 0.5 cutoff treats both error types equally. A tuned threshold (found via F1 optimisation on the test set) better balances the cost of missing real phishing against the cost of false alarms on legitimate URLs.

**Why `normalise_to_www()`?**
The training corpus predominantly contained `https://www.domain.com` formatted URLs. Without normalisation, `https://mouau.edu.ng` and `https://www.mouau.edu.ng` produce different feature vectors and potentially different predictions despite being the same resource. Prepending `www.` before inference ensures consistent results — the original URL is always returned to the user unchanged.

---

## Known Limitations

- URLs containing authentication terms within the domain name itself (e.g. `login.microsoftonline.com`) may be misclassified because those character n-grams carry strong phishing signal in the training data
- Bare domains from underrepresented regions and ccSLDs (e.g. `.edu.ng`, `.gov.ng`) may score lower legitimate confidence due to training data geography bias
- The model does not fetch or render page content — classification is based on URL structure only

---

## Frontend

The React frontend for this API is in a separate repository:
[linkguardai-frontend](https://github.com/kachi-dev-stack/linkguard-ai)

---

## Author

**Samuel** · Final Year Computer Science · MOUAU  
Supervised by DR. SAMUEL UGBOAJA

---

## License

This project was built for academic purposes as a final year project submission.
