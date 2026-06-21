# 🧠 Spotter — ML based AI Ad Detection Extension

Spotter is a full-stack project that detects whether selected text contains promotional or advertisement-like content.
It combines a Chrome Extension UI, a FastAPI backend, and a dedicated ML inference server powered by ModernBERT-large.

The system is designed to be **fast and modular**, with feedback collection separated from model inference.

## Warning : it only works with Korean review/text

# Examples
For example, Spotter was tested on the following dining review website (https://www.diningcode.com/profile.php?rid=hAebbrQ1gHyi)

Popup view after pressing 'Analyze' button:
![Popup analysis](assets/screenshots/Screenshot%202026-02-26%20at%2012.12.03.png)

Feedback homepage:
![Progress and summary](assets/screenshots/Screenshot%202026-02-26%20at%2012.12.42.png)

Feedback page (User can rate the text):
![Progress and summary](assets/screenshots/Screenshot%202026-02-26%20at%2012.12.50.png)

---

# 🚀 Features

* Detect advertisement probability from highlighted text
* Lightweight API server with caching support
* Dedicated ML server for efficient inference
* ModernBERT-large ad detector

---

# 🧩 Architecture Overview

```
Chrome Extension
        ↓
API Server (FastAPI)
        ↓
ML Server (ModernBERT Inference)
        ↓
Hugging Face Model
```

### Why separate API and ML servers?

* API server stays lightweight and responsive
* ModernBERT loads only once inside the ML server
* Feedback collection can run independently from model inference

---

# 📁 Project Structure

```
Spotter/
│
├── api-server/          # FastAPI backend
│   ├── routes/
│   ├── services/
│   ├── ml_client.py
│   └── main.py
│
├── ml-server/           # Model inference server
│   ├── inference.py
│   └── requirements.txt
│
├── extension/           # Chrome extension UI
│
└── docker-compose.yml
```

---

# ⚙️ Tech Stack

**Backend**

* FastAPI
* PostgreSQL (feedback storage)
* Redis or in-memory cache

**Machine Learning**

* PyTorch
* Transformers
* ModernBERT-large

**Frontend**

* Chrome Extension (Vanilla JS)

---

# 🧪 How It Works

## 1️⃣ User selects text

The extension sends:

```
POST /detect-ad
```

---

## 2️⃣ API Server

The API server:

* Creates a cache key
* Checks Redis or local cache
* Calls ML server if result is not cached

```
prob = request_inference(text)
```

---

## 3️⃣ ML Server

The ML server:

* Loads `answerdotai/ModernBERT-large`
* Tokenizes the text
* Runs sequence-classification inference

Output:

```
prob_ad = softmax(logits)[ad_label]
```

---

# ⚡ Setup Guide

## 1. Clone Repository

```
git clone https://github.com/yourname/spotter.git
cd spotter
```

---

## 2. Start API Server

```
cd api-server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

---

## 3. Start ML Server

```
cd ml-server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8001
```

---

## 4. Redis (Optional)

```
brew install redis
redis-server
```

Environment variables:

```
REDIS_HOST=localhost
REDIS_PORT=6379
```

If Redis is unavailable, Spotter falls back to an in-memory cache.

---

# 🧩 API Endpoint

## Detect Advertisement

```
POST /detect-ad
```

Request:

```
{
  "text": "example review text"
}
```

Response:

```
{
  "prob_ad": 82.3,
  "is_ad": true,
  "cached": false
}
```

---

# 🔄 Model Configuration

The ML server defaults to:

```
AD_DETECTOR_MODEL=answerdotai/ModernBERT-large
```

Set `AD_DETECTOR_MAX_LENGTH` to change token truncation length. The default is `512`.

---

# 🌍 Deployment Notes

Recommended setup:

* API Server → GCP VM
* ML Server → Same VM or separate instance
* Redis → Local Redis or Memorystore

---

# 👨‍💻 Author

Jihun Park
Computer Science & Communication Engineering
Waseda University

---

# ⭐ Motivation

Spotter explores how real user interaction and feedback can be integrated into a practical AI pipeline, combining lightweight backend engineering with an evolving machine learning model.
