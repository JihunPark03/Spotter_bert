# 🧠 Spotter — ML based AI Ad Detection Extension

Spotter is a full-stack project that detects whether selected text contains promotional or advertisement-like content.
It combines a Chrome Extension UI, a FastAPI backend, and a dedicated ML inference server powered by ModernBERT-large.

The system is designed to be **fast and modular**, with feedback collection separated from model inference.

## Language Scope

Spotter currently works for **English product reviews only**. The training data and ModernBERT classifier are built around English original-vs-synthetic review detection, so Korean or other-language text is outside the supported scope.

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
├── backend_server/      # FastAPI backend
│   ├── services/
│   ├── ml_client.py
│   └── main.py
│
├── ml_server/           # Model inference server
│   ├── inference.py
│   └── requirements.txt
│
├── manifest.json        # Chrome extension manifest
├── popup.html/js        # Chrome extension popup
├── contentScript.js     # Chrome extension content script
├── training/            # ModernBERT-large DoRA fine-tuning
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
git clone git@github.com:JihunPark03/Spotter_bert.git
cd Spotter_bert
```

---

## 2. Start API Server

```
cd backend_server
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
uvicorn main:app --port 8000
```

---

## 3. Start ML Server

```
cd ml_server
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

# 🏋️ Training in Background

Install training dependencies from the repository root:

```bash
cd /home/jihun/Spotter_bert
python3 -m venv .venv
.venv/bin/pip install -r training/requirements.txt
```

Edit `training/config.yaml` to adjust DoRA or training settings:

```yaml
dora:
  rank: 16
  alpha: 32
```

Run training in the background:

```bash
nohup .venv/bin/python training/train_modernbert.py \
  --config training/config.yaml \
  > training/modernbert-large.log 2>&1 &
```

Check progress:

```bash
tail -f training/modernbert-large.log
```

Check the running process:

```bash
pgrep -af "training/train_modernbert.py"
```

Stop the background run:

```bash
pkill -f "training/train_modernbert.py"
```

To use a different config file:

```bash
nohup .venv/bin/python training/train_modernbert.py \
  --config path/to/config.yaml \
  > training/modernbert-large.log 2>&1 &
```

Training progress is reported to TensorBoard by default. Set `report_to: wandb` in `training/config.yaml` if you want W&B logging.

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
