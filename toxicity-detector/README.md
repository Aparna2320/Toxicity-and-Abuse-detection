
# 🧪 Toxicity Detector API

A machine learning-powered API that detects toxic, flagged, or safe text comments using logistic regression and a TF-IDF vectorizer. Built with **FastAPI**, configurable via `config.json`, and fully containerized with **Docker**.

---

## 🚀 Features

- ✅ Predicts toxicity from text input
- ✅ Labels: `safe`, `flagged`, `toxic`
- ✅ Returns reasons, confidence score, and action
- ✅ Live API docs at `/docs` (Swagger UI)
- ✅ Fully Dockerized for deployment

---

## 🗂️ Project Structure

```

toxicity-detector/
├── app/
│   ├── main.py              # FastAPI app
│   ├── classifier.py        # Prediction logic & model loader
│   ├── config.json          # Thresholds for scoring
├── models/
│   └── model.pkl            # Trained ML model
├── test\_samples/
│   └── test\_api.py          # API test script
├── requirements.txt         # Dependencies
├── Dockerfile               # Docker config
└── README.md                # Project description

````

---

## ⚙️ Requirements

Install Python dependencies locally:

```bash
pip install -r requirements.txt
````

---
 
## 🐳 Prerequisites

- Python 3.10+ (for local runs)
- Docker Desktop (for containerized runs)
 
 
## 🧪 Run Locally (No Docker)

```bash
uvicorn app.main:app --reload
```

Then open:

```
http://127.0.0.1:8000/docs
```

Try the `/analyze-text` endpoint with a payload like:

```json
{
  "user_id": "u1",
  "post_id": "p1",
  "text": "You are a fool and a loser"
}
```

---

## 🧪 Test Locally

Run the test script:

bash
python test_samples/test_api.py
```

---

## 🐳 Run with Docker

### 1. Build the image:

bash
docker build -t toxicity-detector .
```

### 2. Run the container:

bash
docker run -p 8000:8000 toxicity-detector
```

Then open in your browser:

```
http://127.0.0.1:8000/docs
```

---

## ⚙️ Configuration

Edit thresholds in `app/config.json`:

```json
{
  "flag_threshold": 0.5,
  "toxicity_threshold": 0.75
}
```

Reload them without restarting:

```http
POST /reload-config
```

---

## 📊 Response Example

```json
{
  "user_id": "u1",
  "post_id": "p1",
  "toxicity_score": 0.83,
  "label": "toxic",
  "action": "blocked",
  "reasons": ["toxic"],
  "threshold": 0.75
}
```

---

## 📞 Endpoints Summary

| Method | Endpoint         | Description         |
| ------ | ---------------- | ------------------- |
| GET    | `/health`        | Health check        |
| GET    | `/version`       | Returns version     |
| POST   | `/analyze-text`  | Detects toxicity    |
| POST   | `/reload-config` | Reloads config live |

---

