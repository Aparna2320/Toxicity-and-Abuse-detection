# Toxicity & Abuse Detection – Offline AI Microservice

This microservice detects toxic, abusive, or inappropriate language in user-generated text using a locally trained NLP model. It is fully offline, built using FastAPI, and deployable via Docker.



##  Prerequisites

- Python 3.10+ (for local runs)
- Docker Desktop (for containerized runs)
 

##  Installation

### 1. Clone the Project

git clone <https://github.com/Aparna2320/Toxicity-and-Abuse-detection>
cd Toxicity-detection


### 2. Build the Docker Image


docker build -t toxicity-detector .


### 4. Run the Container


docker run -p 8000:8000 toxicity-detector


Access API at: [http://localhost:8000/docs]



##  Model Details

* Framework: Scikit-learn
* Pipeline: TF-IDF Vectorizer + Logistic Regression
* Model file: `app/model.pkl` (offline serialized)
* Categories Detected: toxic, insult, harassment, obscene, identity attack, threat

### Threshold Logic:

| Score Range | Label   | Action   |
| ----------- | ------- | -------- |
| ≥ 0.75      | toxic   | blocked  |
| 0.50 – 0.74 | flagged | flagged  |
| < 0.50      | safe    | approved |

### Configuration File (`config.json`):

json
{
  "toxicity_threshold": 0.75,
  "flag_threshold": 0.5,
  "enabled_categories": ["toxic", "insult", "harassment"]
}


---

##  Testing

### 1. API Test Script

Run:

uvicorn app.main:app --reload

python test_samples/test_api.py


This script:

* Sends 3 predefined inputs to `/analyze-text`
* Prints expected vs actual labels
* Validates model logic

Example output:


Text: You're stupid and worthless
Expected: toxic | Predicted: toxic
---
Text: Thank you for the feedback. It was helpful.
Expected: safe | Predicted: safe


### 2. Manual Test via Swagger UI

Visit [http://localhost:8000/docs], try:


{
  "user_id": "test01",
  "post_id": "p001",
  "text": "You're pathetic and dumb"
}


Response:


{
  "user_id": "test01",
  "post_id": "p001",
  "toxicity_score": 0.91,
  "label": "toxic",
  "action": "blocked",
  "reasons": ["insult", "harassment"],
  "threshold": 0.75
}




##  Files

* `app/main.py`: FastAPI endpoints
* `app/classifier.py`: Preprocessing + inference
* `app/model.pkl`: Offline trained model
* `app/config.json`: Threshold and label settings
* `test_samples/test_api.py`: API test script
* `Dockerfile`: Offline deployment setup

---

##  Summary

| Requirement       | Status                                  |
| ----------------- | --------------------------------------- |
| Offline support   |  Fully Offline           |
| FastAPI endpoints |  /analyze-text, /health, /version      |
| Dockerized        |  Ready for containerized deployment    |
| Configurable      |  Supports thresholds and label filters |
| Model             |  TF-IDF + LogisticRegression           |




