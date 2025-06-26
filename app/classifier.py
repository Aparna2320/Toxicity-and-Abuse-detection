import joblib
import json
import os

# Path to config file
config_path = os.path.join(os.path.dirname(__file__), 'config.json')

# Load the model
model = joblib.load(os.path.join('models', 'model.pkl'))

# Load initial config
with open(config_path) as f:
    config = json.load(f)

# ✅ ADD THIS FUNCTION:
def reload_config():
    global config
    with open(config_path) as f:
        config = json.load(f)

# Main prediction function
def predict_toxicity(text: str) -> dict:
    score = model.predict_proba([text])[0][1]

    if score >= config["toxicity_threshold"]:
        label = "toxic"
        action = "blocked"
    elif score >= config["flag_threshold"]:
        label = "flagged"
        action = "flagged"
    else:
        label = "safe"
        action = "approved"

    reasons = ["toxic"] if label != "safe" else []

    return {
        "toxicity_score": round(score, 2),
        "label": label,
        "action": action,
        "reasons": reasons,
        "threshold": config["toxicity_threshold"]
    }

