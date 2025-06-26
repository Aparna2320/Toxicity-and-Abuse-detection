import requests

API_URL = "http://127.0.0.1:8000/analyze-text"

test_cases = [
    {
        "input": {
            "user_id": "u1",
            "post_id": "p1",
            "text": "Thank you for the feedback. It was helpful."
        },
        "expected_label": "safe"
    },
    {
        "input": {
            "user_id": "u2",
            "post_id": "p2",
            "text": "You're stupid and worthless"
        },
        "expected_label": "toxic"
    },
    {
        "input": {
            "user_id": "u3",
            "post_id": "p3",
            "text": "What the hell is wrong with you"
        },
        "expected_label": "flagged"
    }
]

for test in test_cases:
    response = requests.post(API_URL, json=test["input"])
    result = response.json()
    print(f"Text: {test['input']['text']}")
    print(f"Expected: {test['expected_label']} | Predicted: {result['label']}")
    print("---")
