import requests
import json
import re

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "mistral"

ALLOWED_INTENTS = {"Complaint", "Request", "Query", "Feedback", "Other"}
ALLOWED_PRIORITIES = {"Low", "Medium", "High"}
ALLOWED_SENTIMENTS = {"Positive", "Neutral", "Negative"}


def build_prompt(subject: str, body: str) -> str:
    return f"""
Classify the email below.

Respond with ONLY this JSON format and NOTHING else:
{{"intent":"Complaint|Request|Query|Feedback|Other",
  "priority":"Low|Medium|High",
  "sentiment":"Positive|Neutral|Negative"}}

Subject: {subject}
Body: {body}
"""


def classify_email(subject: str, body: str) -> dict:
    prompt = build_prompt(subject, body)

    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL_NAME,
            "prompt": prompt,
            "stream": False
        },
        timeout=180

    )

    # Fail fast if Ollama fails
    response.raise_for_status()

    raw_output = response.json().get("response", "")

    # Extract JSON safely
    match = re.search(r"\{[\s\S]*?\}", raw_output)
    if not match:
        raise ValueError("LLM response did not contain valid JSON")

    try:
        result = json.loads(match.group())
    except json.JSONDecodeError:
        raise ValueError("Failed to parse JSON from LLM response")

    # Validate required keys
    for key in ["intent", "priority", "sentiment"]:
        if key not in result:
            raise ValueError(f"Missing key in LLM response: {key}")

    # Validate allowed values
    if result["intent"] not in ALLOWED_INTENTS:
        raise ValueError(f"Invalid intent value: {result['intent']}")

    if result["priority"] not in ALLOWED_PRIORITIES:
        raise ValueError(f"Invalid priority value: {result['priority']}")

    if result["sentiment"] not in ALLOWED_SENTIMENTS:
        raise ValueError(f"Invalid sentiment value: {result['sentiment']}")

    return {
        "intent": result["intent"],
        "priority": result["priority"],
        "sentiment": result["sentiment"]
    }
