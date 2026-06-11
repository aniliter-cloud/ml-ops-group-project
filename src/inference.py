"""
Task 6/7 - Run emotion classification inference.
Reads input from environment variables — no hardcoded values.

Environment variables:
  INPUT_TEXT      (required) Text to classify
  HF_MODEL_NAME   (optional) Hugging Face repo, default: Maxii2tj/emotion-distilbert
  HF_TOKEN        (optional) For private repos; leave unset for public models

Run locally:
  INPUT_TEXT="I feel amazing today" python src/inference.py

Docker:
  docker run --rm -e INPUT_TEXT="I feel amazing today" local/mlops-emotion:latest
"""
import os
import sys

from transformers import pipeline


HF_MODEL_NAME = os.environ.get("HF_MODEL_NAME", "Maxii2tj/emotion-distilbert")
INPUT_TEXT = os.environ.get("INPUT_TEXT", "").strip()
HF_TOKEN = os.environ.get("HF_TOKEN") or None  # None = unauthenticated (public model)

if not INPUT_TEXT:
    print("Error: INPUT_TEXT environment variable is not set or empty.", file=sys.stderr)
    sys.exit(1)

print(f"Model : {HF_MODEL_NAME}")
print(f"Input : {INPUT_TEXT}")

classifier = pipeline(
    "text-classification",
    model=HF_MODEL_NAME,
    token=HF_TOKEN,
)

result = classifier(INPUT_TEXT)[0]
print(f"Predicted emotion : {result['label']}  (confidence: {result['score']:.4f})")
