"""
Emotion Classification Inference.

Environment Variables
---------------------
INPUT_TEXT    : Required. Text to classify.
HF_MODEL_NAME : Optional. Hugging Face model repo.
                Defaults to Maxii2tj/emotion-distilbert.
HF_TOKEN      : Optional. Required only for private Hugging Face models.
"""

import os
import sys

from transformers import pipeline


DEFAULT_MODEL = "Maxii2tj/emotion-distilbert"


def main() -> None:
    model_name = os.getenv("HF_MODEL_NAME", DEFAULT_MODEL)
    input_text = os.getenv("INPUT_TEXT", "").strip()
    hf_token = os.getenv("HF_TOKEN")

    if not input_text:
        print(
            "Error: INPUT_TEXT environment variable is missing or empty.",
            file=sys.stderr,
        )
        sys.exit(1)

    print(f"Model : {model_name}")
    print(f"Input : {input_text}")
    print(
        "Authentication : "
        + ("Enabled" if hf_token else "Anonymous/Public model access")
    )

    pipeline_args = {
        "task": "text-classification",
        "model": model_name,
    }

    if hf_token:
        pipeline_args["token"] = hf_token

    try:
        classifier = pipeline(**pipeline_args)
        result = classifier(input_text)[0]

        print(
            f"Predicted emotion : {result['label']} "
            f"(confidence: {result['score']:.4f})"
        )

    except Exception as exc:
        print(f"Inference failed: {exc}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()