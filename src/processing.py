"""
Tasks 2 & 3 - Data preparation and model loading.

Task 2: Cleans dair-ai/emotion and saves id2label.json
Task 3: Loads DistilBERT tokenizer + model using id2label.json

Run:
  python src/load_model.py --prepare   # clean data + save id2label.json
  python src/load_model.py             # just verify model loads correctly
"""
import argparse
import json
import re
from collections import Counter
from pathlib import Path

from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast


BASE_MODEL = "distilbert-base-uncased"
ID2LABEL_PATH = Path(__file__).parent.parent / "id2label.json"


# ── Task 2: Data Preparation ─────────────────────────────────────────────────

def clean_text(example):
    text = example["text"].lower()
    text = re.sub(r"http\S+|www\S+", "", text)  # remove URLs
    text = re.sub(r"@\w+", "", text)            # remove @mentions
    text = re.sub(r"#\w+", "", text)            # remove #hashtags
    text = re.sub(r"\s+", " ", text).strip()    # collapse whitespace
    return {"text": text}


def prepare_data():
    from datasets import load_dataset

    print("Loading dair-ai/emotion...")
    raw = load_dataset("dair-ai/emotion")

    label_names = raw["train"].features["label"].names
    id2label = {i: name for i, name in enumerate(label_names)}
    print(f"Labels : {id2label}")
    print(f"Class distribution (train):\n{Counter(raw['train']['label'])}")

    cleaned = raw.map(clean_text)
    cleaned = cleaned.filter(lambda x: len(x["text"]) > 0)
    print(f"Train: {len(cleaned['train'])}  Val: {len(cleaned['validation'])}  Test: {len(cleaned['test'])}")

    # Save id2label.json — committed to repo
    with open(ID2LABEL_PATH, "w") as f:
        json.dump({str(k): v for k, v in id2label.items()}, f, indent=2)
    print(f"Saved: {ID2LABEL_PATH}")

    # Save cleaned CSVs locally — NOT committed (see .gitignore)
    data_dir = Path(__file__).parent.parent / "data"
    data_dir.mkdir(exist_ok=True)
    for split in ["train", "validation", "test"]:
        out = data_dir / f"{split}.csv"
        cleaned[split].to_csv(str(out), index=False)
        print(f"Saved: {out}")


# ── Task 3: Model Loading ─────────────────────────────────────────────────────

def load_id2label():
    with open(ID2LABEL_PATH) as f:
        raw = json.load(f)
    return {int(k): v for k, v in raw.items()}


def load_tokenizer():
    return DistilBertTokenizerFast.from_pretrained(BASE_MODEL)


def load_model(id2label):
    label2id = {v: k for k, v in id2label.items()}
    return DistilBertForSequenceClassification.from_pretrained(
        BASE_MODEL,
        num_labels=len(id2label),
        id2label=id2label,
        label2id=label2id,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--prepare", action="store_true", help="Run data preparation (Task 2)")
    args = parser.parse_args()

    if args.prepare:
        prepare_data()

    id2label = load_id2label()
    tokenizer = load_tokenizer()
    model = load_model(id2label)

    total_params = sum(p.numel() for p in model.parameters()) / 1e6
    print(f"\nModel : {BASE_MODEL}  |  {total_params:.1f}M parameters")
    print(f"Labels: {id2label}")
