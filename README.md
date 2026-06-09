# ml-ops-group-project — Emotion Detection Pipeline

## Overview

To design and crate End-to-end MLOps pipeline for **class emotion detection** using `distilbert-base-uncased` fine-tuned on the [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion) dataset. Built as part of the MLOps Group Project coursework.

The pipeline is orchestrated with:
- Dataset loading, cleaning, and preprocessing on Kaggle GPU
- Tokenization using Hugging Face `DistilBertTokenizerFast`
- Fine-tuning with the Hugging Face Trainer API on Kaggle GPU
- Two experiment versions with different hyperparameters
- Experiment tracking with Weights & Biases (W&B)
- Model and tokenizer upload to Hugging Face Hub
- Dockerised inference container
- CI and inference automation via GitHub Actions

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     dair-ai/emotion dataset                      │
│                  ~20k samples · 6 emotion classes                │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│              Data Cleaning & Tokenisation (notebook)             │
│        lowercase · strip URLs/mentions · DistilBertTokenizer     │
└───────────────────────────────┬─────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                   Kaggle Notebook · GPU T4                       │
│  ┌──────────────────────────┐  ┌──────────────────────────────┐ │
│  │       Version 1          │  │         Version 2            │ │
│  │  lr=3e-5 · batch=16      │  │   lr=5e-5 · batch=32         │ │
│  │  epochs=3                │  │   epochs=3                   │ │
│  └────────────┬─────────────┘  └──────────────┬───────────────┘ │
└───────────────┼──────────────────────────────┬─┘────────────────┘
                │                              │
        ┌───────▼───────┐            ┌─────────▼──────────┐
        │ Weights&Biases│            │  Hugging Face Hub   │
        │  loss · acc   │            │  model + tokenizer  │
        │  F1 · config  │            │  publicly pullable  │
        └───────────────┘            └─────────┬───────────┘
                                               │
                                     ┌─────────▼───────────┐
                                     │    Docker image      │
                                     │  python:3.11-slim    │
                                     │  inference only      │
                                     └─────────┬────────────┘
                                               │
┌──────────────────────────────────────────────▼─────────────────┐
│                      GitHub Repository                          │
│          src/ · .github/workflows/ · Dockerfile                 │
└────────────────────────┬───────────────────┬────────────────────┘
                         │                   │
              ┌──────────▼──────┐  ┌─────────▼──────────────┐
              │   CI workflow    │  │   Inference workflow    │
              │ flake8 · develop │  │ workflow_dispatch       │
              │ push/PR to main  │  │ → src/inference.py      │
              └─────────────────┘  └────────────────────────┘
```

---

## Project Resources

| Resource | Link |
|---|---|
| GitHub Repository | https://github.com/aniliter-cloud/ml-ops-group-project |
| Kaggle Notebook  | https://www.kaggle.com/code/anilg25ait2009/mlops-group-projects/ |
| Hugging Face Model | https://huggingface.co/Maxii2tj/emotion-distilbert |
| W&B Project Dashboard | https://wandb.ai/maxi-i2tj-na/mlops-group-project |
| Docker Image | *(add Docker Hub URL here)* |

---

## Project Structure

```
.
├── .github/
│   └── workflows/
│       ├── ci.yml                  # Linting on push to develop
│       └── inference.yml           # Manual inference trigger
├── src/
│   └── inference.py                # Inference script
├── Dockerfile                      # Inference container
├── id2label.json                   # Label mapping (6 emotion classes)
├── requirements.txt
└── README.md
```

---

## Dataset

**Dataset:** [`dair-ai/emotion`](https://huggingface.co/datasets/dair-ai/emotion)  
**Task:** 6-class text classification  
**Classes:** `sadness`, `joy`, `love`, `anger`, `fear`, `surprise`  
**Size:** ~20,000 samples (train + test)

### Cleaning Steps Applied

| Step | Reason |
|---|---|
| Lowercase all text | DistilBERT-uncased expects lowercase input |
| Strip URLs | Carry no emotional signal |
| Remove `@mentions` and `#hashtags` | Noise in short social text |
| Collapse whitespace | Normalises newlines and multi-space artefacts |
| Filter empty texts | Removes any rows that became empty after cleaning |

---

## Model

**Model:** `distilbert-base-uncased`  
**Parameters:** ~66M  
**HF Model Card:** https://huggingface.co/distilbert-base-uncased

DistilBERT is a knowledge-distilled, lighter version of BERT — approximately 40% smaller and 60% faster — making it well-suited to Kaggle's free GPU tier. The uncased variant was chosen because emotion classification depends on semantic meaning rather than capitalisation patterns, reducing vocabulary size and improving token consistency. With 6 output labels loaded from `id2label.json`, the model fits comfortably within Kaggle's free GPU hours.

---

## Training Configuration — Two Experiment Versions

| Parameter | Version 1 (v1) | Version 2 (v2) |
|---|---|---|
| Model | distilbert-base-uncased | distilbert-base-uncased |
| Learning Rate | 3e-5 | 5e-5 |
| Batch Size | 16 | 32 |
| Epochs | 3 | 3 |
| Max Token Length | 512 | 512 |
| Weight Decay | 0.01 | 0.01 |
| Warmup Steps | 100 | 100 |
| Eval Strategy | epoch | epoch |
| Platform | Kaggle T4 GPU | Kaggle T4 GPU |
| Experiment Tracking | W&B | W&B |

---

## Setup Instructions

### Prerequisites

- Python 3.11+
- Docker (for inference container)
- Accounts: Kaggle, Hugging Face, Weights & Biases

### 1. Clone the Repository

```bash
git clone https://github.com/aniliter-cloud/ml-ops-group-project.git
cd ml-ops-group-project
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Environment Variables

```bash
export HF_TOKEN=<hf_token>
export WANDB_API_KEY=<wandb_key>
```

---

## Running on Kaggle (Recommended for Training)

1. Open the Kaggle notebook link above
2. Enable GPU: **Settings → Accelerator → GPU T4 x2**
3. Add the following Kaggle Secrets (**Add-ons → Secrets**):
   - `WANDB_API_KEY`
   - `HF_TOKEN`
4. Set `RUN_VERSION = "v1"` (or `"v2"`) in the config cell
5. Run all cells sequentially

The notebook will:
- Load and clean the `dair-ai/emotion` dataset
- Tokenize using `DistilBertTokenizerFast`
- Train for 3 epochs with the selected hyperparameters
- Log all metrics to W&B
- Save an `eval_report.json` artifact to W&B
- Push the best model and tokenizer to Hugging Face Hub

---

## Running Inference Locally

```bash
# Pull the Docker image
docker pull <your-dockerhub-username>/mlops-a3-inference:latest

# Run inference
docker run --rm \
  -e HF_TOKEN=$HF_TOKEN \
  -e INPUT_TEXT="I feel so happy today" \
  <your-dockerhub-username>/mlops-a3-inference:latest
```

Or run the script directly:

```bash
HF_TOKEN=$HF_TOKEN INPUT_TEXT="I feel so happy today" python src/inference.py
```

---

## GitHub Actions Workflows

### CI Workflow (`ci.yml`)
- **Triggers:** push to `develop`, pull request to `main`
- **Steps:** checkout → setup Python 3.11 → install deps → run `flake8` on `src/`

### Inference Workflow (`inference.yml`)
- **Triggers:** manual (`workflow_dispatch`) with `input_text` parameter
- **Steps:** checkout → setup Python 3.11 → install deps → run `src/inference.py`
- **Secrets required:** `HF_TOKEN` (set in GitHub → Settings → Secrets and Variables → Actions)

---

## W&B Experiment Tracking

Both runs are logged to the project [`mlops-group-project`](https://wandb.ai/maxi-i2tj-na/mlops-group-project) on W&B.

Metrics tracked per run:
- Training loss (per step)
- Validation loss (per epoch)
- Accuracy
- Weighted F1-score
- All hyperparameters (learning rate, batch size, epochs, model name, platform, version)
- `eval_report.json` uploaded as a versioned W&B artifact

---

## Technologies Used

- Python 3.11
- PyTorch
- Hugging Face Transformers & Datasets
- Scikit-learn
- Weights & Biases
- Kaggle Notebooks (GPU T4)
- Docker
- GitHub Actions

---

## Authors

| Name | Role |
|---|---|
| Anil Kumar Das | G25AIT2009 |
| Ravi Kant Pandey | G25AIT2085 |
