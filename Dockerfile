# Inference-only container for the emotion detection model.
# jiitravi account is being used, but for testing with group project, this will be switched to admin account.
# Build:  docker build --build-arg HF_MODEL_NAME=jiitravi/emotion-distilbert -t jiitravi/mlops-emotion:latest .
# Test:   docker run --rm -e INPUT_TEXT="I feel amazing today" jiitravi/mlops-emotion:latest
# Push:   docker push jiitravi/mlops-emotion:latest

FROM python:3.11-slim

# Model repo — override at build time to switch models without changing code
ARG HF_MODEL_NAME=jiitravi/emotion-distilbert
ENV HF_MODEL_NAME=${HF_MODEL_NAME}

WORKDIR /app

# Install CPU-only torch first (keeps image lean — no CUDA drivers needed for inference)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

# Install remaining inference deps
COPY requirements-inference.txt .
RUN pip install --no-cache-dir -r requirements-inference.txt

# Copy only the inference script — no training code in the container
COPY src/inference.py .

# HF_TOKEN  — pass at runtime via -e HF_TOKEN=<token> (only needed for private models)
# INPUT_TEXT — pass at runtime via -e INPUT_TEXT="input"
CMD ["python", "inference.py"]
