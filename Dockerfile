FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/opt/hf \
    MODEL_REPO=adichaudhary/deepsentinel \
    MODEL_FILE=deepsentinel_effb4.pt

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /srv
COPY requirements-api.txt .
RUN pip install --no-cache-dir -r requirements-api.txt

# Bake model weights into the image so containers start without a network fetch.
RUN python -c "import os; from huggingface_hub import hf_hub_download; \
print(hf_hub_download(os.environ['MODEL_REPO'], os.environ['MODEL_FILE']))"

COPY app ./app

RUN useradd --create-home appuser && chown -R appuser /srv /opt/hf
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s --retries=3 \
    CMD curl -fsS http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
