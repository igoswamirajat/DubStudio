# DubStudio — one image: API + UI (Dummy TTS ready out of the box)
# Optional GPU / OmniVoice: see docker-compose.gpu.yml

FROM node:22-bookworm-slim AS web
WORKDIR /web
COPY web/package.json web/package-lock.json* ./
RUN npm install --prefer-offline --no-audit --no-fund
COPY web/ ./
RUN npm run build

FROM python:3.12-slim-bookworm

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    DUBSTUDIO_TTS_ENGINE=dummy \
    DUBSTUDIO_TRANSLATOR=demo \
    DUBSTUDIO_SKIP_SEPARATION=1 \
    DUBSTUDIO_HOST=0.0.0.0 \
    DUBSTUDIO_PORT=8080 \
    DATA_DIR=/data

RUN apt-get update && apt-get install -y --no-install-recommends \
      ffmpeg \
      libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY pyproject.toml README.md ./
COPY dubstudio ./dubstudio
COPY scripts ./scripts
RUN pip install --no-cache-dir -e .

COPY --from=web /web/dist ./web/dist

RUN mkdir -p /data/jobs
VOLUME ["/data"]
EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
  CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8080/api/v1/health')" || exit 1

CMD ["uvicorn", "dubstudio.main:app", "--host", "0.0.0.0", "--port", "8080"]
