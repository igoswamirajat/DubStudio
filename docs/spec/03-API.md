# 03 — HTTP API (`/api/v1`)

Base URL: `http://127.0.0.1:8080/api/v1`

- GET /health
- POST /jobs (multipart file, target_language)
- GET /jobs
- GET /jobs/{job_id}
- GET /jobs/{job_id}/segments
- GET /jobs/{job_id}/events (SSE)
- GET /jobs/{job_id}/download?artifact=mp4|srt|json
- POST /jobs/{job_id}/cancel

No auth. Bind 127.0.0.1 only by default.
