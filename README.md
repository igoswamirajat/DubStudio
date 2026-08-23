# DubStudio

Local-first AI dubbing studio. Upload a video, get a dubbed MP4 plus SRT — speech translated and cloned per speaker, original music/SFX kept.

**Phase 0 is in this commit:** API + job state machine + upload UI. Pipeline stages are named correctly but stubbed. No Whisper / Demucs / TTS yet.

## Run Phase 0

```bash
python3.11 -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make api
```

In another terminal:

```bash
cd web && npm install && npm run dev
```

Open http://127.0.0.1:5173 — drop any file, watch the 12 stages walk to `completed`.

`GET http://127.0.0.1:8080/api/v1/health`

## Spec

Read `docs/spec/` first. Default TTS is **Chatterbox** behind `VoiceEngine`.

## Stack

Python 3.11 · FastAPI · SQLite · React/Vite · FFmpeg · WhisperX · Demucs · Ollama · Chatterbox
