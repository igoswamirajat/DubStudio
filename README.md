# DubStudio

Local-first AI dubbing studio. Upload a video → extract → transcribe → translate → synthesize → fit timing → mix over bed → remux MP4 + SRT.

**Phase 1 locked:** full pipeline runs end-to-end. WhisperX / Chatterbox / Demucs plug in when installed; CI uses mock ASR + DummyEngine so tests pass without a GPU.

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make test
make api
```

UI: `cd web && npm install && npm run dev` → http://127.0.0.1:5173

## Tests

```bash
make test
```

9 tests: state machine, ffmpeg extract/remux, segments, timing, full e2e fixture → MP4+SRT.

## Spec

`docs/LOCK.md` and `docs/spec/`.
