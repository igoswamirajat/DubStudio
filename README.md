# DubStudio

**Local-first AI video dubbing studio**

Upload a video → keep the original music & SFX → replace only the spoken dialogue with cloned (or new) voices in any language → download polished dubbed MP4 + SRT.

## Locked decisions

- **Primary TTS + Voice Cloning:** OmniVoice
- **Fallback (CI / no GPU):** DummyEngine
- **BGM + SFX:** Always preserved (Demucs → bed track + light ducking)
- **ASR:** WhisperX (mock fallback)
- **Media:** FFmpeg
- **UI:** Clean, simple, beautiful — speaker cards + one-click dub

See:
- `docs/LOCK.md` — hard decisions
- `docs/PRODUCT.md` — product brief
- `docs/FINAL_PLAN.md` — full compiled plan + phases

## Quick start

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
make test          # must pass
make api           # http://127.0.0.1:8080/api/v1/health
```

UI (other terminal):

```bash
cd web && npm install && npm run dev
```

## Tests (run after every build)

```bash
make test
```

## Engines

| Stage | Primary | Fallback |
|---|---|---|
| TTS + Cloning | **OmniVoice** | DummyEngine |
| Optional later | Veena (Hindi), Chatterbox | — |
| ASR | WhisperX | Mock |
| Separation | Demucs | Copy full mix (Phase 1) |
| Translate | Ollama | Demo map |

```bash
# Force engines
export DUBSTUDIO_TTS_ENGINE=omnivoice   # or dummy
export DUBSTUDIO_TRANSLATOR=demo
```

## Pipeline (locked)

1. Upload
2. FFmpeg extract
3. Demucs → vocals + bed (music/SFX)
4. WhisperX transcription + diarization
5. Translate
6. Voice enroll / clone (OmniVoice)
7. Synthesize
8. Time-correct
9. Mix dialogue over bed
10. Export MP4 + SRT

## Stack

Python 3.11+ · FastAPI · SQLite · React/Vite · FFmpeg · WhisperX · Demucs · OmniVoice · Ollama

## Agent rule

After every module: run tests. Only green builds are accepted.
