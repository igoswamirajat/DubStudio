# DubStudio

**Local-first AI video dubbing studio**

Upload a video → keep original music & SFX → replace dialogue → download MP4 + SRT.

## Fastest path (Docker)

```bash
docker compose up --build
```

Open **http://localhost:8080**

Windows:

```powershell
.\scripts\docker-up.ps1
```

Details: [docs/DOCKER.md](docs/DOCKER.md)

| Out of the box | Extra |
|---|---|
| Full pipeline + UI + export | Real OmniVoice |
| Dummy TTS + demo translate | GPU + `pip install omnivoice` |
| Bed preserved (copy) | Demucs |

## Local Windows (no Docker)

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -e ".[dev]"
uvicorn dubstudio.main:app --host 127.0.0.1 --port 8080 --reload
```

## Real voice (optional)

```powershell
pip install omnivoice demucs
$env:DUBSTUDIO_TTS_ENGINE="omnivoice"
```

## Full local pipeline (GPU)

Turns every stage real: Demucs separation, faster-whisper ASR, pyannote
diarization, Ollama translation, OmniVoice TTS.

```powershell
# 1. Install all model deps into the venv
pip install -e ".[full]"

# 2. Translation — install Ollama and pull a model
#    https://ollama.com/download
ollama pull qwen2.5:7b

# 3. Diarization — needs a HuggingFace token AND accepting the model terms:
#    https://huggingface.co/pyannote/speaker-diarization-3.1
#    then set it (also read as HF_TOKEN):
$env:DUBSTUDIO_HF_TOKEN="hf_xxx"
```

Copy `.env.example` to `.env` to configure engines/models. Defaults already
select the real engines; the first run downloads the Whisper / OmniVoice /
Demucs / pyannote models (several GB). Check `/api/v1/health` to confirm what is
active (GPU, ASR, diarization, TTS, translator).

`GET /api/v1/health` reports each subsystem, e.g.:

```json
{"ok": true, "ffmpeg": true, "gpu": true, "asr": "faster-whisper",
 "diarization": "installed", "tts": "omnivoice", "translator": "ollama"}
```

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```
