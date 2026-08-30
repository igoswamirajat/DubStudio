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

## Tests

```bash
pip install -e ".[dev]" && pytest -q
```
