# DubStudio — Docker (simple install)

## What you get

| Mode | Command | Voice | Music/SFX | Needs |
|---|---|---|---|---|
| **Simple (recommended first)** | `docker compose up --build` | Dummy (auto) | Full mix copy | Docker only |
| **GPU real voice** | compose + `docker-compose.gpu.yml` | OmniVoice | Demucs | NVIDIA + toolkit |

**One URL after start:** http://localhost:8080  
UI + API same origin. Upload → wait → download. Pipeline automated.

---

## Windows (Docker Desktop)

1. Install [Docker Desktop](https://www.docker.com/products/docker-desktop/)
2. In PowerShell:

```powershell
cd "D:\Coding Projects\DubStudio\DubStudio"
.\scripts\docker-up.ps1
```

Or:

```powershell
docker compose up --build
```

3. Browser → **http://localhost:8080**

Stop:

```powershell
docker compose down
```

---

## OmniVoice (real cloning) — optional

**Not** baked into the default image (too large + needs GPU).

### Host install (Windows)

```powershell
.\.venv\Scripts\Activate.ps1
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install omnivoice demucs
$env:DUBSTUDIO_TTS_ENGINE="omnivoice"
$env:DUBSTUDIO_SKIP_SEPARATION="0"
uvicorn dubstudio.main:app --host 127.0.0.1 --port 8080
```

First run downloads models (time + disk). After that, UI is the same simple flow.

---

## Honest limits

- Default Docker = ready to test full pipeline, demo voices, bed preserved as copy.
- OmniVoice cannot be zero-time install — models are GBs.
- CPU OmniVoice is slow; Dummy is default for instant success.
