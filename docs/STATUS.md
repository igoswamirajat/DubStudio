# DubStudio — All Phases Status (2026-08-25)

| Phase | Status | Deliverable |
|---|---|---|
| 0 Skeleton | DONE | FastAPI + SQLite + folder tree |
| 1 Basic pipeline | DONE | FFmpeg → ASR stub → translate → TTS → mix → export |
| 2 OmniVoice + speakers | DONE | Real engine adapter, speaker API, voice_mode |
| 2.5 Speaker cards UI | DONE | Clone / design / auto + ref preview |
| **3 Demucs bed** | **DONE** | bed always preserved, ducking mix, Demucs path |
| **4 Translation quality** | **DONE** | Duration budget, multi-lang glossary, quality score |
| **5 Timing rewrite** | **DONE** | atempo clamp + needs_rewrite + rewrite loop |
| **6 Long video** | **DONE** | max_duration_s=600, chunk settings, soft warnings |
| **7 Checkpoint/resume** | **DONE** | 12 stage markers, resume transitions |
| **8 Timeline editor** | **DONE** | Segment patch + resynth API + React timeline |

## Tests
```
30 passed
SMOKE OK — 12 checkpoints, bed.wav present, Hindi translation, output.mp4
```

## Run
```bash
pip install -e ".[dev]"
make test
make api
cd web && npm i && npm run dev
```

## Production extras
```bash
pip install omnivoice demucs
export DUBSTUDIO_TTS_ENGINE=omnivoice
```

## API map
- POST /api/v1/jobs
- GET  /api/v1/jobs/{id}
- GET/PATCH /api/v1/jobs/{id}/speakers/{sid}
- GET/PATCH /api/v1/jobs/{id}/segments/{seg}
- POST /api/v1/jobs/{id}/segments/{seg}/resynth
- POST /api/v1/jobs/{id}/resume
- GET  /api/v1/jobs/{id}/download?artifact=mp4|srt
