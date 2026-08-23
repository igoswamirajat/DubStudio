# Locked decisions (do not re-litigate mid-build)

| Area | Lock |
|---|---|
| Product | Local-first AI dubbing studio, short clips first |
| ASR | WhisperX when installed; else `MockASR` for CI |
| TTS | Chatterbox when installed; else `DummyEngine` for CI |
| Separation | Demucs when installed; else copy full mix (Phase 1 OK) |
| Translate | Ollama if up; else demo map |
| Media | FFmpeg only |
| Jobs | SQLite + filesystem checkpoints |
| API | FastAPI `/api/v1` on 127.0.0.1 |
| Phase 1 bar | 30s clip → dubbed MP4 + SRT |

After every module land: run `pytest -q`.
