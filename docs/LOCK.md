# LOCKED DECISIONS — DubStudio
**Do not re-litigate mid-build. Change only via explicit new lock.**

Last locked: 2026-08-24

---

## 1. Product Identity

| Decision | Lock |
|---|---|
| Product name | **DubStudio** |
| Type | Local-first AI video dubbing studio |
| Primary use | Indian context (Hindi/Hinglish) + cross-lingual (KO/JA → EN etc.) |
| Scope v1 | Short clips first (30s–2 min), then 5–10 min, then long-form |
| Philosophy | Beautiful simple UI + pluggable engines. User never fights models. |

---

## 2. Primary Voice Engine

| Decision | Lock |
|---|---|
| **Primary TTS + Voice Cloning** | **OmniVoice** (k2-fsa/OmniVoice) |
| Why | 646 languages, zero-shot cloning (3–25s), voice design, fast, open |
| Fallback (CI / no GPU) | `DummyEngine` (duration-correct sine tones) |
| Optional later engines | Veena (Hindi specialist), Chatterbox (expressive EN + 23L) |
| Interface | `VoiceEngine` abstract class (already exists) |
| Selection | Per-job config: `voice_engine: "omnivoice" \| "veena" \| "chatterbox" \| "dummy"` |

**OmniVoice is the base.** We do not reimplement cloning. We only control it.

---

## 3. BGM / SFX Preservation (Critical)

| Decision | Lock |
|---|---|
| Strategy | **Always preserve original music + sound effects** |
| Tool | **Demucs** (`htdemucs_ft` two-stem: vocals + no_vocals) |
| Phase 1 | Fallback allowed (copy full mix → vocals + bed) so CI stays green |
| Phase 3 | Real Demucs mandatory for production quality |
| Final mix | Dubbed dialogue **over** original bed (music + SFX) with light ducking (~10 dB under speech) |
| Never | Strip or re-generate music/SFX unless user explicitly asks |

**Final outcome must still feel like the original video — only dialogue language changes.**

---

## 4. Full Runtime Pipeline (locked order)

```
1. Upload video
2. Extract audio          → FFmpeg (48 kHz PCM)
3. Source separation      → Demucs → vocals.wav + bed.wav (music+SFX)
4. Transcribe + diarize   → WhisperX (word-level + speakers)
5. Segment build          → speaker-aware dialogue segments
6. Translate              → target language (Ollama / demo map / better later)
7. Voice enroll           → short ref clip per speaker from vocals
8. Synthesize             → OmniVoice (clone or chosen/designed voice)
9. Time-correct           → atempo stretch + optional line rewrite
10. Mix                   → dialogue over bed + ducking
11. Export                → remux MP4 (video copy) + SRT
```

---

## 5. User Flow (UI must deliver this)

1. **Upload video**
2. Auto-detect:
   - How many verbal speakers
   - Ignore pure BGM / SFX regions
3. **Speaker Cards** appear:
   - Short original clip
   - Option A: **Clone this voice**
   - Option B: **Choose / Design new voice** (OmniVoice voice design)
4. Pick target language + (optional) engine
5. One-click **Dub**
6. Progress + timeline
7. Download: `output.mp4` + `output.srt` (+ optional multi-track)

UI must stay **beautiful and simple**. Complexity lives in the backend.

---

## 6. Tech Stack (locked)

| Layer | Choice |
|---|---|
| Backend | FastAPI + SQLite + filesystem checkpoints |
| Frontend | React + Vite |
| Media | FFmpeg only |
| ASR | WhisperX (mock fallback) |
| Separation | Demucs (copy fallback Phase 1) |
| TTS | OmniVoice primary (Dummy fallback) |
| Translate | Ollama if available, else demo map |
| Jobs | JobStateMachine + `data/jobs/{id}/` |
| Tests | pytest after every module; e2e fixture video required |

---

## 7. MVP Phases (locked)

| Phase | Goal | Engine / Notes |
|---|---|---|
| **0** | Skeleton (done) | — |
| **1** | 30s–1 min clip → full pipeline + tests green | Dummy + OmniVoice adapter skeleton |
| **2** | Real OmniVoice + speaker consistency + voice choice | OmniVoice primary |
| **3** | Real Demucs BGM/SFX preserve + proper mix | Demucs required |
| **4** | Translation quality + Hindi/Hinglish path | Optional Veena for pure Hindi |
| **5** | Duration correction + line rewrite loop | — |
| **6** | 5–10 min videos | — |
| **7** | Long-form + checkpointing + queue | — |
| **8** | Polished UI + timeline editor | — |

---

## 8. Non-negotiables

- Local-first (no cloud required for core path)
- BGM + SFX always preserved in final output
- Engines pluggable (never hard-code one model forever)
- Tests green after every build slice
- Short clips first, long-form later
- Beautiful simple UI over feature bloat

---

## 9. Agent rule

After every module:
1. Update code
2. `make test` (or `pytest -q`) must pass
3. At least one FFmpeg / audio integration case when media touched
4. Commit / push only when green

---

**This document is the source of truth.**  
If anything conflicts with older notes, this LOCK wins.
