# 04 — Phases, acceptance, build order

**Source of truth:** `docs/LOCK.md` + `docs/FINAL_PLAN.md` + `docs/PHASE2.md`

---

## Phase 0 — Skeleton (DONE)

## Phase 1 — 30s basic dub (DONE)
- Full pipeline + Dummy + smoke_job + tests green

## Phase 2 — Real OmniVoice + speakers (CODE DONE)
- Real OmniVoiceEngine.generate (clone / design / auto)
- SynthRequest: voice_mode, ref_text, instruct
- Speaker cards API: GET/PATCH speakers + ref audio
- Enroll + synthesis honor voice_mode
- create_job accepts tts_engine
- Docs: `docs/PHASE2.md`

**On your GPU machine:**
```bash
pip install omnivoice
export DUBSTUDIO_TTS_ENGINE=omnivoice
python scripts/smoke_job.py --lang hi --engine omnivoice
```

**Phase 2.5 (next):** React speaker cards UI

## Phase 3 — Bed preservation (Demucs)
- Real Demucs htdemucs_ft
- Final video still has original music/SFX

## Phase 4 — Translation + Hindi path (optional Veena)

## Phase 5 — Timing rewrite loop

## Phase 6–8 — Longer videos, queue, polish UI
