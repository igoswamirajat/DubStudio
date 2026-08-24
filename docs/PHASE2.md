# Phase 2 — Real OmniVoice + Speaker voice choice

**Status:** implemented in code (requires GPU + `pip install omnivoice` for real voices)

## What landed

### 1. Real OmniVoiceEngine
- `from omnivoice import OmniVoice`
- Modes:
  - **clone** — `ref_audio` (+ optional `ref_text`)
  - **design** — `instruct` string e.g. `"female, young adult, hindi accent"`
  - **auto** — model picks a voice
- Output 24 kHz mono → pipeline resamples to 48 kHz in mix
- Device auto: CUDA → MPS → CPU
- Missing package → factory falls back to DummyEngine (CI still green)

### 2. SynthRequest extended
```
voice_mode, ref_text, instruct
```

### 3. Speaker cards API
```
GET  /api/v1/jobs/{id}/speakers
PATCH /api/v1/jobs/{id}/speakers/{speaker_id}
GET  /api/v1/jobs/{id}/speakers/{speaker_id}/ref   # ref wav
```

Patch body:
```json
{
  "voice_mode": "clone" | "design" | "fixed" | "auto",
  "design_prompt": "female, young adult, hindi accent",
  "label": "Hero",
  "ref_text": "exact words in the ref clip",
  "voice_id": "S00"
}
```

### 4. Enroll + synthesis
- Enroll cuts 4–8s ref per speaker, stores `voice_mode=clone` by default
- Synthesis reads speaker_map and passes mode / instruct / ref to engine

## Install OmniVoice (your machine)

```bash
# 1) PyTorch for your CUDA (example cu128)
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128

# 2) OmniVoice
pip install omnivoice

# 3) run with real engine
export DUBSTUDIO_TTS_ENGINE=omnivoice
make api
python scripts/smoke_job.py --lang hi --engine omnivoice
```

Without GPU / package, keep:
```bash
export DUBSTUDIO_TTS_ENGINE=dummy
```

## UI contract (next polish)
1. After job reaches `enrolling_voices` / post-enroll, fetch speakers
2. Show one card per speaker with audio preview (`.../ref`)
3. Toggle: Clone original | Design new voice
4. If Design → text field for instruct
5. Continue → synthesize uses the choices

## Acceptance
- [x] Engine adapter real generate path
- [x] Factory still falls back when package missing
- [x] Speaker list + patch API
- [x] Synthesis honors voice_mode
- [ ] Manual GPU smoke with real OmniVoice (on your machine)
- [ ] UI speaker cards wired (Phase 2.5 / UI slice)

## Next
- Phase 2.5: React speaker cards UI
- Phase 3: real Demucs BGM preserve quality
