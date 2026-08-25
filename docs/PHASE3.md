# Phase 3 — BGM / SFX preservation (Demucs)

## Goal
Final dubbed video **still has original music and sound effects**.
Only dialogue is replaced.

## How
1. `run_separation(skip=False)` runs Demucs `htdemucs_ft` two-stem
2. `vocals.wav` → ASR / enroll / synth
3. `bed.wav` (no_vocals = music + SFX) → mix under new dialogue with ducking
4. Export remuxes video + mixed audio

## Install
```bash
pip install demucs
# UI default: skip_separation=false
# or DUBSTUDIO_SKIP_SEPARATION=0
```

## Acceptance
- Demucs available → mode `demucs_htdemucs_ft`
- Missing Demucs → safe fallback copy (CI green)
- Final mix clearly retains bed

## Status
Code path ready in `dubstudio/pipeline/separation.py` + mix ducking already in mixing.py.
Install demucs on GPU machine for production quality.
