# 00 — Product Brief (zero-context)

Build **DubStudio**: a local-first AI dubbing studio that takes a video file, extracts speech, labels speakers, translates dialogue, generates matching voices, keeps original music/SFX, and exports a dubbed MP4.

## Hard product rules
1. Local-first. Default path never uploads media.
2. Segment-centric. Unit of work is a dialogue segment.
3. Preserve non-speech. Only dialogue is replaced.
4. Timing is first-class. Fit or rewrite/stretch.
5. Speaker consistency. Same speaker_id → same voice_id.
6. Short clips first. Phase 1 is a 30-second, 1–2 speaker clip.
7. No lip-sync in MVP.
8. No invented facts in translation.

## Phase 1 success
30s clip → dubbed MP4, same video frames, target-language speech aligned within ±150 ms, bed still audible, artifacts written, UI shows real stages.
