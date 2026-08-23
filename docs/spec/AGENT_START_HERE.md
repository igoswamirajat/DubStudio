# AGENT START HERE

You are building **DubStudio**, a local-first AI video dubbing studio.

You were not at the product meeting. Everything you need is this folder.

## What the product does
Upload a video → extract audio → transcribe with word timestamps → label speakers → split dialogue from music/SFX → translate each line → clone each speaker's voice in the target language → fit each line into the original time window → mix speech back over the original bed → remux MP4 + SRT.

## What you must not do
- Do not call Meta Voicebox. Use the `VoiceEngine` interface; default implementation is **Chatterbox**.
- Do not build lip-sync, cloud auth, or 2-hour movie infrastructure in Phase 1.
- Do not treat the film as one TTS request. Work **per segment**.
- Do not send user media to the internet on the default path.

## Read next
`00-PRODUCT.md` → `01-ARCHITECTURE.md` → `02-DATA-CONTRACTS.md` → `03-API.md` → `04-PHASES.md` → `05-AGENT-PROMPTS.md` starting with Prompt A.

## Locked stack
Python 3.11, FastAPI, SQLite, React+Vite, FFmpeg, WhisperX, Demucs (htdemucs_ft), Ollama translator, Chatterbox TTS.

## Phase 1 bar
A 30-second clip becomes a dubbed MP4 on localhost with progress in the UI.
