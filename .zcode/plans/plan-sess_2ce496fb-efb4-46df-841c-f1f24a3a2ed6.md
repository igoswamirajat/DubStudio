# Sab kuch UI se — full control panel

Har cheez jo abhi env/`.env` ya hardcoded hai, use UI me le aata hoon. Local single-user hai isliye ek runtime **Settings** panel + expanded job form + job history + richer segment editor.

## 1. Backend: runtime settings API
- `GET /api/v1/settings` — current editable config (secrets masked: `hf_token`, `openai_api_key` sirf "set/unset" dikhega).
- `PATCH /api/v1/settings` — in-process `settings` update **aur** `.env` me persist (restart ke baad bhi rahe).
- Editable fields: `translator` (ollama/openai/demo), `ollama_host/model`, `openai_base_url/api_key/model`, `tts_engine`, `asr_engine`, `whisper_model`, `enable_diarization`, `hf_token`, `max_speakers`, `skip_separation`, `low_vram`.
- **Translation API** (jo aapne poocha) yahin se on hoga — koi bhi OpenAI-compatible provider (OpenAI/Groq/DeepSeek/OpenRouter). `translation.py` me `openai` branch add karunga.

## 2. Frontend: Settings panel (naya)
- Collapsible "Settings ⚙️" section: translator dropdown → openai select karne par base_url/key/model fields; TTS engine; ASR/whisper model; diarization toggle + HF token; separation + low-VRAM toggles.
- "Save" → PATCH; live `/health` badge refresh dikhaega kya active hai.

## 3. Frontend: job form expand
- **Source language** (Auto-detect + list), **skip separation** toggle per-job — dono backend already accept karta hai, UI abhi nahi bhejta.

## 4. Frontend: job history
- `GET /jobs` se recent jobs ki list; click karke purana job dobara khol/preview/download.

## 5. Segment editor richer
- Har segment par **speaker reassign** dropdown + **start/end timing** edit (backend `PATCH` already supports), text edit + re-synth ke saath.

## Verify
- Naye settings endpoints ke HTTP tests + openai translator ka mocked-httpx unit test.
- `npm run build` clean + `pytest -q` green.

Secrets `.env` me plain save honge (local single-user, expected). Chahein to key ko UI me masked rakhunga.