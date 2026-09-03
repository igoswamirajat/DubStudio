import { useEffect, useState } from "react";
import { getSettings, patchSettings, type AppSettings } from "../api";

const TRANSLATORS = ["ollama", "openai", "demo"];
const TTS = ["omnivoice", "chatterbox", "dummy"];
const ASR = ["faster-whisper", "mock"];
const WHISPER = ["large-v3", "large-v3-turbo", "medium", "small", "base", "tiny"];

export function SettingsPanel({ onSaved }: { onSaved?: () => void }) {
  const [open, setOpen] = useState(false);
  const [s, setS] = useState<AppSettings | null>(null);
  const [draft, setDraft] = useState<Record<string, unknown>>({});
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");
  const [saving, setSaving] = useState(false);

  useEffect(() => {
    if (open && !s) getSettings().then(setS).catch((e) => setErr(String(e)));
  }, [open, s]);

  function set(k: string, v: unknown) {
    setDraft((d) => ({ ...d, [k]: v }));
  }
  function val<K extends keyof AppSettings>(k: K): AppSettings[K] | undefined {
    return (draft[k as string] as AppSettings[K]) ?? s?.[k];
  }

  async function save() {
    setSaving(true);
    setErr("");
    setMsg("");
    try {
      const next = await patchSettings(draft);
      setS(next);
      setDraft({});
      setMsg("Saved");
      onSaved?.();
    } catch (e) {
      setErr(String(e));
    } finally {
      setSaving(false);
    }
  }

  const translator = (val("translator") as string) || "ollama";

  return (
    <section className="card">
      <button className="btn ghost settings-toggle" onClick={() => setOpen((o) => !o)}>
        ⚙️ Settings {open ? "▲" : "▼"}
      </button>
      {open && !s && <p className="muted">Loading…</p>}
      {open && s && (
        <div className="settings-body">
          <div className="grid3">
            <div>
              <label>Translator</label>
              <select value={translator} onChange={(e) => set("translator", e.target.value)}>
                {TRANSLATORS.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </div>
            {translator === "ollama" && (
              <>
                <div>
                  <label>Ollama host</label>
                  <input
                    type="text"
                    value={(val("ollama_host") as string) || ""}
                    onChange={(e) => set("ollama_host", e.target.value)}
                  />
                </div>
                <div>
                  <label>Ollama model</label>
                  <input
                    type="text"
                    value={(val("ollama_model") as string) || ""}
                    onChange={(e) => set("ollama_model", e.target.value)}
                  />
                </div>
              </>
            )}
            {translator === "openai" && (
              <>
                <div>
                  <label>API base URL</label>
                  <input
                    type="text"
                    value={(val("openai_base_url") as string) || ""}
                    onChange={(e) => set("openai_base_url", e.target.value)}
                  />
                </div>
                <div>
                  <label>API model</label>
                  <input
                    type="text"
                    value={(val("openai_model") as string) || ""}
                    onChange={(e) => set("openai_model", e.target.value)}
                  />
                </div>
                <div>
                  <label>API key {s.openai_api_key_set ? "(set)" : ""}</label>
                  <input
                    type="password"
                    placeholder={s.openai_api_key_set ? "•••••• (leave blank to keep)" : "sk-…"}
                    onChange={(e) => set("openai_api_key", e.target.value)}
                  />
                </div>
              </>
            )}
          </div>

          <div className="grid3">
            <div>
              <label>TTS engine</label>
              <select value={(val("tts_engine") as string) || "omnivoice"} onChange={(e) => set("tts_engine", e.target.value)}>
                {TTS.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label>ASR engine</label>
              <select value={(val("asr_engine") as string) || "faster-whisper"} onChange={(e) => set("asr_engine", e.target.value)}>
                {ASR.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </div>
            <div>
              <label>Whisper model</label>
              <select value={(val("whisper_model") as string) || "large-v3"} onChange={(e) => set("whisper_model", e.target.value)}>
                {WHISPER.map((t) => (
                  <option key={t}>{t}</option>
                ))}
              </select>
            </div>
          </div>

          <div className="grid3">
            <div>
              <label>HF token {s.hf_token_set ? "(set)" : ""}</label>
              <input
                type="password"
                placeholder={s.hf_token_set ? "•••••• (leave blank to keep)" : "hf_…"}
                onChange={(e) => set("hf_token", e.target.value)}
              />
            </div>
            <div>
              <label>Max speakers (0 = auto)</label>
              <input
                type="number"
                min={0}
                value={Number(val("max_speakers") ?? 0)}
                onChange={(e) => set("max_speakers", Number(e.target.value))}
              />
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={!!val("enable_diarization")}
                onChange={(e) => set("enable_diarization", e.target.checked)}
              />
              Diarization
            </label>
          </div>

          <div className="grid3">
            <label className="check">
              <input
                type="checkbox"
                checked={!!val("skip_separation")}
                onChange={(e) => set("skip_separation", e.target.checked)}
              />
              Skip separation
            </label>
            <label className="check">
              <input
                type="checkbox"
                checked={!!val("low_vram")}
                onChange={(e) => set("low_vram", e.target.checked)}
              />
              Low VRAM
            </label>
            <div className="actions">
              <button disabled={saving || !Object.keys(draft).length} onClick={save}>
                {saving ? "Saving…" : "Save settings"}
              </button>
            </div>
          </div>
          {msg && <p className="muted">{msg}</p>}
          {err && <p className="err">{err}</p>}
        </div>
      )}
    </section>
  );
}
