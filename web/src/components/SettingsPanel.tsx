import { useEffect, useState } from "react";
import {
  Settings as SettingsIcon,
  Globe,
  Cpu,
  Radio,
  Sliders,
  Check,
  ChevronDown,
  ChevronUp,
  Save,
  Key,
} from "lucide-react";
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
      setMsg("Studio configuration updated and saved to .env");
      setTimeout(() => setMsg(""), 3000);
      onSaved?.();
    } catch (e) {
      setErr(String(e));
    } finally {
      setSaving(false);
    }
  }

  const translator = (val("translator") as string) || "ollama";

  return (
    <section className="settings-card">
      <button
        className="settings-toggle-trigger"
        onClick={() => setOpen((o) => !o)}
      >
        <div className="trigger-left">
          <SettingsIcon size={16} className="trigger-icon" />
          <span className="trigger-title">Studio Engine &amp; AI Settings</span>
          <span className="badge-pill">Runtime Config</span>
        </div>
        <div className="trigger-right">
          {open ? <ChevronUp size={16} /> : <ChevronDown size={16} />}
        </div>
      </button>

      {open && !s && <div className="settings-loading">Loading studio config…</div>}

      {open && s && (
        <div className="settings-content-body">
          {/* Section 1: Translation Provider */}
          <div className="settings-section">
            <h4 className="section-head">
              <Globe size={14} /> AI Translation Engine
            </h4>
            <div className="settings-grid">
              <div className="input-group">
                <label>Provider</label>
                <select
                  value={translator}
                  onChange={(e) => set("translator", e.target.value)}
                >
                  {TRANSLATORS.map((t) => (
                    <option key={t} value={t}>
                      {t.toUpperCase()}
                    </option>
                  ))}
                </select>
              </div>

              {translator === "ollama" && (
                <>
                  <div className="input-group">
                    <label>Ollama Host URL</label>
                    <input
                      type="text"
                      value={(val("ollama_host") as string) || ""}
                      placeholder="http://127.0.0.1:11434"
                      onChange={(e) => set("ollama_host", e.target.value)}
                    />
                  </div>
                  <div className="input-group">
                    <label>Model Name</label>
                    <input
                      type="text"
                      value={(val("ollama_model") as string) || ""}
                      placeholder="qwen2.5:7b"
                      onChange={(e) => set("ollama_model", e.target.value)}
                    />
                  </div>
                </>
              )}

              {translator === "openai" && (
                <>
                  <div className="input-group">
                    <label>API Base URL</label>
                    <input
                      type="text"
                      value={(val("openai_base_url") as string) || ""}
                      placeholder="https://api.openai.com/v1"
                      onChange={(e) => set("openai_base_url", e.target.value)}
                    />
                  </div>
                  <div className="input-group">
                    <label>Model</label>
                    <input
                      type="text"
                      value={(val("openai_model") as string) || ""}
                      placeholder="gpt-4o-mini"
                      onChange={(e) => set("openai_model", e.target.value)}
                    />
                  </div>
                  <div className="input-group">
                    <label>API Key {s.openai_api_key_set && "✓ (Saved)"}</label>
                    <input
                      type="password"
                      placeholder={
                        s.openai_api_key_set ? "•••••••• (Leave blank to keep)" : "sk-..."
                      }
                      onChange={(e) => set("openai_api_key", e.target.value)}
                    />
                  </div>
                </>
              )}
            </div>
          </div>

          {/* Section 2: Speech & Voice Models */}
          <div className="settings-section">
            <h4 className="section-head">
              <Cpu size={14} /> Voice &amp; Transcription Engines
            </h4>
            <div className="settings-grid">
              <div className="input-group">
                <label>TTS Primary Engine</label>
                <select
                  value={(val("tts_engine") as string) || "omnivoice"}
                  onChange={(e) => set("tts_engine", e.target.value)}
                >
                  {TTS.map((t) => (
                    <option key={t} value={t}>
                      {t}
                    </option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label>ASR Speech-to-Text</label>
                <select
                  value={(val("asr_engine") as string) || "faster-whisper"}
                  onChange={(e) => set("asr_engine", e.target.value)}
                >
                  {ASR.map((a) => (
                    <option key={a} value={a}>
                      {a}
                    </option>
                  ))}
                </select>
              </div>

              <div className="input-group">
                <label>Whisper Model Size</label>
                <select
                  value={(val("whisper_model") as string) || "large-v3"}
                  onChange={(e) => set("whisper_model", e.target.value)}
                >
                  {WHISPER.map((w) => (
                    <option key={w} value={w}>
                      {w}
                    </option>
                  ))}
                </select>
              </div>
            </div>
          </div>

          {/* Section 3: Hardware & Separation */}
          <div className="settings-section">
            <h4 className="section-head">
              <Sliders size={14} /> Separation &amp; Hardware Flags
            </h4>
            <div className="checkbox-row">
              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={Boolean(val("enable_diarization"))}
                  onChange={(e) => set("enable_diarization", e.target.checked)}
                />
                <span className="toggle-slider" />
                <span className="toggle-text">Pyannote Speaker Diarization</span>
              </label>

              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={Boolean(val("skip_separation"))}
                  onChange={(e) => set("skip_separation", e.target.checked)}
                />
                <span className="toggle-slider" />
                <span className="toggle-text">Skip Demucs Separation (Fast demo mode)</span>
              </label>

              <label className="toggle-switch">
                <input
                  type="checkbox"
                  checked={Boolean(val("low_vram"))}
                  onChange={(e) => set("low_vram", e.target.checked)}
                />
                <span className="toggle-slider" />
                <span className="toggle-text">Low VRAM Optimization (sequential models)</span>
              </label>
            </div>
          </div>

          {/* Footer Actions */}
          <div className="settings-footer">
            <button
              className="btn primary"
              disabled={saving || Object.keys(draft).length === 0}
              onClick={save}
            >
              <Save size={14} />
              <span>{saving ? "Saving Changes…" : "Save Studio Config"}</span>
            </button>

            {msg && (
              <span className="settings-feedback-toast">
                <Check size={14} /> {msg}
              </span>
            )}
            {err && <span className="settings-err-toast">{err}</span>}
          </div>
        </div>
      )}
    </section>
  );
}
