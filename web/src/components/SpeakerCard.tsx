import { useState } from "react";
import { Mic, Wand2, Bot, Volume2, Sparkles, UserCheck } from "lucide-react";
import type { Speaker } from "../api";
import { patchSpeaker, speakerRefUrl } from "../api";

const MODES = [
  { id: "clone", label: "Clone Original", icon: Mic },
  { id: "native", label: "Native Character", icon: UserCheck },
  { id: "design", label: "Design Voice", icon: Wand2 },
] as const;

const VEENA_OPTIONS = [
  { id: "agastya", label: "Agastya (♂ Tech Explainer)" },
  { id: "vinaya", label: "Vinaya (♂ Deep Narrator)" },
  { id: "kavya", label: "Kavya (♀ Bright Dynamic)" },
  { id: "maitri", label: "Maitri (♀ Warm Conversational)" },
];

const PRESETS = [
  "male, deep narrator, calm tone",
  "female, young adult, clear diction",
  "energetic male, conversational",
  "authoritative female, documentary",
];

type Props = {
  jobId: string;
  speaker: Speaker;
  onChange: (s: Speaker) => void;
};

export function SpeakerCard({ jobId, speaker, onChange }: Props) {
  const [busy, setBusy] = useState(false);
  const [label, setLabel] = useState(speaker.label || speaker.speaker_id);
  const [mode, setMode] = useState(speaker.voice_mode || "clone");
  const [voiceId, setVoiceId] = useState(speaker.voice_id || "agastya");
  const [design, setDesign] = useState(speaker.design_prompt || "");
  const [err, setErr] = useState("");

  async function save(patch: Partial<Speaker>) {
    setBusy(true);
    setErr("");
    try {
      const updated = await patchSpeaker(jobId, speaker.speaker_id, patch);
      onChange(updated);
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(false);
    }
  }

  return (
    <article className="modern-speaker-card">
      <header className="spk-header">
        <div className="spk-avatar">
          {(label || "S").slice(0, 1).toUpperCase()}
        </div>
        <div className="spk-meta">
          <input
            className="spk-label-input"
            value={label}
            disabled={busy}
            placeholder="Speaker name"
            onChange={(e) => setLabel(e.target.value)}
            onBlur={() => {
              if (label !== speaker.label) save({ label });
            }}
          />
          <span className="spk-subinfo">
            <code>{speaker.speaker_id}</code> · {speaker.segment_count ?? 0}{" "}
            lines spoken
          </span>
        </div>
      </header>

      {/* Reference Audio Player */}
      <div className="spk-audio-box">
        <div className="audio-label">
          <Volume2 size={13} className="audio-icon" />
          <span>Vocal Reference Sample (Original)</span>
        </div>
        <audio
          controls
          src={speakerRefUrl(jobId, speaker.speaker_id)}
          preload="none"
          className="spk-audio-player"
        />
      </div>

      {/* Voice Mode Selector Pills */}
      <div className="spk-mode-pills">
        {MODES.map((m) => {
          const Icon = m.icon;
          const isActive = mode === m.id;
          return (
            <button
              key={m.id}
              type="button"
              className={`mode-pill ${isActive ? "active" : ""}`}
              disabled={busy}
              onClick={async () => {
                setMode(m.id);
                await save({ voice_mode: m.id });
              }}
            >
              <Icon size={13} />
              <span>{m.label}</span>
            </button>
          );
        })}
      </div>

      {/* Native Voice Selector Field */}
      {mode === "native" && (
        <div className="spk-design-section">
          <label className="design-label">
            <UserCheck size={12} /> Veena Character Voice
          </label>
          <select
            className="design-input"
            value={voiceId}
            disabled={busy}
            onChange={(e) => {
              const vid = e.target.value;
              setVoiceId(vid);
              save({ voice_id: vid, voice_mode: "native" });
            }}
          >
            {VEENA_OPTIONS.map((opt) => (
              <option key={opt.id} value={opt.id}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      )}

      {/* Voice Design Instruct Field */}
      {mode === "design" && (
        <div className="spk-design-section">
          <label className="design-label">
            <Sparkles size={12} /> Voice Prompt (OmniVoice instruct)
          </label>
          <input
            className="design-input"
            placeholder="e.g. female, young adult, hindi accent"
            value={design}
            disabled={busy}
            onChange={(e) => setDesign(e.target.value)}
            onBlur={() => {
              if (design !== (speaker.design_prompt || "")) {
                save({ design_prompt: design, voice_mode: "design" });
              }
            }}
          />
          <div className="preset-chips">
            {PRESETS.map((p) => (
              <button
                key={p}
                type="button"
                className="preset-chip"
                onClick={() => {
                  setDesign(p);
                  save({ design_prompt: p, voice_mode: "design" });
                }}
              >
                {p}
              </button>
            ))}
          </div>
        </div>
      )}

      {err && <p className="spk-err">{err}</p>}
    </article>
  );
}
