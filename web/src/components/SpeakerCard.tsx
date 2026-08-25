import { useState } from "react";
import type { Speaker } from "../api";
import { patchSpeaker, speakerRefUrl } from "../api";

const MODES = [
  { id: "clone", label: "Clone original" },
  { id: "design", label: "Design new voice" },
  { id: "auto", label: "Auto voice" },
] as const;

type Props = {
  jobId: string;
  speaker: Speaker;
  onChange: (s: Speaker) => void;
};

export function SpeakerCard({ jobId, speaker, onChange }: Props) {
  const [busy, setBusy] = useState(false);
  const [label, setLabel] = useState(speaker.label || speaker.speaker_id);
  const [mode, setMode] = useState(speaker.voice_mode || "clone");
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
    <article className="speaker-card">
      <header>
        <div className="avatar">{(label || "S").slice(0, 1).toUpperCase()}</div>
        <div className="meta">
          <input
            className="label-input"
            value={label}
            disabled={busy}
            onChange={(e) => setLabel(e.target.value)}
            onBlur={() => {
              if (label !== speaker.label) save({ label });
            }}
          />
          <span className="muted">{speaker.speaker_id} · {speaker.segment_count ?? 0} lines</span>
        </div>
      </header>

      <audio controls src={speakerRefUrl(jobId, speaker.speaker_id)} preload="none" />

      <div className="mode-row">
        {MODES.map((m) => (
          <button
            key={m.id}
            type="button"
            className={mode === m.id ? "chip active" : "chip"}
            disabled={busy}
            onClick={async () => {
              setMode(m.id);
              await save({ voice_mode: m.id });
            }}
          >
            {m.label}
          </button>
        ))}
      </div>

      {mode === "design" && (
        <div className="design-box">
          <label>Voice design prompt</label>
          <input
            placeholder='e.g. female, young adult, hindi accent'
            value={design}
            disabled={busy}
            onChange={(e) => setDesign(e.target.value)}
            onBlur={() => {
              if (design !== (speaker.design_prompt || "")) {
                save({ design_prompt: design, voice_mode: "design" });
              }
            }}
          />
          <p className="hint">OmniVoice instruct: gender, age, pitch, accent…</p>
        </div>
      )}

      {err && <p className="err">{err}</p>}
    </article>
  );
}
