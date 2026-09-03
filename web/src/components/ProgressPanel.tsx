import { useEffect, useRef, useState } from "react";
import type { Job } from "../api";

// The 12 pipeline stages, in order, with friendly labels + icons.
const STAGES: { key: string; label: string; icon: string }[] = [
  { key: "ingesting", label: "Ingest", icon: "📥" },
  { key: "extracting", label: "Extract audio", icon: "🎧" },
  { key: "separating", label: "Separate music", icon: "🎼" },
  { key: "transcribing", label: "Transcribe", icon: "📝" },
  { key: "diarizing", label: "Speakers", icon: "🗣️" },
  { key: "segmenting", label: "Segments", icon: "✂️" },
  { key: "translating", label: "Translate", icon: "🌐" },
  { key: "enrolling_voices", label: "Enroll voices", icon: "🎙️" },
  { key: "synthesizing", label: "Synthesize", icon: "🔊" },
  { key: "fitting", label: "Fit timing", icon: "⏱️" },
  { key: "mixing", label: "Mix", icon: "🎚️" },
  { key: "exporting", label: "Export", icon: "🎬" },
];

const TERMINAL = ["completed", "failed", "canceled"];

function useElapsed(active: boolean): number {
  const [t, setT] = useState(0);
  const start = useRef<number | null>(null);
  useEffect(() => {
    if (!active) {
      start.current = null;
      return;
    }
    start.current = Date.now();
    const id = setInterval(() => setT(Math.floor((Date.now() - (start.current || Date.now())) / 1000)), 1000);
    return () => clearInterval(id);
  }, [active]);
  return t;
}

export function ProgressPanel({ job }: { job: Job }) {
  const state = job.state;
  const active = !TERMINAL.includes(state) && state !== "created";
  const failed = state === "failed";
  const done = state === "completed";
  const elapsed = useElapsed(active);

  const currentIdx = STAGES.findIndex((s) => s.key === state);
  const percent = Math.max(0, Math.min(100, job.percent ?? 0));

  return (
    <div className="progress-wrap">
      <div className="progress-top">
        <div className="progress-title">
          <span className={`dot ${done ? "ok" : failed ? "bad" : active ? "live" : ""}`} />
          <strong className="cap">{state.replace(/_/g, " ")}</strong>
          {job.message && <span className="muted"> · {job.message}</span>}
        </div>
        <div className="progress-meta muted">
          {job.stage_index ? `Step ${job.stage_index}/${job.stage_total ?? 12}` : ""}
          {active && ` · ${fmtTime(elapsed)}`}
        </div>
      </div>

      <div className={`bar big ${failed ? "bar-fail" : ""}`}>
        <i className={active ? "striped" : ""} style={{ width: `${failed ? 100 : percent}%` }} />
        <span className="bar-pct">{failed ? "failed" : `${percent}%`}</span>
      </div>

      <div className="stepper">
        {STAGES.map((s, i) => {
          const isDone = done || (currentIdx >= 0 && i < currentIdx);
          const isCurrent = i === currentIdx && active;
          const isFail = failed && i === currentIdx;
          return (
            <div
              key={s.key}
              className={`step${isDone ? " done" : ""}${isCurrent ? " current" : ""}${isFail ? " fail" : ""}`}
              title={s.label}
            >
              <span className="step-ico">{isDone ? "✓" : isFail ? "✕" : s.icon}</span>
              <span className="step-lbl">{s.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m ? `${m}m ${sec}s` : `${sec}s`;
}
