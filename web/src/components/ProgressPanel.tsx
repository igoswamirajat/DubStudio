import { useEffect, useRef, useState } from "react";
import confetti from "canvas-confetti";
import {
  CheckCircle2,
  AlertCircle,
  Clock,
  Loader2,
  Sparkles,
  Layers,
} from "lucide-react";
import type { Job } from "../api";

const STAGES: { key: string; label: string; desc: string }[] = [
  {
    key: "ingesting",
    label: "Ingest",
    desc: "Validating video container & audio format",
  },
  {
    key: "extracting",
    label: "Audio Extract",
    desc: "Extracting 48kHz PCM audio with FFmpeg",
  },
  {
    key: "separating",
    label: "Stem Separation",
    desc: "Separating dialogue from background music & SFX (Demucs)",
  },
  {
    key: "transcribing",
    label: "Transcription",
    desc: "Transcribing word timestamps with Faster-Whisper",
  },
  {
    key: "diarizing",
    label: "Diarization",
    desc: "Clustering & labeling speaker voices (Pyannote)",
  },
  {
    key: "segmenting",
    label: "Segmentation",
    desc: "Assembling speaker-aware dialogue segments",
  },
  {
    key: "translating",
    label: "Translation",
    desc: "Translating dialogue lines preserving tone & timing",
  },
  {
    key: "enrolling_voices",
    label: "Voice Enrollment",
    desc: "Capturing vocal timbre reference per speaker",
  },
  {
    key: "synthesizing",
    label: "Voice Synthesis",
    desc: "Synthesizing target speech with OmniVoice / neural TTS",
  },
  {
    key: "fitting",
    label: "Timing Fit",
    desc: "Tempo-fitting audio into original dialogue window",
  },
  {
    key: "mixing",
    label: "Audio Mix",
    desc: "Ducking original bed and mixing speech at -10dB",
  },
  {
    key: "exporting",
    label: "Final Export",
    desc: "Remuxing MP4 video stream + generating SRT subtitles",
  },
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
    const id = setInterval(
      () => setT(Math.floor((Date.now() - (start.current || Date.now())) / 1000)),
      1000
    );
    return () => clearInterval(id);
  }, [active]);

  return t;
}

function fmtTime(s: number): string {
  const m = Math.floor(s / 60);
  const sec = s % 60;
  return m ? `${m}m ${sec}s` : `${sec}s`;
}

export function ProgressPanel({ job }: { job: Job }) {
  const state = job.state;
  const active = !TERMINAL.includes(state) && state !== "created";
  const failed = state === "failed";
  const done = state === "completed";
  const elapsed = useElapsed(active);
  const celebratedRef = useRef(false);

  const currentIdx = STAGES.findIndex((s) => s.key === state);
  const activeStage = currentIdx >= 0 ? STAGES[currentIdx] : null;
  const percent = Math.max(0, Math.min(100, job.percent ?? 0));

  // Confetti on completion
  useEffect(() => {
    if (done && !celebratedRef.current) {
      celebratedRef.current = true;
      try {
        confetti({
          particleCount: 80,
          spread: 70,
          origin: { y: 0.6 },
          colors: ["#6366f1", "#06b6d4", "#10b981", "#8b5cf6"],
        });
      } catch {
        // ignore
      }
    }
    if (!done) {
      celebratedRef.current = false;
    }
  }, [done]);

  return (
    <div className="progress-container">
      {/* Header with state & timer */}
      <div className="progress-header">
        <div className="progress-status-badge">
          {done ? (
            <CheckCircle2 size={18} className="icon-success" />
          ) : failed ? (
            <AlertCircle size={18} className="icon-danger" />
          ) : active ? (
            <Loader2 size={18} className="spin icon-accent" />
          ) : (
            <Clock size={18} className="icon-muted" />
          )}
          <span className="status-title">
            {state.replace(/_/g, " ").toUpperCase()}
          </span>
          {active && activeStage && (
            <span className="status-subtext">· {activeStage.desc}</span>
          )}
          {job.message && !activeStage && (
            <span className="status-subtext">· {job.message}</span>
          )}
        </div>

        <div className="progress-time-meta">
          <Layers size={14} className="meta-icon" />
          <span>
            {job.stage_index
              ? `Stage ${job.stage_index}/${job.stage_total || 12}`
              : `${done ? "12/12 Stages" : "12 Stages"}`}
          </span>
          {active && (
            <>
              <span className="meta-divider">•</span>
              <Clock size={14} className="meta-icon" />
              <span className="tabular">{fmtTime(elapsed)}</span>
            </>
          )}
        </div>
      </div>

      {/* Main Bar */}
      <div className={`progress-bar-track ${failed ? "failed" : ""}`}>
        <div
          className={`progress-bar-fill ${active ? "active-glow" : ""} ${done ? "done-glow" : ""}`}
          style={{ width: `${failed ? 100 : percent}%` }}
        />
        <span className="progress-percent-label">
          {failed ? "Pipeline Error" : `${percent}%`}
        </span>
      </div>

      {/* 12-Stage Stepper Track */}
      <div className="pipeline-stepper">
        {STAGES.map((s, i) => {
          const isDone = done || (currentIdx >= 0 && i < currentIdx);
          const isCurrent = i === currentIdx && active;
          const isFail = failed && i === currentIdx;

          return (
            <div
              key={s.key}
              className={`pipeline-step ${isDone ? "is-done" : ""} ${isCurrent ? "is-current" : ""} ${isFail ? "is-fail" : ""}`}
              title={`${s.label}: ${s.desc}`}
            >
              <div className="step-bullet">
                {isDone ? (
                  <CheckCircle2 size={12} className="bullet-check" />
                ) : isCurrent ? (
                  <span className="bullet-pulse" />
                ) : isFail ? (
                  <span className="bullet-fail">✕</span>
                ) : (
                  <span className="bullet-num">{i + 1}</span>
                )}
              </div>
              <span className="step-name">{s.label}</span>
            </div>
          );
        })}
      </div>
    </div>
  );
}
