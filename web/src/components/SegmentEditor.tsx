import { useEffect, useRef, useState } from "react";
import {
  Clock,
  Play,
  RotateCw,
  User,
  Check,
  Subtitles,
  Volume2,
} from "lucide-react";
import {
  listSegments,
  patchSegment,
  resynthSegment,
  type Segment,
} from "../api";

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const ms = Math.floor((t - Math.floor(t)) * 100);
  return `${m}:${s.toString().padStart(2, "0")}.${ms.toString().padStart(2, "0")}`;
}

type Props = {
  jobId: string;
  currentTime?: number;
  speakerIds?: string[];
  onSeek?: (t: number) => void;
  onArtifacts?: () => void;
};

export function SegmentEditor({
  jobId,
  currentTime,
  speakerIds,
  onSeek,
  onArtifacts,
}: Props) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [savedId, setSavedId] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const drafts = useRef<Record<string, string>>({});

  async function reload() {
    try {
      const r = await listSegments(jobId);
      const items = Array.isArray(r) ? r : Array.isArray(r?.segments) ? r.segments : [];
      setSegments(items);
    } catch {
      setSegments([]);
    }
  }

  useEffect(() => {
    reload().catch((e) => setErr(String(e)));
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [jobId]);

  async function saveText(seg: Segment) {
    const text = drafts.current[seg.segment_id];
    if (text == null || text === seg.translated_text) return;
    try {
      await patchSegment(jobId, seg.segment_id, { translated_text: text });
      setSegments((prev) =>
        (prev || []).map((s) =>
          s.segment_id === seg.segment_id ? { ...s, translated_text: text } : s
        )
      );
      setSavedId(seg.segment_id);
      setTimeout(() => setSavedId(null), 1500);
    } catch (e) {
      setErr(String(e));
    }
  }

  async function saveField(seg: Segment, patch: Partial<Segment>) {
    try {
      await patchSegment(jobId, seg.segment_id, patch);
      setSegments((prev) =>
        (prev || []).map((s) =>
          s.segment_id === seg.segment_id ? { ...s, ...patch } : s
        )
      );
    } catch (e) {
      setErr(String(e));
    }
  }

  const speakers = Array.isArray(speakerIds) && speakerIds.length ? speakerIds : ["S00"];

  async function regen(seg: Segment) {
    setBusy(seg.segment_id);
    setErr("");
    try {
      const text = drafts.current[seg.segment_id];
      await resynthSegment(jobId, seg.segment_id, text ?? undefined);
      await reload();
      onArtifacts?.();
    } catch (e) {
      setErr(String(e));
    } finally {
      setBusy(null);
    }
  }

  const safeSegments = Array.isArray(segments) ? segments : [];
  if (!safeSegments.length) return null;

  return (
    <div className="segment-editor-card">
      <div className="segment-editor-header">
        <div>
          <h3 className="section-title">
            <Subtitles size={18} className="title-icon" /> Timeline &amp; Line
            Editor
          </h3>
          <p className="section-subtitle">
            Fine-tune translated dialogue, adjust window timestamps, or trigger
            one-click re-synthesis for any line.
          </p>
        </div>
        <span className="badge-count">{safeSegments.length} lines</span>
      </div>

      {err && <div className="toast-err">{err}</div>}

      <div className="segment-track-list">
        {safeSegments.map((seg, idx) => {
          const active =
            currentTime != null &&
            currentTime >= seg.start &&
            currentTime < seg.end;
          const isBusy = busy === seg.segment_id;
          const isJustSaved = savedId === seg.segment_id;

          return (
            <div
              key={seg.segment_id}
              className={`segment-card-row ${active ? "active-playback" : ""}`}
            >
              {/* Left Column: Jump time & Speaker tag */}
              <div className="seg-meta-col">
                <button
                  className="seg-seek-pill tabular"
                  title="Jump to timecode"
                  onClick={() => onSeek?.(seg.start)}
                >
                  <Play size={11} className="seek-play-icon" />
                  <span>{fmt(seg.start)}</span>
                </button>

                <div className="speaker-select-pill">
                  <User size={12} className="spk-icon" />
                  <select
                    value={seg.speaker_id || "S00"}
                    onChange={(e) =>
                      saveField(seg, { speaker_id: e.target.value })
                    }
                  >
                    {speakers.map((sp) => (
                      <option key={sp} value={sp}>
                        {sp}
                      </option>
                    ))}
                  </select>
                </div>

                <span className="seg-index-tag">#{idx + 1}</span>
              </div>

              {/* Middle Column: Original source & Translated editor */}
              <div className="seg-content-col">
                {seg.source_text && (
                  <p className="seg-original-text">
                    <span className="lang-tag">ORIGINAL</span>
                    {seg.source_text}
                  </p>
                )}

                <div className="seg-editor-wrapper">
                  <textarea
                    className="seg-textarea"
                    defaultValue={seg.translated_text || ""}
                    placeholder="Enter translated dialogue line…"
                    onChange={(e) =>
                      (drafts.current[seg.segment_id] = e.target.value)
                    }
                    onBlur={() => saveText(seg)}
                    rows={2}
                  />
                  {isJustSaved && (
                    <span className="saved-indicator" title="Auto-saved">
                      <Check size={12} /> Saved
                    </span>
                  )}
                </div>

                <div className="seg-timing-controls">
                  <div className="time-input-group">
                    <Clock size={12} />
                    <label>Start</label>
                    <input
                      type="number"
                      step="0.05"
                      defaultValue={seg.start}
                      className="tabular"
                      onBlur={(e) => {
                        const v = parseFloat(e.target.value);
                        if (!Number.isNaN(v) && v !== seg.start)
                          saveField(seg, { start: v });
                      }}
                    />
                    <span>s</span>
                  </div>

                  <div className="time-input-group">
                    <Clock size={12} />
                    <label>End</label>
                    <input
                      type="number"
                      step="0.05"
                      defaultValue={seg.end}
                      className="tabular"
                      onBlur={(e) => {
                        const v = parseFloat(e.target.value);
                        if (!Number.isNaN(v) && v !== seg.end)
                          saveField(seg, { end: v });
                      }}
                    />
                    <span>s</span>
                  </div>

                  <span className="duration-tag tabular">
                    Window: {(seg.end - seg.start).toFixed(2)}s
                  </span>
                </div>
              </div>

              {/* Right Column: Re-synth button */}
              <div className="seg-actions-col">
                <button
                  className={`btn ghost seg-resynth-btn ${isBusy ? "busy" : ""}`}
                  disabled={isBusy}
                  onClick={() => regen(seg)}
                  title="Re-synthesize this line"
                >
                  <RotateCw size={13} className={isBusy ? "spin" : ""} />
                  <span>{isBusy ? "Synthesizing…" : "Re-synth"}</span>
                </button>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
