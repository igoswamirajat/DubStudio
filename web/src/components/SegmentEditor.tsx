import { useEffect, useRef, useState } from "react";
import { listSegments, patchSegment, resynthSegment, type Segment } from "../api";

function fmt(t: number): string {
  const m = Math.floor(t / 60);
  const s = Math.floor(t % 60);
  const ms = Math.floor((t - Math.floor(t)) * 1000);
  return `${m}:${s.toString().padStart(2, "0")}.${ms.toString().padStart(3, "0")}`;
}

type Props = {
  jobId: string;
  currentTime?: number;
  speakerIds?: string[];
  onSeek?: (t: number) => void;
  onArtifacts?: () => void;
};

export function SegmentEditor({ jobId, currentTime, speakerIds, onSeek, onArtifacts }: Props) {
  const [segments, setSegments] = useState<Segment[]>([]);
  const [busy, setBusy] = useState<string | null>(null);
  const [err, setErr] = useState("");
  const drafts = useRef<Record<string, string>>({});

  async function reload() {
    const r = await listSegments(jobId);
    setSegments(r.segments);
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
        prev.map((s) => (s.segment_id === seg.segment_id ? { ...s, translated_text: text } : s)),
      );
    } catch (e) {
      setErr(String(e));
    }
  }

  async function saveField(seg: Segment, patch: Partial<Segment>) {
    try {
      await patchSegment(jobId, seg.segment_id, patch);
      setSegments((prev) =>
        prev.map((s) => (s.segment_id === seg.segment_id ? { ...s, ...patch } : s)),
      );
    } catch (e) {
      setErr(String(e));
    }
  }

  const speakers = speakerIds && speakerIds.length ? speakerIds : ["S00"];

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

  if (!segments.length) return null;

  return (
    <section>
      <h2>Timeline &amp; transcript</h2>
      <p className="muted">
        Edit a translated line and re-generate just that segment. The final MP4 is re-mixed after each
        re-synth.
      </p>
      {err && <p className="err">{err}</p>}
      <div className="seg-list">
        {segments.map((seg) => {
          const active =
            currentTime != null && currentTime >= seg.start && currentTime < seg.end;
          return (
            <div key={seg.segment_id} className={`seg-row${active ? " active" : ""}`}>
              <button className="seg-time" title="Jump to" onClick={() => onSeek?.(seg.start)}>
                {fmt(seg.start)}
              </button>
              <select
                className="seg-spk-sel"
                value={seg.speaker_id || "S00"}
                onChange={(e) => saveField(seg, { speaker_id: e.target.value })}
              >
                {speakers.map((sp) => (
                  <option key={sp} value={sp}>
                    {sp}
                  </option>
                ))}
              </select>
              <div className="seg-texts">
                {seg.source_text && <p className="seg-src muted">{seg.source_text}</p>}
                <textarea
                  className="seg-edit"
                  defaultValue={seg.translated_text || ""}
                  onChange={(e) => (drafts.current[seg.segment_id] = e.target.value)}
                  onBlur={() => saveText(seg)}
                  rows={2}
                />
                <div className="seg-timing">
                  <label>
                    start
                    <input
                      type="number"
                      step="0.01"
                      defaultValue={seg.start}
                      onBlur={(e) => {
                        const v = parseFloat(e.target.value);
                        if (!Number.isNaN(v) && v !== seg.start) saveField(seg, { start: v });
                      }}
                    />
                  </label>
                  <label>
                    end
                    <input
                      type="number"
                      step="0.01"
                      defaultValue={seg.end}
                      onBlur={(e) => {
                        const v = parseFloat(e.target.value);
                        if (!Number.isNaN(v) && v !== seg.end) saveField(seg, { end: v });
                      }}
                    />
                  </label>
                </div>
              </div>
              <button
                className="btn ghost seg-regen"
                disabled={busy === seg.segment_id}
                onClick={() => regen(seg)}
              >
                {busy === seg.segment_id ? "…" : "Re-synth"}
              </button>
            </div>
          );
        })}
      </div>
    </section>
  );
}
