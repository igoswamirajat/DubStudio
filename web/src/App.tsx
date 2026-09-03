import { useEffect, useRef, useState } from "react";
import {
  cancelJob,
  createJob,
  deleteJob,
  downloadUrl,
  getJob,
  health,
  listJobs,
  listSpeakers,
  mediaUrl,
  resumeJob,
  subscribeJob,
  type Job,
  type Speaker,
} from "./api";
import { SpeakerCard } from "./components/SpeakerCard";
import { SegmentEditor } from "./components/SegmentEditor";
import { SettingsPanel } from "./components/SettingsPanel";
import { ProgressPanel } from "./components/ProgressPanel";

const LANGS = [
  { id: "hi", label: "Hindi" },
  { id: "en", label: "English" },
  { id: "es", label: "Spanish" },
  { id: "fr", label: "French" },
  { id: "de", label: "German" },
  { id: "ja", label: "Japanese" },
  { id: "ko", label: "Korean" },
  { id: "zh", label: "Chinese" },
];

const ENGINES = [
  { id: "omnivoice", label: "OmniVoice (primary)" },
  { id: "dummy", label: "Dummy (CI / no GPU)" },
];

const ACTIVE = (s?: string) =>
  !!s && !["completed", "failed", "canceled", "created"].includes(s);

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [lang, setLang] = useState("hi");
  const [srcLang, setSrcLang] = useState("");
  const [skipSep, setSkipSep] = useState(false);
  const [engine, setEngine] = useState("omnivoice");
  const [job, setJob] = useState<Job | null>(null);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [err, setErr] = useState("");
  const [health_, setHealth] = useState<Record<string, unknown> | null>(null);
  const [submitting, setSubmitting] = useState(false);
  const [curTime, setCurTime] = useState(0);
  const [mediaBust, setMediaBust] = useState(0);
  const videoRef = useRef<HTMLVideoElement>(null);

  function refreshHealth() {
    health().then(setHealth).catch(() => setHealth(null));
  }
  function refreshJobs() {
    listJobs().then((r) => setJobs(r.jobs || [])).catch(() => void 0);
  }

  useEffect(() => {
    refreshHealth();
    refreshJobs();
  }, []);

  // Live progress via SSE, with a polling safety net.
  useEffect(() => {
    if (!job?.job_id || !ACTIVE(job.state)) return;
    const id = job.job_id;
    const unsub = subscribeJob(id, (j) => setJob(j));
    const poll = setInterval(() => {
      getJob(id).then(setJob).catch(() => void 0);
    }, 2000);
    return () => {
      unsub();
      clearInterval(poll);
    };
  }, [job?.job_id, job?.state]);

  // Load speaker cards once voices are enrolled.
  useEffect(() => {
    if (!job?.job_id) return;
    if ((job.percent ?? 0) < 65 && ACTIVE(job.state)) return;
    listSpeakers(job.job_id)
      .then((sp) => sp.speakers?.length && setSpeakers(sp.speakers))
      .catch(() => void 0);
  }, [job?.job_id, job?.percent, job?.state]);

  async function start() {
    if (!file || submitting) return;
    setErr("");
    setSpeakers([]);
    setSubmitting(true);
    try {
      const created = await createJob(file, lang, engine, skipSep, srcLang);
      setJob(await getJob(created.job_id));
      refreshJobs();
    } catch (e) {
      setErr(String(e));
    } finally {
      setSubmitting(false);
    }
  }

  async function openJob(id: string) {
    setErr("");
    setSpeakers([]);
    try {
      const j = await getJob(id);
      setJob(j);
      const sp = await listSpeakers(id);
      if (sp.speakers?.length) setSpeakers(sp.speakers);
    } catch (e) {
      setErr(String(e));
    }
  }

  async function onCancel() {
    if (!job) return;
    try {
      await cancelJob(job.job_id);
      setJob(await getJob(job.job_id));
      refreshJobs();
    } catch (e) {
      setErr(String(e));
    }
  }

  async function onResume() {
    if (!job) return;
    try {
      await resumeJob(job.job_id);
      setJob(await getJob(job.job_id));
      refreshJobs();
    } catch (e) {
      setErr(String(e));
    }
  }

  async function onDelete() {
    if (!job) return;
    try {
      await deleteJob(job.job_id);
      setJob(null);
      setSpeakers([]);
      refreshJobs();
    } catch (e) {
      setErr(String(e));
    }
  }

  const done = job?.state === "completed";
  const videoSrc = job
    ? `${mediaUrl(job.job_id, done ? "output" : "source")}?v=${mediaBust}`
    : "";

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Local-first · OmniVoice · BGM preserved</p>
          <h1>DubStudio</h1>
          <p className="lede">
            Upload a clip. Clone speakers or design new voices. Keep original music &amp; SFX.
          </p>
        </div>
        <div className="status-pill">{healthLabel(health_)}</div>
      </header>

      <SettingsPanel onSaved={refreshHealth} />

      <section className="card grid">
        <div>
          <label>Video</label>
          <input
            type="file"
            accept="video/*"
            onChange={(e) => setFile(e.target.files?.[0] ?? null)}
          />
        </div>
        <div>
          <label>Source language</label>
          <select value={srcLang} onChange={(e) => setSrcLang(e.target.value)}>
            <option value="">Auto-detect</option>
            {LANGS.map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>Target language</label>
          <select value={lang} onChange={(e) => setLang(e.target.value)}>
            {LANGS.map((l) => (
              <option key={l.id} value={l.id}>
                {l.label}
              </option>
            ))}
          </select>
        </div>
        <div>
          <label>TTS engine</label>
          <select value={engine} onChange={(e) => setEngine(e.target.value)}>
            {ENGINES.map((e) => (
              <option key={e.id} value={e.id}>
                {e.label}
              </option>
            ))}
          </select>
        </div>
        <label className="check">
          <input type="checkbox" checked={skipSep} onChange={(e) => setSkipSep(e.target.checked)} />
          Skip separation
        </label>
        <div className="actions">
          <button disabled={!file || submitting} onClick={start}>
            {submitting ? "Uploading…" : "Start dub"}
          </button>
        </div>
      </section>

      {jobs.length > 0 && (
        <section className="card">
          <h2 className="hist-h">Recent jobs</h2>
          <div className="hist-list">
            {jobs.slice(0, 8).map((j) => (
              <button
                key={j.job_id}
                className={`hist-row${job?.job_id === j.job_id ? " active" : ""}`}
                onClick={() => openJob(j.job_id)}
              >
                <span className="hist-state">{j.state}</span>
                <span className="hist-lang muted">→ {j.target_language || "?"}</span>
                <code className="muted">{j.job_id.slice(0, 18)}…</code>
              </button>
            ))}
          </div>
        </section>
      )}

      {job && (
        <section className="card">
          <div className="job-head">
            <code className="muted">{job.job_id}</code>
          </div>

          <ProgressPanel job={job} />

          {videoSrc && (
            <video
              ref={videoRef}
              className="preview"
              src={videoSrc}
              controls
              preload="metadata"
              onError={() => setErr("")}
              onTimeUpdate={(e) => setCurTime((e.target as HTMLVideoElement).currentTime)}
            />
          )}

          <div className="downloads">
            {done && (
              <>
                <a className="btn" href={downloadUrl(job.job_id, "mp4")}>
                  Download MP4
                </a>
                <a className="btn ghost" href={downloadUrl(job.job_id, "srt")}>
                  Download SRT
                </a>
              </>
            )}
            {ACTIVE(job.state) && (
              <button className="btn ghost" onClick={onCancel}>
                Cancel
              </button>
            )}
            {(job.state === "failed" || job.state === "canceled") && (
              <button className="btn" onClick={onResume}>
                Resume
              </button>
            )}
            {!ACTIVE(job.state) && (
              <button className="btn danger" onClick={onDelete}>
                Delete
              </button>
            )}
          </div>
          {job.state === "failed" && <p className="err">{job.error || "failed"}</p>}
        </section>
      )}

      {speakers.length > 0 && job && (
        <section>
          <h2>Speakers</h2>
          <p className="muted">
            Clone the original voice, or design a new one. Changes apply to synthesis for this job.
          </p>
          <div className="speaker-grid">
            {speakers.map((s) => (
              <SpeakerCard
                key={s.speaker_id}
                jobId={job.job_id}
                speaker={s}
                onChange={(updated) =>
                  setSpeakers((prev) =>
                    prev.map((x) => (x.speaker_id === updated.speaker_id ? updated : x)),
                  )
                }
              />
            ))}
          </div>
        </section>
      )}

      {job && (job.percent ?? 0) >= 48 && (
        <SegmentEditor
          jobId={job.job_id}
          currentTime={curTime}
          speakerIds={speakers.map((s) => s.speaker_id)}
          onSeek={(t) => {
            if (videoRef.current) {
              videoRef.current.currentTime = t;
              videoRef.current.play().catch(() => void 0);
            }
          }}
          onArtifacts={() => setMediaBust((n) => n + 1)}
        />
      )}

      {err && <p className="err">{err}</p>}
    </main>
  );
}

function healthLabel(h: Record<string, unknown> | null): string {
  if (!h) return "Checking API…";
  const parts: string[] = [];
  parts.push(h.ffmpeg ? "FFmpeg" : "no FFmpeg");
  if (h.gpu) parts.push("GPU");
  if (h.asr) parts.push(`ASR:${h.asr}`);
  if (h.tts) parts.push(`TTS:${h.tts}`);
  return parts.join(" · ");
}
