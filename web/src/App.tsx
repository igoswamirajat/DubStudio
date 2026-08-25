import { useEffect, useState } from "react";
import {
  createJob,
  downloadUrl,
  getJob,
  health,
  listSpeakers,
  type Job,
  type Speaker,
} from "./api";
import { SpeakerCard } from "./components/SpeakerCard";

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

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [lang, setLang] = useState("hi");
  const [engine, setEngine] = useState("omnivoice");
  const [job, setJob] = useState<Job | null>(null);
  const [speakers, setSpeakers] = useState<Speaker[]>([]);
  const [err, setErr] = useState("");
  const [ff, setFf] = useState<boolean | null>(null);

  useEffect(() => {
    health().then((h) => setFf(!!h.ffmpeg)).catch(() => setFf(false));
  }, []);

  useEffect(() => {
    if (!job?.job_id || job.state === "completed" || job.state === "failed") return;
    const t = setInterval(async () => {
      try {
        const j = await getJob(job.job_id);
        setJob(j);
        if (
          ["enrolling_voices", "synthesizing", "fitting", "mixing", "exporting", "completed"].includes(
            j.state,
          ) ||
          (j.percent ?? 0) >= 65
        ) {
          const sp = await listSpeakers(j.job_id);
          if (sp.speakers?.length) setSpeakers(sp.speakers);
        }
      } catch (e) {
        setErr(String(e));
      }
    }, 500);
    return () => clearInterval(t);
  }, [job?.job_id, job?.state]);

  async function start() {
    if (!file) return;
    setErr("");
    setSpeakers([]);
    try {
      const created = await createJob(file, lang, engine, false);
      setJob(await getJob(created.job_id));
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <main>
      <header className="hero">
        <div>
          <p className="eyebrow">Local-first · OmniVoice · BGM preserved</p>
          <h1>DubStudio</h1>
          <p className="lede">
            Upload a clip. Clone speakers or design new voices. Keep original music & SFX.
          </p>
        </div>
        <div className="status-pill">
          {ff === null ? "Checking API…" : ff ? "API · FFmpeg ready" : "API up · FFmpeg missing"}
        </div>
      </header>

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
        <div className="actions">
          <button disabled={!file} onClick={start}>
            Start dub
          </button>
        </div>
      </section>

      {job && (
        <section className="card">
          <div className="job-head">
            <div>
              <strong>{job.state}</strong>
              <span className="muted"> · {job.message}</span>
            </div>
            <code className="muted">{job.job_id}</code>
          </div>
          <div className="bar">
            <i style={{ width: `${job.percent ?? 0}%` }} />
          </div>
          <p className="muted">{job.percent ?? 0}%</p>

          {job.state === "completed" && (
            <div className="downloads">
              <a className="btn" href={downloadUrl(job.job_id, "mp4")}>
                Download MP4
              </a>
              <a className="btn ghost" href={downloadUrl(job.job_id, "srt")}>
                Download SRT
              </a>
            </div>
          )}
          {job.state === "failed" && <p className="err">{job.error || "failed"}</p>}
        </section>
      )}

      {speakers.length > 0 && (
        <section>
          <h2>Speakers</h2>
          <p className="muted">
            Clone original voice, or design a new one. Changes apply to synthesis for this job.
          </p>
          <div className="speaker-grid">
            {speakers.map((s) => (
              <SpeakerCard
                key={s.speaker_id}
                jobId={job!.job_id}
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

      {err && <p className="err">{err}</p>}
    </main>
  );
}
