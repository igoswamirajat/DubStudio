import { useEffect, useState } from "react";
import { createJob, getJob, health } from "./api";

export function App() {
  const [file, setFile] = useState<File | null>(null);
  const [lang, setLang] = useState("hi");
  const [job, setJob] = useState<any>(null);
  const [err, setErr] = useState<string>("");
  const [ff, setFf] = useState<boolean | null>(null);

  useEffect(() => {
    health().then((h) => setFf(h.ffmpeg)).catch(() => setFf(false));
  }, []);

  useEffect(() => {
    if (!job?.job_id || job.state === "completed" || job.state === "failed") return;
    const t = setInterval(async () => {
      try {
        setJob(await getJob(job.job_id));
      } catch (e) {
        setErr(String(e));
      }
    }, 400);
    return () => clearInterval(t);
  }, [job?.job_id, job?.state]);

  async function start() {
    if (!file) return;
    setErr("");
    const created = await createJob(file, lang);
    setJob(await getJob(created.job_id));
  }

  return (
    <main>
      <h1>DubStudio</h1>
      <p className="muted">Local-first AI dubbing. Phase 0 skeleton — stages are stubbed.</p>
      <div className="card">
        <p className="muted">API {ff === null ? "..." : ff ? "FFmpeg found" : "FFmpeg missing (ok for Phase 0)"}</p>
        <label>Video</label>
        <input type="file" accept="video/*" onChange={(e) => setFile(e.target.files?.[0] ?? null)} />
        <label>Target language</label>
        <select value={lang} onChange={(e) => setLang(e.target.value)}>
          <option value="hi">Hindi</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
          <option value="de">German</option>
          <option value="ja">Japanese</option>
        </select>
        <div style={{ marginTop: 16 }}>
          <button disabled={!file} onClick={start}>Start job</button>
        </div>
        {job && (
          <div style={{ marginTop: 20 }}>
            <div className="bar"><i style={{ width: `${job.percent ?? 0}%` }} /></div>
            <p>{job.state} — {job.message}</p>
            <p className="muted">{job.job_id}</p>
          </div>
        )}
        {err && <p>{err}</p>}
      </div>
    </main>
  );
}
