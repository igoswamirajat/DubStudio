import { useEffect, useRef, useState } from "react";
import {
  UploadCloud,
  Film,
  Sparkles,
  Play,
  RotateCcw,
  Trash2,
  Download,
  AlertCircle,
  FileText,
  Clock,
  ChevronRight,
  RefreshCw,
  Sliders,
  CheckCircle2,
  HardDrive,
} from "lucide-react";
import {
  cancelJob,
  createJob,
  deleteJob,
  downloadUrl,
  getJob,
  getStorageInfo,
  health,
  listJobs,
  listSpeakers,
  resumeJob,
  subscribeJob,
  type Job,
  type Speaker,
  type StorageInfo,
} from "./api";
import { StudioPlayer } from "./components/StudioPlayer";
import { SpeakerCard } from "./components/SpeakerCard";
import { SegmentEditor } from "./components/SegmentEditor";
import { SettingsPanel } from "./components/SettingsPanel";
import { ProgressPanel } from "./components/ProgressPanel";
import { ModelManagerModal } from "./components/ModelManagerModal";
import { VoiceSelectionPanel } from "./components/VoiceSelectionPanel";

const LANGS = [
  { id: "hi", label: "Hindi (हिन्दी)" },
  { id: "en", label: "English" },
  { id: "es", label: "Spanish (Español)" },
  { id: "fr", label: "French (Français)" },
  { id: "de", label: "German (Deutsch)" },
  { id: "ja", label: "Japanese (日本語)" },
  { id: "ko", label: "Korean (한국어)" },
  { id: "zh", label: "Chinese (中文)" },
];

const ENGINES = [
  { id: "omnivoice", label: "OmniVoice (Recommended — 100% Consistent Neural Cloning)" },
  { id: "veena", label: "Veena by Maya Research (Experimental Indian Character Voices)" },
  { id: "dummy", label: "Dummy Engine (Fast CI / Mock)" },
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
  const [health_, setHealth] = useState<Record<string, unknown>>({});
  const [submitting, setSubmitting] = useState(false);
  const [curTime, setCurTime] = useState(0);
  const [mediaBust, setMediaBust] = useState(0);
  const [activeSegmentText, setActiveSegmentText] = useState("");
  const [showModelManager, setShowModelManager] = useState(false);
  const [storageInfo, setStorageInfo] = useState<StorageInfo | null>(null);
  const videoRef = useRef<HTMLVideoElement>(null);

  function refreshStorage() {
    getStorageInfo().then(setStorageInfo).catch(() => void 0);
  }

  function refreshHealth() {
    health().then(setHealth).catch(() => setHealth({}));
  }

  function refreshJobs() {
    listJobs()
      .then((r) => setJobs(r.jobs || []))
      .catch(() => void 0);
  }

  // Handle URL param ?job=<job_id>
  useEffect(() => {
    refreshHealth();
    refreshJobs();
    refreshStorage();

    const params = new URLSearchParams(window.location.search);
    const initialJobId = params.get("job");
    if (initialJobId) {
      openJob(initialJobId);
    }
  }, []);

  // Update URL search query when job changes
  function selectJob(j: Job | null) {
    setJob(j);
    const url = new URL(window.location.href);
    if (j?.job_id) {
      url.searchParams.set("job", j.job_id);
    } else {
      url.searchParams.delete("job");
    }
    window.history.replaceState({}, "", url.toString());
  }

  // Live SSE progress subscription
  useEffect(() => {
    if (!job?.job_id || !ACTIVE(job.state)) return;
    const id = job.job_id;
    const unsub = subscribeJob(id, (j) => {
      setJob((prev) => (prev?.job_id === id ? { ...prev, ...j } : prev));
    });
    const poll = setInterval(() => {
      getJob(id)
        .then((j) => setJob((prev) => (prev?.job_id === id ? j : prev)))
        .catch(() => void 0);
    }, 2000);
    return () => {
      unsub();
      clearInterval(poll);
    };
  }, [job?.job_id, job?.state]);

  // Fetch speakers when voices are enrolled or job completed
  useEffect(() => {
    if (!job?.job_id) return;
    if ((job.percent ?? 0) < 55 && ACTIVE(job.state)) return;
    listSpeakers(job.job_id)
      .then((sp) => {
        if (sp.speakers?.length) setSpeakers(sp.speakers);
      })
      .catch(() => void 0);
  }, [job?.job_id, job?.percent, job?.state]);

  async function start() {
    if (!file || submitting) return;
    setErr("");
    setSpeakers([]);
    setSubmitting(true);
    try {
      const created = await createJob(file, lang, engine, skipSep, srcLang);
      const j = await getJob(created.job_id);
      selectJob(j);
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
      selectJob(j);
      const sp = await listSpeakers(id);
      if (sp.speakers?.length) setSpeakers(sp.speakers);
    } catch (e) {
      setErr(`Failed to open project ${id}: ${String(e)}`);
    }
  }

  async function onCancel() {
    if (!job) return;
    try {
      await cancelJob(job.job_id);
      const updated = await getJob(job.job_id);
      setJob(updated);
      refreshJobs();
    } catch (e) {
      setErr(String(e));
    }
  }

  async function onResume() {
    if (!job) return;
    try {
      await resumeJob(job.job_id);
      const updated = await getJob(job.job_id);
      setJob(updated);
      refreshJobs();
    } catch (e) {
      setErr(String(e));
    }
  }

  async function onDelete() {
    if (!job) return;
    try {
      await deleteJob(job.job_id);
      selectJob(null);
      setSpeakers([]);
      refreshJobs();
    } catch (e) {
      setErr(String(e));
    }
  }

  const isDone = job?.state === "completed";
  const safeJobs = Array.isArray(jobs) ? jobs : [];
  const safeSpeakers = Array.isArray(speakers) ? speakers : [];

  return (
    <div className="studio-app">
      {/* Studio Header */}
      <header className="studio-topbar">
        <div className="topbar-branding">
          <div className="studio-logo-icon">
            <Sparkles size={20} />
          </div>
          <div>
            <h1 className="studio-title">DubStudio</h1>
            <p className="studio-tagline">
              Local-first AI Video Dubbing &amp; Voice Cloning Studio
            </p>
          </div>
        </div>

        <div className="topbar-controls">
          <button
            className="btn secondary model-storage-trigger-btn"
            onClick={() => setShowModelManager(true)}
            title="Manage AI Model Weights & Storage Drive"
          >
            <HardDrive size={15} />
            <span>AI Models &amp; Storage</span>
            {storageInfo && (
              <span className={`badge-pill ${storageInfo.is_low_space ? "warning" : ""}`}>
                {storageInfo.drive} {storageInfo.free_gb} GB free
              </span>
            )}
          </button>

          <div className="health-badge-container">
            <span
              className={`health-dot ${health_.ok ? "live" : "offline"}`}
            />
            <span className="health-text">{healthSummary(health_)}</span>
          </div>
          <button
            className="icon-circle-btn"
            onClick={() => {
              refreshHealth();
              refreshJobs();
              refreshStorage();
            }}
            title="Refresh Studio Status"
          >
            <RefreshCw size={15} />
          </button>
        </div>
      </header>

      <main className="studio-main-layout">
        {/* Global Settings Drawer */}
        <SettingsPanel onSaved={refreshHealth} />

        {/* Top Studio Grid: Upload & Studio Workspace */}
        <div className="studio-grid-layout">
          {/* Left / Top: Upload Card */}
          <section className="studio-card create-job-card">
            <h2 className="card-headline">
              <UploadCloud size={18} className="headline-icon" /> New Dubbing Job
            </h2>
            <p className="card-subtext">
              Drop any video file. Dialogue will be translated and voice-cloned
              while preserving original background music &amp; SFX.
            </p>

            <div className="upload-dropzone">
              <input
                type="file"
                id="video-upload-input"
                accept="video/*"
                onChange={(e) => setFile(e.target.files?.[0] ?? null)}
              />
              <label htmlFor="video-upload-input" className="dropzone-label">
                <Film size={28} className="dropzone-icon" />
                <span className="dropzone-title">
                  {file ? file.name : "Choose a video clip or drag it here"}
                </span>
                <span className="dropzone-meta">
                  {file
                    ? `${(file.size / (1024 * 1024)).toFixed(1)} MB`
                    : "MP4, MOV, WebM, MKV up to 500 MB"}
                </span>
              </label>
            </div>

            <div className="job-options-grid">
              <div className="option-field">
                <label>Source Language</label>
                <select
                  value={srcLang}
                  onChange={(e) => setSrcLang(e.target.value)}
                >
                  <option value="">Auto-detect (Whisper)</option>
                  {LANGS.map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="option-field">
                <label>Target Language</label>
                <select
                  value={lang}
                  onChange={(e) => setLang(e.target.value)}
                >
                  {LANGS.map((l) => (
                    <option key={l.id} value={l.id}>
                      {l.label}
                    </option>
                  ))}
                </select>
              </div>

              <div className="option-field">
                <label>Voice Clone Engine</label>
                <select
                  value={engine}
                  onChange={(e) => setEngine(e.target.value)}
                >
                  {ENGINES.map((e) => (
                    <option key={e.id} value={e.id}>
                      {e.label}
                    </option>
                  ))}
                </select>
              </div>
            </div>

            {lang === "hi" && (
              <div className="engine-tip-banner">
                <Sparkles size={14} className="tip-icon text-accent" />
                <span className="tip-text">
                  <strong>Recommended:</strong> <strong>OmniVoice</strong> clones the presenter's exact vocal timbre, pitch, and gender with 100% consistency across all dialogue segments.
                </span>
                <button
                  type="button"
                  className="tip-action-btn"
                  onClick={() => setShowModelManager(true)}
                >
                  Manage AI Models →
                </button>
              </div>
            )}

            <div className="job-flags-row">
              <label className="checkbox-pill">
                <input
                  type="checkbox"
                  checked={skipSep}
                  onChange={(e) => setSkipSep(e.target.checked)}
                />
                <span>Skip Demucs Separation (Faster preview)</span>
              </label>
            </div>

            <div className="submit-action-bar">
              <button
                className="btn primary large-btn"
                disabled={!file || submitting}
                onClick={start}
              >
                <Sparkles size={16} />
                <span>
                  {submitting ? "Uploading Video Clip…" : "Start AI Dubbing"}
                </span>
              </button>
            </div>
          </section>

          {/* Right / Top: Recent Projects Drawer */}
          {safeJobs.length > 0 && (
            <aside className="studio-card recent-jobs-panel">
              <div className="panel-header">
                <h3>
                  <Clock size={16} /> Recent Projects
                </h3>
                <span className="badge-count">{safeJobs.length}</span>
              </div>
              <div className="recent-jobs-scroller">
                {safeJobs.slice(0, 10).map((j) => {
                  const isSelected = job?.job_id === j.job_id;
                  const isCompleted = j.state === "completed";
                  const isFailed = j.state === "failed";

                  return (
                    <button
                      key={j.job_id}
                      className={`recent-job-row ${isSelected ? "selected" : ""}`}
                      onClick={() => openJob(j.job_id)}
                    >
                      <div className="row-left">
                        <span
                          className={`state-bullet ${
                            isCompleted ? "ok" : isFailed ? "err" : "live"
                          }`}
                        />
                        <div className="job-info">
                          <span className="job-filename">
                            {j.source_filename || j.job_id.slice(0, 20)}
                          </span>
                          <span className="job-sub">
                            {(j.target_language || "hi").toUpperCase()} ·{" "}
                            {j.state.replace(/_/g, " ")}
                          </span>
                        </div>
                      </div>
                      <ChevronRight size={14} className="chevron" />
                    </button>
                  );
                })}
              </div>
            </aside>
          )}
        </div>

        {/* Active Project Workspace */}
        {job && (
          <section className="studio-card active-job-workspace">
            <div className="workspace-header">
              <div className="workspace-title-group">
                <span className="job-id-pill">
                  <code>{job.job_id}</code>
                </span>
                <span
                  className={`status-chip ${
                    isDone ? "chip-ok" : job.state === "failed" ? "chip-fail" : "chip-live"
                  }`}
                >
                  {job.state.replace(/_/g, " ").toUpperCase()}
                </span>
                {job.duration_s && (
                  <span className="duration-pill">
                    {job.duration_s.toFixed(1)}s duration
                  </span>
                )}
              </div>

              <div className="workspace-actions">
                {isDone && (
                  <>
                    <a
                      className="btn primary"
                      href={downloadUrl(job.job_id, "mp4")}
                      download="dubbed_output.mp4"
                    >
                      <Download size={14} /> Download Dubbed MP4
                    </a>
                    <a
                      className="btn ghost"
                      href={downloadUrl(job.job_id, "srt")}
                      download="subtitles.srt"
                    >
                      <FileText size={14} /> Download SRT
                    </a>
                  </>
                )}
                {ACTIVE(job.state) && (
                  <button className="btn ghost" onClick={onCancel}>
                    Cancel Job
                  </button>
                )}
                {(job.state === "failed" || job.state === "canceled") && (
                  <button className="btn warning" onClick={onResume}>
                    <RotateCcw size={14} /> Resume Checkpoint
                  </button>
                )}
                {!ACTIVE(job.state) && (
                  <button
                    className="btn danger ghost"
                    onClick={onDelete}
                    title="Delete Job"
                  >
                    <Trash2 size={14} /> Delete
                  </button>
                )}
              </div>
            </div>

            {/* 12-Stage Honest Pipeline Progress */}
            <ProgressPanel job={job} />

            {/* High-End Studio Video Player */}
            <StudioPlayer
              job={job}
              currentTime={curTime}
              activeSegmentText={activeSegmentText}
              mediaBust={mediaBust}
              videoRef={videoRef}
              onTimeUpdate={(t) => setCurTime(t)}
            />

            {job.state === "failed" && job.error && (
              <div className="failure-alert">
                <AlertCircle size={18} />
                <div className="alert-content">
                  <strong>Pipeline Stalled:</strong>
                  <p>{job.error}</p>
                </div>
              </div>
            )}
          </section>
        )}

        {/* Human-in-the-Loop Voice Selection Panel */}
        {job && job.state === "awaiting_voice_selection" && (
          <VoiceSelectionPanel
            jobId={job.job_id}
            onApplied={async () => {
              const updated = await getJob(job.job_id);
              setJob(updated);
              refreshJobs();
            }}
          />
        )}

        {/* Speaker Enrollment & Voice Cloning Section (Post-Completion & Custom Tuning) */}
        {safeSpeakers.length > 0 && job && job.state !== "awaiting_voice_selection" && (
          <section className="studio-card speakers-section">
            <div className="section-header">
              <h2 className="section-title">
                <Sparkles size={18} className="title-icon" /> Speaker Timbre &amp;
                Voice Cards
              </h2>
              <p className="section-subtitle">
                Original vocal samples are automatically isolated. Choose whether
                to clone each voice verbatim, design a new voice with natural
                language prompts, or auto-match.
              </p>
            </div>

            <div className="speakers-grid">
              {safeSpeakers.map((s) => (
                <SpeakerCard
                  key={s.speaker_id}
                  jobId={job.job_id}
                  speaker={s}
                  onChange={(updated) =>
                    setSpeakers((prev) =>
                      (prev || []).map((x) =>
                        x.speaker_id === updated.speaker_id ? updated : x
                      )
                    )
                  }
                />
              ))}
            </div>
          </section>
        )}

        {/* Interactive Segment Timeline Editor */}
        {job && (job.percent ?? 0) >= 45 && (
          <SegmentEditor
            jobId={job.job_id}
            currentTime={curTime}
            speakerIds={safeSpeakers.map((s) => s.speaker_id)}
            onSeek={(t) => {
              if (videoRef.current) {
                videoRef.current.currentTime = t;
                videoRef.current.play().catch(() => void 0);
              }
            }}
            onArtifacts={() => setMediaBust((n) => n + 1)}
          />
        )}

        {err && (
          <div className="global-error-toast">
            <AlertCircle size={16} />
            <span>{err}</span>
          </div>
        )}
      </main>

      <ModelManagerModal
        isOpen={showModelManager}
        onClose={() => setShowModelManager(false)}
        onModelsChanged={refreshStorage}
      />
    </div>
  );
}

function healthSummary(h: Record<string, unknown>): string {
  if (!h || !Object.keys(h).length) return "Connecting…";
  const parts: string[] = [];
  if (h.gpu) parts.push("GPU Active");
  if (h.asr) parts.push(`ASR: ${h.asr}`);
  if (h.tts) parts.push(`TTS: ${h.tts}`);
  if (h.ffmpeg) parts.push("FFmpeg OK");
  return parts.length ? parts.join(" · ") : "Online";
}
