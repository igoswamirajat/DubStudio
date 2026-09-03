const base = "/api/v1";

export type Job = {
  job_id: string;
  state: string;
  percent?: number;
  message?: string;
  stage_index?: number;
  stage_total?: number;
  target_language?: string;
  tts_engine?: string;
  speakers?: Speaker[];
  artifacts?: Record<string, string>;
  error?: string | null;
};

export type Speaker = {
  speaker_id: string;
  label?: string;
  voice_id?: string;
  voice_mode?: string;
  design_prompt?: string | null;
  ref_text?: string | null;
  ref_wav?: string;
  segment_count?: number;
};

export type Segment = {
  segment_id: string;
  speaker_id?: string;
  start: number;
  end: number;
  source_text?: string;
  translated_text?: string;
  status?: string;
  voice_mode?: string;
  generated_duration_ms?: number;
};

export async function health() {
  const res = await fetch(`${base}/health`);
  return res.json();
}

export async function createJob(
  file: File,
  targetLanguage: string,
  ttsEngine = "omnivoice",
  skipSeparation = false,
  sourceLanguage = "",
) {
  const body = new FormData();
  body.append("file", file);
  body.append("target_language", targetLanguage);
  body.append("tts_engine", ttsEngine);
  body.append("skip_separation", skipSeparation ? "true" : "false");
  if (sourceLanguage) body.append("source_language", sourceLanguage);
  const res = await fetch(`${base}/jobs`, { method: "POST", body });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ job_id: string; state: string; tts_engine?: string }>;
}

export async function listJobs(): Promise<{ jobs: Job[] }> {
  const res = await fetch(`${base}/jobs`);
  if (!res.ok) return { jobs: [] };
  return res.json();
}

export async function getJob(id: string): Promise<Job> {
  const res = await fetch(`${base}/jobs/${id}`);
  if (!res.ok) throw new Error("job not found");
  return res.json();
}

export async function listSpeakers(jobId: string): Promise<{ speakers: Speaker[] }> {
  const res = await fetch(`${base}/jobs/${jobId}/speakers`);
  if (!res.ok) return { speakers: [] };
  return res.json();
}

export async function patchSpeaker(
  jobId: string,
  speakerId: string,
  body: Partial<Speaker>,
): Promise<Speaker> {
  const res = await fetch(`${base}/jobs/${jobId}/speakers/${speakerId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export function speakerRefUrl(jobId: string, speakerId: string) {
  return `${base}/jobs/${jobId}/speakers/${speakerId}/ref`;
}

export function downloadUrl(jobId: string, artifact: "mp4" | "srt" | "json") {
  return `${base}/jobs/${jobId}/download?artifact=${artifact}`;
}

export function mediaUrl(jobId: string, kind: "source" | "output") {
  return `${base}/jobs/${jobId}/media/${kind}`;
}

export async function listSegments(jobId: string): Promise<{ segments: Segment[] }> {
  const res = await fetch(`${base}/jobs/${jobId}/segments`);
  if (!res.ok) return { segments: [] };
  return res.json();
}

export async function patchSegment(
  jobId: string,
  segmentId: string,
  body: Partial<Segment>,
): Promise<Segment> {
  const res = await fetch(`${base}/jobs/${jobId}/segments/${segmentId}`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function resynthSegment(
  jobId: string,
  segmentId: string,
  text?: string,
): Promise<{ segment: Segment; artifacts: Record<string, string> }> {
  const res = await fetch(`${base}/jobs/${jobId}/segments/${segmentId}/resynth`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(text ? { text } : {}),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function cancelJob(jobId: string): Promise<void> {
  await fetch(`${base}/jobs/${jobId}/cancel`, { method: "POST" });
}

export async function deleteJob(jobId: string): Promise<void> {
  const res = await fetch(`${base}/jobs/${jobId}`, { method: "DELETE" });
  if (!res.ok) throw new Error(await res.text());
}

export async function resumeJob(jobId: string): Promise<void> {
  const res = await fetch(`${base}/jobs/${jobId}/resume`, { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
}

/**
 * Subscribe to live job progress over SSE. Returns an unsubscribe function.
 * Falls back silently — callers should keep a polling safety net.
 */
export function subscribeJob(jobId: string, onJob: (job: Job) => void): () => void {
  const es = new EventSource(`${base}/jobs/${jobId}/events`);
  const handle = (ev: MessageEvent) => {
    if (!ev || typeof ev.data !== "string" || !ev.data) return;
    try {
      onJob(JSON.parse(ev.data) as Job);
    } catch {
      /* ignore malformed frame */
    }
  };
  // Named SSE events carry the job JSON. The backend also emits a named
  // "error" event (with data) for failed jobs; the guard above ignores the
  // browser's native bare "error" event fired on disconnect.
  es.addEventListener("progress", handle as EventListener);
  es.addEventListener("done", handle as EventListener);
  es.addEventListener("error", handle as EventListener);
  return () => es.close();
}

export type AppSettings = {
  translator: string;
  ollama_host: string;
  ollama_model: string;
  openai_base_url: string;
  openai_model: string;
  openai_api_key_set: boolean;
  tts_engine: string;
  asr_engine: string;
  whisper_model: string;
  enable_diarization: boolean;
  diarization_model: string;
  hf_token_set: boolean;
  max_speakers: number;
  skip_separation: boolean;
  low_vram: boolean;
};

export async function getSettings(): Promise<AppSettings> {
  const res = await fetch(`${base}/settings`);
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}

export async function patchSettings(body: Partial<Record<string, unknown>>): Promise<AppSettings> {
  const res = await fetch(`${base}/settings`, {
    method: "PATCH",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body),
  });
  if (!res.ok) throw new Error(await res.text());
  return res.json();
}
