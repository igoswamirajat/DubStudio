const base = "/api/v1";

export type Job = {
  job_id: string;
  state: string;
  percent?: number;
  message?: string;
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

export async function health() {
  const res = await fetch(`${base}/health`);
  return res.json();
}

export async function createJob(
  file: File,
  targetLanguage: string,
  ttsEngine = "omnivoice",
  skipSeparation = false,
) {
  const body = new FormData();
  body.append("file", file);
  body.append("target_language", targetLanguage);
  body.append("tts_engine", ttsEngine);
  body.append("skip_separation", skipSeparation ? "true" : "false");
  const res = await fetch(`${base}/jobs`, { method: "POST", body });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ job_id: string; state: string; tts_engine?: string }>;
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
