const base = "/api/v1";

export async function createJob(file: File, targetLanguage: string) {
  const body = new FormData();
  body.append("file", file);
  body.append("target_language", targetLanguage);
  const res = await fetch(`${base}/jobs`, { method: "POST", body });
  if (!res.ok) throw new Error(await res.text());
  return res.json() as Promise<{ job_id: string; state: string }>;
}

export async function getJob(id: string) {
  const res = await fetch(`${base}/jobs/${id}`);
  if (!res.ok) throw new Error("job not found");
  return res.json();
}

export async function health() {
  const res = await fetch(`${base}/health`);
  return res.json();
}
