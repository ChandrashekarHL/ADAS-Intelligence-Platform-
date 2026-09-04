/**
 * Typed client for the AIP backend (spec §18). Shapes mirror backend/app/api/schemas.py.
 * The dashboard never computes anything itself: every number on screen came from the API.
 */

export const API_BASE =
  process.env.NEXT_PUBLIC_API_URL?.replace(/\/$/, "") ?? "http://127.0.0.1:8000";

export class ApiError extends Error {
  constructor(
    public status: number,
    public detail: string,
  ) {
    super(`${status}: ${detail}`);
  }
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE}${path}`, {
    ...init,
    headers: {
      ...(init?.body instanceof FormData ? {} : { "Content-Type": "application/json" }),
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const body = (await res.json()) as { detail?: unknown };
      if (typeof body.detail === "string") detail = body.detail;
      else if (body.detail) detail = JSON.stringify(body.detail);
    } catch {
      /* non-JSON error body */
    }
    throw new ApiError(res.status, detail);
  }
  return (await res.json()) as T;
}

async function requestText(path: string): Promise<string> {
  const res = await fetch(`${API_BASE}${path}`, { cache: "no-store" });
  if (!res.ok) throw new ApiError(res.status, res.statusText);
  return res.text();
}

// --- types --------------------------------------------------------------------------------

export type Health = {
  status: string;
  version: string;
  llm_provider: string;
  rag_index_loaded: boolean;
};

export type Project = { id: string; name: string; created_at: string };

export type LogFile = {
  id: string;
  project_id: string;
  original_name: string;
  data_origin: string;
  scenario_id: string | null;
  row_count: number;
  duration_s: number | null;
  quality_id: string | null;
  quality_verdict: "pass" | "degraded" | "blocked" | null;
  uploaded_at: string;
};

export type IngestionJob = {
  job_id: string;
  file_id: string;
  status: "completed" | "blocked";
  quality_verdict: string;
  events: number;
  metrics_available: number;
  metrics_missing: number;
  primary_window_id: string | null;
};

export type Event = {
  id: string;
  file_id: string;
  project_id: string;
  event_type: string;
  t_s: number;
  description: string;
};

export type Metric = {
  id: string;
  file_id: string;
  name: string;
  value: number | null;
  unit: string;
  passed: boolean | null;
  t_s: number | null;
  window_id: string | null;
};

export type Signals = {
  file_id: string;
  rows_total: number;
  step: number;
  columns: string[];
  data: Record<string, (number | null)[]>;
};

export type AccessLevel = "public" | "internal" | "restricted";

export type QueryResult = {
  answer: string;
  confidence: string;
  evidence_ids: string[];
  unsupported_claims: string[];
  recommended_next_tests: string[];
  human_review_required: boolean;
  run_id: string;
  verification_id: string;
  evidence_support_rate: number;
};

export type VerifiedHypothesis = {
  hypothesis: {
    cause: string;
    failure_class: string;
    evidence_ids: string[];
    confidence: number;
  };
  status: string;
  resolved_ids: string[];
  dropped_ids: string[];
  independent_sources: string[];
  agent_confidence: number;
  adjusted_confidence: number;
  confidence_label: string;
  notes: string[];
};

export type Verification = {
  verification_id: string;
  report_confidence: string;
  human_review_required: boolean;
  review_reasons: string[];
  hypotheses: VerifiedHypothesis[];
  stripped: { hypothesis: { cause: string }; reason: string }[];
  applied_rules: { rule: string; effect: string; detail: string; hypothesis_index: number | null }[];
  flagged_observations: string[];
  missing_evidence: string[];
  recommended_next_tests: string[];
  limitations: string[];
  evidence_support_rate: number;
  unsupported_claim_rate: number;
};

export type RunSummary = {
  id: string;
  project_id: string;
  file_id: string;
  agent: string;
  provider: string;
  model: string;
  question: string;
  verification_id: string;
  report_confidence: string;
  human_review_required: boolean;
  evidence_support_rate: number;
  prompt_tokens: number;
  completion_tokens: number;
  latency_s: number;
  created_at: string;
};

export type RunDetail = RunSummary & {
  run: {
    observations?: never;
    output: { observations: string[]; missing_evidence: string[]; recommended_next_tests: string[] };
    attempts: number;
    unresolved_ids: string[];
    injection_flags: { evidence_id: string; pattern: string; excerpt: string }[];
    offered_evidence_ids: string[];
  };
  verification: Verification;
};

export type Chunk = {
  chunk_id: string;
  document_title: string;
  heading: string;
  text: string;
  source_type: string;
  access_level: string;
  version: string | null;
  requirement_ids: string[];
};

export type ReportCreated = {
  report_id: string;
  project_id: string;
  file_id: string;
  run_id: string | null;
  report_confidence: string;
  approval_id: string;
  human_review_required: boolean;
  markdown_url: string;
  json_url: string;
};

export type ReportListItem = {
  report_id: string;
  project_id: string;
  file_id: string;
  run_id: string | null;
  report_confidence: string;
  created_at: string;
  approval_id: string | null;
  approval_status: string | null;
};

export type Approval = {
  id: string;
  project_id: string;
  report_id: string;
  status: string;
  human_review_required: boolean;
  review_reasons: string[];
  reviewer: string | null;
  decision: string | null;
  reason: string | null;
  decided_at: string | null;
};

export type Dashboard = {
  projects: number;
  files: number;
  agent_runs: number;
  reports: number;
  reports_by_confidence: Record<string, number>;
  approvals_pending: number;
  avg_evidence_support_rate: number | null;
  llm_prompt_tokens: number;
  llm_completion_tokens: number;
  llm_provider: string;
  rag_index_loaded: boolean;
};

// --- calls --------------------------------------------------------------------------------

export const api = {
  health: () => request<Health>("/api/health"),
  dashboard: () => request<Dashboard>("/api/dashboard/summary"),

  listProjects: () => request<Project[]>("/api/projects"),
  createProject: (name: string) =>
    request<Project>("/api/projects", { method: "POST", body: JSON.stringify({ name }) }),
  getProject: (id: string) => request<Project>(`/api/projects/${id}`),

  listFiles: (projectId: string) => request<LogFile[]>(`/api/projects/${projectId}/files`),
  getFile: (fileId: string) => request<LogFile>(`/api/files/${fileId}`),
  uploadFile: (projectId: string, telemetry: File, sidecar: File | null) => {
    const form = new FormData();
    form.append("telemetry", telemetry);
    if (sidecar) form.append("sidecar", sidecar);
    return request<LogFile>(`/api/projects/${projectId}/files`, { method: "POST", body: form });
  },
  runIngestion: (fileId: string) =>
    request<IngestionJob>("/api/ingestion/jobs", {
      method: "POST",
      body: JSON.stringify({ file_id: fileId }),
    }),

  listEvents: (fileId: string) => request<Event[]>(`/api/events?file_id=${fileId}`),
  listMetrics: (fileId: string) => request<Metric[]>(`/api/files/${fileId}/metrics`),
  signals: (fileId: string, maxPoints = 600) =>
    request<Signals>(`/api/files/${fileId}/signals?max_points=${maxPoints}`),

  query: (body: { project_id: string; file_id: string; question?: string; access_level: AccessLevel }) =>
    request<QueryResult>("/api/query", { method: "POST", body: JSON.stringify(body) }),
  listRuns: (fileId: string) => request<RunSummary[]>(`/api/runs?file_id=${fileId}`),
  getRun: (runId: string) => request<RunDetail>(`/api/runs/${runId}`),
  getChunk: (chunkId: string, access: AccessLevel) =>
    request<Chunk>(`/api/chunks/${chunkId}?access=${access}`),

  createReport: (body: {
    project_id: string;
    file_id: string;
    run_id: string | null;
    access_level: AccessLevel;
  }) => request<ReportCreated>("/api/reports", { method: "POST", body: JSON.stringify(body) }),
  listReports: (projectId: string) => request<ReportListItem[]>(`/api/reports?project_id=${projectId}`),
  reportMarkdown: (reportId: string) => requestText(`/api/reports/${reportId}?format=md`),
  reportJson: (reportId: string) => request<Record<string, unknown>>(`/api/reports/${reportId}`),

  listApprovals: (status?: string) =>
    request<Approval[]>(`/api/approvals${status ? `?status=${status}` : ""}`),
  getApproval: (id: string) => request<Approval>(`/api/approvals/${id}`),
  decide: (id: string, body: { reviewer: string; decision: "approved" | "rejected"; reason: string }) =>
    request<Approval>(`/api/approvals/${id}/decision`, { method: "POST", body: JSON.stringify(body) }),
};
