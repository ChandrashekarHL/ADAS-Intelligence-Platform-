"use client";

import Link from "next/link";
import { useState } from "react";

import { api, type LogFile, type Project } from "@/lib/api";
import { describeError, useAsync } from "@/lib/use-async";

import { Badge, Button, Card, Empty, ErrorBox, Id, Spinner, Stat, Td, Th, pct } from "./ui";

export function ProjectOverview() {
  const dash = useAsync(() => api.dashboard(), []);
  const projects = useAsync(() => api.listProjects(), []);
  const [selected, setSelected] = useState<string | null>(null);
  const projectId = selected ?? projects.data?.[0]?.id ?? null;

  return (
    <div className="space-y-6">
      <div>
        <h1 className="text-2xl font-semibold">Project overview</h1>
        <p className="mt-1 text-sm text-muted">
          Upload an AEB log, run the quality gates and metrics, then open the incident to ask the
          diagnostic agent. Every number here came from the API; nothing is computed in the browser.
        </p>
      </div>

      <ErrorBox message={dash.error} />
      {dash.data && (
        <div className="grid grid-cols-2 gap-3 md:grid-cols-4 lg:grid-cols-6">
          <Stat label="Projects" value={dash.data.projects} />
          <Stat label="Files" value={dash.data.files} />
          <Stat label="Agent runs" value={dash.data.agent_runs} />
          <Stat label="Reports" value={dash.data.reports} hint={confidenceMix(dash.data.reports_by_confidence)} />
          <Stat label="Pending approvals" value={dash.data.approvals_pending} />
          <Stat
            label="Evidence support"
            value={pct(dash.data.avg_evidence_support_rate)}
            hint={`${dash.data.llm_prompt_tokens + dash.data.llm_completion_tokens} LLM tokens`}
          />
        </div>
      )}

      <div className="grid gap-6 lg:grid-cols-[280px_1fr]">
        <Card title="Projects" actions={<NewProject onCreated={projects.reload} />}>
          {projects.loading && <Spinner />}
          <ErrorBox message={projects.error} />
          {projects.data?.length === 0 && <Empty>No projects yet. Create one to start.</Empty>}
          <ul className="space-y-1">
            {projects.data?.map((p: Project) => (
              <li key={p.id}>
                <button
                  onClick={() => setSelected(p.id)}
                  className={`w-full rounded-md px-2 py-1.5 text-left text-sm hover:bg-line/30 ${
                    p.id === projectId ? "bg-line/40 font-medium" : ""
                  }`}
                >
                  {p.name}
                  <div className="font-mono text-[11px] text-muted">{p.id}</div>
                </button>
              </li>
            ))}
          </ul>
        </Card>

        <div className="space-y-6">
          {projectId ? <FilesPanel projectId={projectId} /> : <Card title="Files"><Empty>Select or create a project.</Empty></Card>}
          {projectId && <ReportsPanel projectId={projectId} />}
        </div>
      </div>
    </div>
  );
}

function confidenceMix(mix: Record<string, number>): string {
  const parts = Object.entries(mix).map(([k, v]) => `${v} ${k}`);
  return parts.length ? parts.join(" · ") : "none yet";
}

function NewProject({ onCreated }: { onCreated: () => void }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async () => {
    if (!name.trim()) return;
    setBusy(true);
    setError(null);
    try {
      await api.createProject(name.trim());
      setName("");
      onCreated();
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <form
      className="flex items-center gap-1"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <input
        value={name}
        onChange={(e) => setName(e.target.value)}
        placeholder="New project"
        className="w-28 rounded-md border border-line bg-bg px-2 py-1 text-xs"
      />
      <Button type="submit" disabled={busy || !name.trim()}>+</Button>
      {error && <span className="text-xs text-bad">{error}</span>}
    </form>
  );
}

function FilesPanel({ projectId }: { projectId: string }) {
  const files = useAsync(() => api.listFiles(projectId), [projectId]);
  return (
    <Card title="Telemetry files" actions={<Upload projectId={projectId} onDone={files.reload} />}>
      {files.loading && <Spinner />}
      <ErrorBox message={files.error} />
      {files.data?.length === 0 && (
        <Empty>
          No files. Upload <code>telemetry.csv</code> (and its <code>scenario.json</code>) from{" "}
          <code>data/demo/aeb_late_braking_seed42</code>.
        </Empty>
      )}
      {files.data && files.data.length > 0 && (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr>
                <Th>File</Th>
                <Th>Origin</Th>
                <Th>Rows</Th>
                <Th>Duration</Th>
                <Th>Quality</Th>
                <Th>Actions</Th>
              </tr>
            </thead>
            <tbody>
              {files.data.map((f) => (
                <FileRow key={f.id} file={f} onChanged={files.reload} />
              ))}
            </tbody>
          </table>
        </div>
      )}
    </Card>
  );
}

function FileRow({ file, onChanged }: { file: LogFile; onChanged: () => void }) {
  const [busy, setBusy] = useState(false);
  const [msg, setMsg] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    setMsg(null);
    try {
      const job = await api.runIngestion(file.id);
      setMsg(
        job.status === "completed"
          ? `${job.events} events, ${job.metrics_available} metrics`
          : `blocked (${job.quality_verdict})`,
      );
      onChanged();
    } catch (e) {
      setMsg(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <tr className="border-t border-line">
      <Td>
        <div>{file.original_name}</div>
        <Id>{file.id}</Id>
      </Td>
      <Td>
        <Badge tone={file.data_origin === "synthetic" ? "degraded" : undefined}>{file.data_origin}</Badge>
      </Td>
      <Td>{file.row_count}</Td>
      <Td>{file.duration_s?.toFixed(2) ?? "—"} s</Td>
      <Td>
        <Badge tone={file.quality_verdict}>{file.quality_verdict ?? "?"}</Badge>
      </Td>
      <Td>
        <div className="flex flex-wrap items-center gap-2">
          <Button kind="ghost" onClick={() => void run()} disabled={busy || file.quality_verdict === "blocked"}>
            {busy ? "Running…" : "Run gates + metrics"}
          </Button>
          <Link href={`/incidents/${file.id}`} className="text-sm text-accent hover:underline">
            Open incident →
          </Link>
          {msg && <span className="text-xs text-muted">{msg}</span>}
        </div>
      </Td>
    </tr>
  );
}

function Upload({ projectId, onDone }: { projectId: string; onDone: () => void }) {
  const [csv, setCsv] = useState<File | null>(null);
  const [sidecar, setSidecar] = useState<File | null>(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const submit = async () => {
    if (!csv) return;
    setBusy(true);
    setError(null);
    try {
      await api.uploadFile(projectId, csv, sidecar);
      setCsv(null);
      setSidecar(null);
      onDone();
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <form
      className="flex flex-wrap items-center gap-2 text-xs"
      onSubmit={(e) => {
        e.preventDefault();
        void submit();
      }}
    >
      <label className="flex items-center gap-1">
        <span className="text-muted">CSV</span>
        <input type="file" accept=".csv" onChange={(e) => setCsv(e.target.files?.[0] ?? null)} />
      </label>
      <label className="flex items-center gap-1">
        <span className="text-muted">sidecar</span>
        <input type="file" accept=".json" onChange={(e) => setSidecar(e.target.files?.[0] ?? null)} />
      </label>
      <Button type="submit" disabled={!csv || busy}>
        {busy ? "Uploading…" : "Upload"}
      </Button>
      {error && <span className="text-bad">{error}</span>}
    </form>
  );
}

function ReportsPanel({ projectId }: { projectId: string }) {
  const reports = useAsync(() => api.listReports(projectId), [projectId]);
  return (
    <Card title="Reports and approvals">
      {reports.loading && <Spinner />}
      <ErrorBox message={reports.error} />
      {reports.data?.length === 0 && <Empty>No reports yet. Create one from an incident.</Empty>}
      {reports.data && reports.data.length > 0 && (
        <table className="w-full text-sm">
          <thead>
            <tr>
              <Th>Report</Th>
              <Th>Confidence</Th>
              <Th>Diagnosis</Th>
              <Th>Approval</Th>
              <Th>Created</Th>
            </tr>
          </thead>
          <tbody>
            {reports.data.map((r) => (
              <tr key={r.report_id} className="border-t border-line">
                <Td>
                  <Link href={`/reports/${r.report_id}`} className="text-accent hover:underline">
                    <Id>{r.report_id}</Id>
                  </Link>
                </Td>
                <Td>
                  <Badge tone={r.report_confidence}>{r.report_confidence}</Badge>
                </Td>
                <Td>{r.run_id ? <Id>{r.run_id}</Id> : <span className="text-muted">metrics-only</span>}</Td>
                <Td>
                  <Badge tone={r.approval_status}>{r.approval_status ?? "—"}</Badge>
                </Td>
                <Td>{new Date(r.created_at).toLocaleString()}</Td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </Card>
  );
}
