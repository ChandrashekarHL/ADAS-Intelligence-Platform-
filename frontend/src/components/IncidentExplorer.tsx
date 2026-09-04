"use client";

import Link from "next/link";
import { useMemo, useState } from "react";

import {
  api,
  type AccessLevel,
  type Chunk,
  type Event,
  type Metric,
  type RunDetail,
  type Signals,
} from "@/lib/api";
import { describeError, useAsync } from "@/lib/use-async";

import { SignalCharts } from "./SignalCharts";
import { Badge, Button, Card, Empty, ErrorBox, Id, PassFail, Spinner, Td, Th, fmt, pct } from "./ui";

export function IncidentExplorer({ fileId }: { fileId: string }) {
  const file = useAsync(() => api.getFile(fileId), [fileId]);
  const events = useAsync(() => api.listEvents(fileId), [fileId]);
  const metrics = useAsync(() => api.listMetrics(fileId), [fileId]);
  const signals = useAsync(() => api.signals(fileId), [fileId]);
  const runs = useAsync(() => api.listRuns(fileId), [fileId]);
  const [activeRun, setActiveRun] = useState<string | null>(null);
  const runId = activeRun ?? runs.data?.[0]?.id ?? null;

  const metricById = useMemo(() => new Map((metrics.data ?? []).map((m) => [m.id, m])), [metrics.data]);
  const eventById = useMemo(() => new Map((events.data ?? []).map((e) => [e.id, e])), [events.data]);

  if (file.error) return <ErrorBox message={file.error} />;
  if (!file.data) return <Spinner label="Loading file" />;
  const f = file.data;
  const needsJob = metrics.data !== null && metrics.data.length === 0;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-2xl font-semibold">{f.original_name}</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted">
            <Id>{f.id}</Id>
            {f.scenario_id && <Id>{f.scenario_id}</Id>}
            <Badge tone={f.data_origin === "synthetic" ? "degraded" : undefined}>{f.data_origin}</Badge>
            <Badge tone={f.quality_verdict}>quality: {f.quality_verdict}</Badge>
            <span>
              {f.row_count} rows · {f.duration_s?.toFixed(2)} s
            </span>
          </div>
        </div>
        <Link href="/" className="ml-auto text-sm text-accent hover:underline">
          ← Overview
        </Link>
      </header>

      {f.data_origin === "synthetic" && (
        <p className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-sm text-warn">
          Synthetic simulation data. Nothing on this page describes real-world vehicle behaviour.
        </p>
      )}

      {needsJob && (
        <RunJob
          fileId={fileId}
          onDone={() => {
            metrics.reload();
            events.reload();
          }}
        />
      )}

      <Card title="Signals and events">
        {signals.loading && <Spinner label="Loading signals" />}
        <ErrorBox message={signals.error} />
        {signals.data && <SignalCharts signals={signals.data as Signals} events={events.data ?? []} />}
      </Card>

      <div className="grid gap-6 lg:grid-cols-2">
        <Card title="Event timeline">
          {events.data?.length === 0 && <Empty>No events stored yet. Run gates + metrics first.</Empty>}
          <ol className="space-y-2">
            {(events.data ?? []).map((e: Event) => (
              <li key={e.id} className="flex items-start gap-3 text-sm">
                <span className="w-16 shrink-0 font-mono text-xs tabular-nums text-muted">{e.t_s.toFixed(2)} s</span>
                <div>
                  <div className="font-medium">{e.event_type.replaceAll("_", " ")}</div>
                  <div className="text-xs text-muted">{e.description}</div>
                  <Id>{e.id}</Id>
                </div>
              </li>
            ))}
          </ol>
        </Card>

        <Card title="Metrics">
          <ErrorBox message={metrics.error} />
          {metrics.data?.length === 0 && <Empty>No metrics stored yet.</Empty>}
          {metrics.data && metrics.data.length > 0 && (
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr>
                    <Th>Metric</Th>
                    <Th>Value</Th>
                    <Th>t [s]</Th>
                    <Th>Result</Th>
                  </tr>
                </thead>
                <tbody>
                  {metrics.data.map((m: Metric) => (
                    <tr key={m.id} className="border-t border-line">
                      <Td>
                        <div>{m.name}</div>
                        <Id>{m.id}</Id>
                      </Td>
                      <Td mono>
                        {fmt(m.value)} {m.unit}
                      </Td>
                      <Td mono>{m.t_s === null ? "" : m.t_s.toFixed(2)}</Td>
                      <Td>
                        <PassFail passed={m.passed} />
                      </Td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          )}
        </Card>
      </div>

      <Diagnosis
        projectId={f.project_id}
        fileId={fileId}
        runs={runs}
        runId={runId}
        onRunCreated={(id) => {
          setActiveRun(id);
          runs.reload();
        }}
        metricById={metricById}
        eventById={eventById}
        blocked={f.quality_verdict === "blocked"}
      />
    </div>
  );
}

function RunJob({ fileId, onDone }: { fileId: string; onDone: () => void }) {
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const run = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.runIngestion(fileId);
      onDone();
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };
  return (
    <div className="flex flex-wrap items-center gap-3 rounded-md border border-line bg-panel px-3 py-2 text-sm">
      <span>Gates and metrics have not been computed for this file yet.</span>
      <Button onClick={() => void run()} disabled={busy}>
        {busy ? "Running…" : "Run gates + metrics"}
      </Button>
      <ErrorBox message={error} />
    </div>
  );
}

type Diag = {
  projectId: string;
  fileId: string;
  runs: ReturnType<typeof useAsync<import("@/lib/api").RunSummary[]>>;
  runId: string | null;
  onRunCreated: (id: string) => void;
  metricById: Map<string, Metric>;
  eventById: Map<string, Event>;
  blocked: boolean;
};

function Diagnosis({ projectId, fileId, runs, runId, onRunCreated, metricById, eventById, blocked }: Diag) {
  const [access, setAccess] = useState<AccessLevel>("internal");
  const [question, setQuestion] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const detail = useAsync(() => (runId ? api.getRun(runId) : Promise.resolve(null)), [runId]);

  const ask = async () => {
    setBusy(true);
    setError(null);
    try {
      const q = await api.query({
        project_id: projectId,
        file_id: fileId,
        question: question.trim() || undefined,
        access_level: access,
      });
      onRunCreated(q.run_id);
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card
      title="AI diagnosis (verified)"
      actions={
        runs.data && runs.data.length > 1 ? (
          <select
            value={runId ?? ""}
            onChange={(e) => onRunCreated(e.target.value)}
            className="rounded-md border border-line bg-bg px-2 py-1 text-xs"
          >
            {runs.data.map((r) => (
              <option key={r.id} value={r.id}>
                {new Date(r.created_at).toLocaleTimeString()} · {r.report_confidence} · {r.id}
              </option>
            ))}
          </select>
        ) : null
      }
    >
      <form
        className="mb-4 flex flex-wrap items-center gap-2"
        onSubmit={(e) => {
          e.preventDefault();
          void ask();
        }}
      >
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          placeholder="Why did the AEB brake late? (default question if empty)"
          className="min-w-64 flex-1 rounded-md border border-line bg-bg px-2 py-1.5 text-sm"
        />
        <label className="flex items-center gap-1 text-xs text-muted">
          access
          <select
            value={access}
            onChange={(e) => setAccess(e.target.value as AccessLevel)}
            className="rounded-md border border-line bg-bg px-2 py-1 text-xs"
          >
            <option value="public">public</option>
            <option value="internal">internal</option>
            <option value="restricted">restricted</option>
          </select>
        </label>
        <Button type="submit" disabled={busy || blocked}>
          {busy ? "Diagnosing…" : "Ask the agent"}
        </Button>
        {blocked && <span className="text-xs text-bad">Blocked by data-quality gates.</span>}
      </form>
      <ErrorBox message={error} />

      {detail.loading && runId && <Spinner label="Loading run" />}
      {!runId && !busy && <Empty>No diagnosis yet. Ask the agent; the answer is verified before it is shown.</Empty>}
      {detail.data && (
        <VerifiedRun
          run={detail.data}
          access={access}
          metricById={metricById}
          eventById={eventById}
          projectId={projectId}
          fileId={fileId}
        />
      )}
    </Card>
  );
}

function VerifiedRun({
  run,
  access,
  metricById,
  eventById,
  projectId,
  fileId,
}: {
  run: RunDetail;
  access: AccessLevel;
  metricById: Map<string, Metric>;
  eventById: Map<string, Event>;
  projectId: string;
  fileId: string;
}) {
  const v = run.verification;
  const [selected, setSelected] = useState<string | null>(null);
  const [report, setReport] = useState<string | null>(null);
  const [reportErr, setReportErr] = useState<string | null>(null);
  const [busy, setBusy] = useState(false);

  const createReport = async () => {
    setBusy(true);
    setReportErr(null);
    try {
      const r = await api.createReport({ project_id: projectId, file_id: fileId, run_id: run.id, access_level: access });
      setReport(r.report_id);
    } catch (e) {
      setReportErr(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <div className="space-y-4">
      <div className="flex flex-wrap items-center gap-2 text-sm">
        <Badge tone={v.report_confidence}>confidence: {v.report_confidence}</Badge>
        <Badge tone={v.human_review_required ? "degraded" : "pass"}>
          {v.human_review_required ? "human review required" : "no review trigger"}
        </Badge>
        <span className="text-muted">
          evidence support {pct(v.evidence_support_rate)} · unsupported claims {pct(v.unsupported_claim_rate)} ·{" "}
          {run.provider}/{run.model} · {run.run.attempts} attempt(s) · {run.prompt_tokens + run.completion_tokens} tokens
        </span>
        <div className="ml-auto flex items-center gap-2">
          {report ? (
            <Link href={`/reports/${report}`} className="text-sm text-accent hover:underline">
              Open report →
            </Link>
          ) : (
            <Button onClick={() => void createReport()} disabled={busy}>
              {busy ? "Building…" : "Create report"}
            </Button>
          )}
        </div>
      </div>
      <ErrorBox message={reportErr} />
      {v.review_reasons.map((r) => (
        <p key={r} className="rounded-md border border-warn/40 bg-warn/10 px-3 py-1.5 text-sm text-warn">
          Review: {r}
        </p>
      ))}

      <div className="grid gap-4 lg:grid-cols-[1fr_360px]">
        <div className="space-y-3">
          {run.run.output.observations.length > 0 && (
            <div>
              <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Observations</h3>
              <ul className="mt-1 list-disc pl-5 text-sm">
                {run.run.output.observations.map((o) => (
                  <li key={o} className={v.flagged_observations.includes(o) ? "text-bad" : ""}>
                    {o}
                    {v.flagged_observations.includes(o) && " (flagged: unknown evidence id)"}
                  </li>
                ))}
              </ul>
            </div>
          )}
          <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Hypotheses (verified, ranked)</h3>
          {v.hypotheses.length === 0 && <Empty>No hypothesis survived verification.</Empty>}
          {v.hypotheses.map((h, i) => (
            <div key={i} className="rounded-md border border-line p-3">
              <div className="flex flex-wrap items-center gap-2">
                <span className="font-semibold">{i + 1}.</span>
                <Badge>{h.hypothesis.failure_class}</Badge>
                <Badge tone={h.confidence_label}>{h.confidence_label}</Badge>
                <span className="text-xs text-muted">
                  agent {h.agent_confidence.toFixed(2)} → adjusted {h.adjusted_confidence.toFixed(2)} · {h.status}
                </span>
              </div>
              <p className="mt-2 text-sm">{h.hypothesis.cause}</p>
              <div className="mt-2 flex flex-wrap gap-1">
                {h.resolved_ids.map((id) => (
                  <button
                    key={id}
                    onClick={() => setSelected(id)}
                    className={`rounded border px-1.5 py-0.5 font-mono text-[11px] hover:border-accent ${
                      selected === id ? "border-accent bg-accent/10" : "border-line"
                    }`}
                  >
                    {id}
                  </button>
                ))}
                {h.dropped_ids.map((id) => (
                  <span key={id} className="rounded border border-bad/40 px-1.5 py-0.5 font-mono text-[11px] text-bad line-through">
                    {id}
                  </span>
                ))}
              </div>
              <div className="mt-1 text-xs text-muted">
                sources: {h.independent_sources.join(", ") || "—"}
                {h.notes.map((n) => ` · ${n}`)}
              </div>
            </div>
          ))}
          {v.stripped.length > 0 && (
            <div className="rounded-md border border-bad/30 bg-bad/5 p-3 text-sm">
              <div className="text-xs font-semibold uppercase tracking-wide text-bad">Removed by the verifier</div>
              <ul className="mt-1 list-disc pl-5">
                {v.stripped.map((s, i) => (
                  <li key={i}>
                    {s.hypothesis.cause} <span className="text-muted">({s.reason})</span>
                  </li>
                ))}
              </ul>
            </div>
          )}
          {(v.missing_evidence.length > 0 || v.recommended_next_tests.length > 0) && (
            <div className="grid gap-3 sm:grid-cols-2 text-sm">
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Missing evidence</h3>
                <ul className="mt-1 list-disc pl-5">
                  {v.missing_evidence.map((m) => (
                    <li key={m}>{m}</li>
                  ))}
                  {v.missing_evidence.length === 0 && <li className="text-muted">none</li>}
                </ul>
              </div>
              <div>
                <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Recommended next tests</h3>
                <ul className="mt-1 list-disc pl-5">
                  {v.recommended_next_tests.map((t) => (
                    <li key={t}>{t}</li>
                  ))}
                </ul>
              </div>
            </div>
          )}
        </div>

        <EvidencePanel id={selected} access={access} metricById={metricById} eventById={eventById} />
      </div>
    </div>
  );
}

function EvidencePanel({
  id,
  access,
  metricById,
  eventById,
}: {
  id: string | null;
  access: AccessLevel;
  metricById: Map<string, Metric>;
  eventById: Map<string, Event>;
}) {
  const isChunk = id?.startsWith("chunk_") ?? false;
  const chunk = useAsync<Chunk | null>(
    () => (id && isChunk ? api.getChunk(id, access) : Promise.resolve(null)),
    [id, access],
  );
  return (
    <aside className="rounded-md border border-line bg-bg p-3 text-sm">
      <h3 className="text-xs font-semibold uppercase tracking-wide text-muted">Evidence</h3>
      {!id && <p className="mt-2 text-muted">Click an evidence ID on a hypothesis to see what it resolves to.</p>}
      {id && <div className="mt-2"><Id>{id}</Id></div>}
      {id && metricById.has(id) && <MetricDetail m={metricById.get(id)!} />}
      {id && eventById.has(id) && (
        <div className="mt-2">
          <div className="font-medium">{eventById.get(id)!.event_type}</div>
          <div className="text-muted">
            t = {eventById.get(id)!.t_s.toFixed(2)} s · {eventById.get(id)!.description}
          </div>
        </div>
      )}
      {id && isChunk && chunk.loading && <Spinner label="Resolving chunk" />}
      {id && isChunk && chunk.error && <ErrorBox message={chunk.error} />}
      {chunk.data && (
        <div className="mt-2 space-y-1">
          <div className="font-medium">
            {chunk.data.document_title} <span className="text-muted">v{chunk.data.version ?? "?"}</span>
          </div>
          <div className="flex flex-wrap gap-1">
            <Badge>{chunk.data.source_type}</Badge>
            <Badge tone={chunk.data.access_level === "restricted" ? "blocked" : undefined}>{chunk.data.access_level}</Badge>
            {chunk.data.requirement_ids.map((r) => (
              <Badge key={r}>{r}</Badge>
            ))}
          </div>
          <pre className="mt-2 max-h-72 overflow-auto whitespace-pre-wrap rounded bg-panel p-2 font-mono text-[11px] leading-relaxed">
            {chunk.data.text}
          </pre>
        </div>
      )}
      {id && !metricById.has(id) && !eventById.has(id) && !isChunk && (
        <p className="mt-2 text-muted">
          This ID is a window, quality or file artifact. It resolves in the report&apos;s evidence appendix.
        </p>
      )}
    </aside>
  );
}

function MetricDetail({ m }: { m: Metric }) {
  return (
    <div className="mt-2">
      <div className="font-medium">{m.name}</div>
      <div className="font-mono text-xs">
        {fmt(m.value)} {m.unit}
        {m.t_s !== null && ` @ ${m.t_s.toFixed(2)} s`}
      </div>
      <div className="mt-1">
        <PassFail passed={m.passed} />
      </div>
      {m.window_id && (
        <div className="mt-1 text-xs text-muted">
          window <Id>{m.window_id}</Id>
        </div>
      )}
    </div>
  );
}
