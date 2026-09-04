"use client";

import Link from "next/link";
import { useState } from "react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";

import { api, type Approval } from "@/lib/api";
import { describeError, useAsync } from "@/lib/use-async";

import { Badge, Button, Card, ErrorBox, Id, Spinner } from "./ui";

export function ReportView({ reportId }: { reportId: string }) {
  const md = useAsync(() => api.reportMarkdown(reportId), [reportId]);
  const json = useAsync(() => api.reportJson(reportId), [reportId]);
  const approvals = useAsync(() => api.listApprovals(), [reportId]);
  const approval = approvals.data?.find((a) => a.report_id === reportId) ?? null;
  const meta = (json.data?.metadata ?? {}) as Record<string, string | null>;
  const confidence = (json.data?.report_confidence as string | undefined) ?? null;

  return (
    <div className="space-y-6">
      <header className="flex flex-wrap items-center gap-3">
        <div>
          <h1 className="text-2xl font-semibold">Diagnostic report</h1>
          <div className="mt-1 flex flex-wrap items-center gap-2 text-sm text-muted">
            <Id>{reportId}</Id>
            {confidence && <Badge tone={confidence}>confidence: {confidence}</Badge>}
            {meta.run_id ? <Id>{meta.run_id}</Id> : <Badge>metrics-only</Badge>}
            {meta.file_id && (
              <Link href={`/incidents/${meta.file_id}`} className="text-accent hover:underline">
                incident →
              </Link>
            )}
          </div>
        </div>
        <Link href="/reports" className="ml-auto text-sm text-accent hover:underline">
          ← Approval queue
        </Link>
      </header>

      <div className="grid gap-6 lg:grid-cols-[1fr_340px]">
        <Card>
          {md.loading && <Spinner label="Loading report" />}
          <ErrorBox message={md.error} />
          {md.data && (
            <article className="report-md">
              <ReactMarkdown remarkPlugins={[remarkGfm]}>{md.data}</ReactMarkdown>
            </article>
          )}
        </Card>
        <div className="space-y-4">
          {approval ? (
            <ApprovalPanel
              approval={approval}
              onDecided={() => {
                approvals.reload();
                md.reload();
                json.reload();
              }}
            />
          ) : (
            <Card title="Approval">{approvals.loading ? <Spinner /> : <p className="text-sm text-muted">No approval task found.</p>}</Card>
          )}
        </div>
      </div>
    </div>
  );
}

function ApprovalPanel({ approval, onDecided }: { approval: Approval; onDecided: () => void }) {
  const [reviewer, setReviewer] = useState("");
  const [decision, setDecision] = useState<"approved" | "rejected">("approved");
  const [reason, setReason] = useState("");
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const pending = approval.status === "pending_review";

  const submit = async () => {
    setBusy(true);
    setError(null);
    try {
      await api.decide(approval.id, { reviewer: reviewer.trim(), decision, reason: reason.trim() });
      onDecided();
    } catch (e) {
      setError(describeError(e));
    } finally {
      setBusy(false);
    }
  };

  return (
    <Card title="Approval">
      <div className="space-y-2 text-sm">
        <div className="flex items-center gap-2">
          <Badge tone={approval.status}>{approval.status}</Badge>
          <Id>{approval.id}</Id>
        </div>
        {approval.human_review_required ? (
          <div className="rounded-md border border-warn/40 bg-warn/10 px-3 py-2 text-warn">
            Human review required: {approval.review_reasons.join("; ")}
          </div>
        ) : (
          <p className="text-muted">No automatic review trigger. Sign-off is still required before use.</p>
        )}
        {!pending && (
          <dl className="grid grid-cols-[auto_1fr] gap-x-3 gap-y-1">
            <dt className="text-muted">Reviewer</dt>
            <dd>{approval.reviewer}</dd>
            <dt className="text-muted">Decision</dt>
            <dd>{approval.decision}</dd>
            <dt className="text-muted">Reason</dt>
            <dd>{approval.reason}</dd>
            <dt className="text-muted">When</dt>
            <dd>{approval.decided_at ? new Date(approval.decided_at).toLocaleString() : "—"}</dd>
          </dl>
        )}
        {pending && (
          <form
            className="space-y-2"
            onSubmit={(e) => {
              e.preventDefault();
              void submit();
            }}
          >
            <input
              value={reviewer}
              onChange={(e) => setReviewer(e.target.value)}
              placeholder="Reviewer name"
              className="w-full rounded-md border border-line bg-bg px-2 py-1.5"
            />
            <select
              value={decision}
              onChange={(e) => setDecision(e.target.value as "approved" | "rejected")}
              className="w-full rounded-md border border-line bg-bg px-2 py-1.5"
            >
              <option value="approved">approve</option>
              <option value="rejected">reject</option>
            </select>
            <textarea
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder="Reason (recorded in the report)"
              rows={3}
              className="w-full rounded-md border border-line bg-bg px-2 py-1.5"
            />
            <Button type="submit" kind={decision === "approved" ? "primary" : "danger"} disabled={busy || !reviewer.trim() || !reason.trim()}>
              {busy ? "Recording…" : `Record ${decision === "approved" ? "approval" : "rejection"}`}
            </Button>
            <ErrorBox message={error} />
          </form>
        )}
      </div>
    </Card>
  );
}
