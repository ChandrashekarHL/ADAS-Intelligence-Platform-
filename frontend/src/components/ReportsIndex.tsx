"use client";

import Link from "next/link";

import { api } from "@/lib/api";
import { useAsync } from "@/lib/use-async";

import { Badge, Card, Empty, ErrorBox, Id, Spinner, Td, Th } from "./ui";

/** Approval queue: every report, pending ones first. */
export function ReportsIndex() {
  const approvals = useAsync(() => api.listApprovals(), []);
  const sorted = [...(approvals.data ?? [])].sort((a, b) =>
    a.status === b.status ? 0 : a.status === "pending_review" ? -1 : 1,
  );
  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Reports and approval queue</h1>
      <Card>
        {approvals.loading && <Spinner />}
        <ErrorBox message={approvals.error} />
        {approvals.data?.length === 0 && <Empty>No reports yet.</Empty>}
        {sorted.length > 0 && (
          <table className="w-full text-sm">
            <thead>
              <tr>
                <Th>Report</Th>
                <Th>Status</Th>
                <Th>Review</Th>
                <Th>Reviewer</Th>
                <Th>Decision</Th>
              </tr>
            </thead>
            <tbody>
              {sorted.map((a) => (
                <tr key={a.id} className="border-t border-line">
                  <Td>
                    <Link href={`/reports/${a.report_id}`} className="text-accent hover:underline">
                      <Id>{a.report_id}</Id>
                    </Link>
                  </Td>
                  <Td>
                    <Badge tone={a.status}>{a.status}</Badge>
                  </Td>
                  <Td>{a.human_review_required ? a.review_reasons.join("; ") || "required" : <span className="text-muted">not triggered</span>}</Td>
                  <Td>{a.reviewer ?? <span className="text-muted">—</span>}</Td>
                  <Td>{a.decision ? `${a.decision}: ${a.reason ?? ""}` : <span className="text-muted">pending</span>}</Td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Card>
    </div>
  );
}
