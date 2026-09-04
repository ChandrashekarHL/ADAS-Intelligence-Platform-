"use client";

import Link from "next/link";

import { api, type LogFile } from "@/lib/api";
import { useAsync } from "@/lib/use-async";

import { Badge, Card, Empty, ErrorBox, Id, Spinner } from "./ui";

/** All files across projects, as entry points into the Incident Explorer. */
export function IncidentIndex() {
  const files = useAsync(async () => {
    const projects = await api.listProjects();
    const lists = await Promise.all(projects.map((p) => api.listFiles(p.id)));
    return projects.map((p, i) => ({ project: p, files: lists[i] }));
  }, []);

  return (
    <div className="space-y-4">
      <h1 className="text-2xl font-semibold">Incidents</h1>
      {files.loading && <Spinner />}
      <ErrorBox message={files.error} />
      {files.data?.every((g) => g.files.length === 0) && <Empty>No telemetry files uploaded yet.</Empty>}
      {files.data?.map(
        (g) =>
          g.files.length > 0 && (
            <Card key={g.project.id} title={g.project.name}>
              <ul className="divide-y divide-line">
                {g.files.map((f: LogFile) => (
                  <li key={f.id} className="flex flex-wrap items-center gap-3 py-2 text-sm">
                    <Link href={`/incidents/${f.id}`} className="text-accent hover:underline">
                      {f.original_name}
                    </Link>
                    <Id>{f.id}</Id>
                    <Badge tone={f.quality_verdict}>{f.quality_verdict ?? "?"}</Badge>
                    <span className="text-muted">
                      {f.row_count} rows · {f.duration_s?.toFixed(2) ?? "—"} s · {f.data_origin}
                    </span>
                  </li>
                ))}
              </ul>
            </Card>
          ),
      )}
    </div>
  );
}
