"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

import { api } from "@/lib/api";
import { useAsync } from "@/lib/use-async";

import { Badge } from "./ui";

export function Nav() {
  const path = usePathname();
  const health = useAsync(() => api.health(), []);
  const link = (href: string, label: string) => (
    <Link
      href={href}
      className={`rounded-md px-3 py-1.5 text-sm ${
        path === href || (href !== "/" && path.startsWith(href))
          ? "bg-line/40 font-semibold"
          : "text-muted hover:text-fg"
      }`}
    >
      {label}
    </Link>
  );
  return (
    <nav className="sticky top-0 z-10 border-b border-line bg-panel/90 backdrop-blur">
      <div className="mx-auto flex max-w-7xl items-center gap-2 px-4 py-2">
        <Link href="/" className="mr-4 flex items-center gap-2 font-semibold">
          <span className="inline-block h-2.5 w-2.5 rounded-full bg-accent" />
          AIP <span className="hidden text-muted sm:inline">· ADAS Intelligence Platform</span>
        </Link>
        {link("/", "Overview")}
        {link("/incidents", "Incidents")}
        {link("/reports", "Reports")}
        <div className="ml-auto flex items-center gap-2 text-xs text-muted">
          {health.error && <Badge tone="blocked">API offline</Badge>}
          {health.data && (
            <>
              <Badge tone={health.data.llm_provider === "fake" ? "degraded" : "pass"}>
                LLM: {health.data.llm_provider}
              </Badge>
              <Badge tone={health.data.rag_index_loaded ? "pass" : "degraded"}>
                RAG {health.data.rag_index_loaded ? "loaded" : "missing"}
              </Badge>
            </>
          )}
        </div>
      </div>
    </nav>
  );
}
