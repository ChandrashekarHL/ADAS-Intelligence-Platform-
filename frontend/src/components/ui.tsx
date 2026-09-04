import type { ReactNode } from "react";

/** Small, dependency-free building blocks. Evidence IDs are always monospace. */

export function Card({
  title,
  children,
  actions,
  className = "",
}: {
  title?: ReactNode;
  children: ReactNode;
  actions?: ReactNode;
  className?: string;
}) {
  return (
    <section className={`rounded-lg border border-line bg-panel p-4 shadow-sm ${className}`}>
      {(title || actions) && (
        <header className="mb-3 flex items-center justify-between gap-3">
          {title && <h2 className="text-sm font-semibold tracking-wide text-muted uppercase">{title}</h2>}
          {actions}
        </header>
      )}
      {children}
    </section>
  );
}

export function Stat({ label, value, hint }: { label: string; value: ReactNode; hint?: string }) {
  return (
    <div className="rounded-lg border border-line bg-panel px-4 py-3">
      <div className="text-xs text-muted uppercase tracking-wide">{label}</div>
      <div className="mt-1 text-2xl font-semibold tabular-nums">{value}</div>
      {hint && <div className="mt-0.5 text-xs text-muted">{hint}</div>}
    </div>
  );
}

const TONES: Record<string, string> = {
  pass: "bg-ok/15 text-ok border-ok/30",
  high: "bg-ok/15 text-ok border-ok/30",
  approved: "bg-ok/15 text-ok border-ok/30",
  completed: "bg-ok/15 text-ok border-ok/30",
  degraded: "bg-warn/15 text-warn border-warn/30",
  medium: "bg-warn/15 text-warn border-warn/30",
  pending_review: "bg-warn/15 text-warn border-warn/30",
  partial: "bg-warn/15 text-warn border-warn/30",
  blocked: "bg-bad/15 text-bad border-bad/30",
  low: "bg-bad/15 text-bad border-bad/30",
  fail: "bg-bad/15 text-bad border-bad/30",
  rejected: "bg-bad/15 text-bad border-bad/30",
  supported: "bg-ok/15 text-ok border-ok/30",
};

export function Badge({ tone, children }: { tone?: string | null; children: ReactNode }) {
  const cls = (tone && TONES[tone.toLowerCase()]) || "bg-line/40 text-muted border-line";
  return (
    <span className={`inline-flex items-center rounded-full border px-2 py-0.5 text-xs font-medium ${cls}`}>
      {children}
    </span>
  );
}

export function PassFail({ passed }: { passed: boolean | null }) {
  if (passed === null) return <Badge>—</Badge>;
  return <Badge tone={passed ? "pass" : "fail"}>{passed ? "PASS" : "FAIL"}</Badge>;
}

export function Id({ children, title }: { children: string; title?: string }) {
  return (
    <code title={title} className="rounded bg-line/40 px-1 py-0.5 font-mono text-[11px] text-fg">
      {children}
    </code>
  );
}

export function Button({
  children,
  onClick,
  disabled,
  kind = "primary",
  type = "button",
}: {
  children: ReactNode;
  onClick?: () => void;
  disabled?: boolean;
  kind?: "primary" | "ghost" | "danger";
  type?: "button" | "submit";
}) {
  const base = "rounded-md px-3 py-1.5 text-sm font-medium transition disabled:opacity-50 disabled:cursor-not-allowed";
  const style =
    kind === "primary"
      ? "bg-accent text-white hover:bg-accent/90"
      : kind === "danger"
        ? "bg-bad text-white hover:bg-bad/90"
        : "border border-line bg-panel hover:bg-line/30";
  return (
    <button type={type} onClick={onClick} disabled={disabled} className={`${base} ${style}`}>
      {children}
    </button>
  );
}

export function ErrorBox({ message }: { message: string | null }) {
  if (!message) return null;
  return (
    <div role="alert" className="rounded-md border border-bad/40 bg-bad/10 px-3 py-2 text-sm text-bad">
      {message}
    </div>
  );
}

export function Spinner({ label = "Loading" }: { label?: string }) {
  return (
    <div className="flex items-center gap-2 text-sm text-muted">
      <span className="inline-block h-3 w-3 animate-spin rounded-full border-2 border-muted border-t-transparent" />
      {label}…
    </div>
  );
}

export function Empty({ children }: { children: ReactNode }) {
  return <p className="text-sm text-muted">{children}</p>;
}

export function Th({ children }: { children: ReactNode }) {
  return <th className="px-2 py-1.5 text-left text-xs font-semibold uppercase tracking-wide text-muted">{children}</th>;
}

export function Td({ children, mono = false }: { children: ReactNode; mono?: boolean }) {
  return <td className={`px-2 py-1.5 align-top ${mono ? "font-mono text-xs" : ""}`}>{children}</td>;
}

export function fmt(v: number | null | undefined, digits = 3): string {
  if (v === null || v === undefined || Number.isNaN(v)) return "—";
  return v.toFixed(digits);
}

export function pct(v: number | null | undefined): string {
  return v === null || v === undefined ? "—" : `${Math.round(v * 100)}%`;
}
