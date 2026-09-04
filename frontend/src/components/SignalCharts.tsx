"use client";

import { useMemo } from "react";
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from "recharts";

import type { Event, Signals } from "@/lib/api";

const EVENT_COLOURS: Record<string, string> = {
  ttc_threshold_crossing: "var(--warn)",
  aeb_brake_command: "var(--accent)",
  collision: "var(--bad)",
};

const PANELS: { title: string; keys: { col: string; label: string; colour: string }[]; unit: string }[] = [
  {
    title: "Ego speed",
    unit: "m/s",
    keys: [{ col: "ego_speed_mps", label: "ego speed", colour: "var(--accent)" }],
  },
  {
    title: "Relative distance",
    unit: "m",
    keys: [{ col: "relative_distance_m", label: "gap to lead", colour: "var(--ok)" }],
  },
  {
    title: "Object confidence",
    unit: "",
    keys: [{ col: "object_confidence", label: "confidence", colour: "var(--warn)" }],
  },
  {
    title: "Brake command / AEB state",
    unit: "",
    keys: [
      { col: "brake_command", label: "brake cmd", colour: "var(--bad)" },
      { col: "aeb_state", label: "aeb state", colour: "var(--muted)" },
    ],
  },
];

type Row = Record<string, number | null>;

export function SignalCharts({ signals, events }: { signals: Signals; events: Event[] }) {
  const rows = useMemo<Row[]>(() => {
    const t = signals.data["timestamp_s"] ?? [];
    return t.map((ts, i) => {
      const row: Row = { t: ts };
      for (const c of signals.columns) row[c] = signals.data[c]?.[i] ?? null;
      return row;
    });
  }, [signals]);

  return (
    <div>
      <p className="mb-2 text-xs text-muted">
        {signals.rows_total} samples, plotted every {signals.step} · vertical lines mark detected events
      </p>
      <div className="grid gap-4 md:grid-cols-2">
        {PANELS.filter((p) => p.keys.some((k) => signals.columns.includes(k.col))).map((p) => (
          <div key={p.title} className="h-56">
            <div className="mb-1 text-xs font-semibold text-muted">
              {p.title} {p.unit && <span className="font-normal">[{p.unit}]</span>}
            </div>
            <ResponsiveContainer width="100%" height="90%">
              <LineChart data={rows} margin={{ top: 4, right: 8, bottom: 4, left: -12 }}>
                <CartesianGrid stroke="var(--line)" strokeDasharray="3 3" />
                <XAxis dataKey="t" type="number" domain={["dataMin", "dataMax"]} tick={{ fontSize: 10 }} tickFormatter={(v: number) => v.toFixed(1)} />
                <YAxis tick={{ fontSize: 10 }} width={48} />
                <Tooltip
                  contentStyle={{ background: "var(--panel)", border: "1px solid var(--line)", fontSize: 12 }}
                  labelFormatter={(v) => `t = ${Number(v).toFixed(2)} s`}
                />
                {p.keys
                  .filter((k) => signals.columns.includes(k.col))
                  .map((k) => (
                    <Line key={k.col} type="monotone" dataKey={k.col} name={k.label} stroke={k.colour} dot={false} strokeWidth={1.5} isAnimationActive={false} connectNulls={false} />
                  ))}
                {events.map((e) => (
                  <ReferenceLine
                    key={e.id}
                    x={e.t_s}
                    stroke={EVENT_COLOURS[e.event_type] ?? "var(--muted)"}
                    strokeDasharray="4 2"
                    label={{ value: e.event_type.split("_")[0], fontSize: 9, position: "insideTopRight", fill: EVENT_COLOURS[e.event_type] ?? "var(--muted)" }}
                  />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
        ))}
      </div>
    </div>
  );
}
