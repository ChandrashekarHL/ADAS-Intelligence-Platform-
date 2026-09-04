"use client";

import { useCallback, useEffect, useState } from "react";

import { ApiError } from "./api";

export type AsyncState<T> = {
  data: T | null;
  error: string | null;
  loading: boolean;
  reload: () => void;
};

type Result<T> = { key: string; data: T | null; error: string | null };

/**
 * Minimal data hook: fetch when `deps` change or reload() is called.
 *
 * `loading` is derived from whether the latest request key has resolved, so the effect
 * never sets state synchronously (React's set-state-in-effect rule). Previous data stays
 * visible while a reload is in flight.
 */
export function useAsync<T>(fn: () => Promise<T>, deps: unknown[]): AsyncState<T> {
  const [tick, setTick] = useState(0);
  const [result, setResult] = useState<Result<T>>({ key: "", data: null, error: null });
  const key = `${JSON.stringify(deps)}#${tick}`;

  useEffect(() => {
    let cancelled = false;
    fn()
      .then((data) => {
        if (!cancelled) setResult({ key, data, error: null });
      })
      .catch((e: unknown) => {
        if (!cancelled) setResult({ key, data: null, error: describeError(e) });
      });
    return () => {
      cancelled = true;
    };
    // `fn` is intentionally excluded: callers pass inline closures; `deps` drive refetching.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [key]);

  const reload = useCallback(() => setTick((t) => t + 1), []);
  return { data: result.data, error: result.key === key ? result.error : null, loading: result.key !== key, reload };
}

export function describeError(e: unknown): string {
  if (e instanceof ApiError) return e.detail;
  if (e instanceof TypeError) return `Cannot reach the API (${e.message}). Is the backend running?`;
  if (e instanceof Error) return e.message;
  return String(e);
}
