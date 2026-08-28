/**
 * Shapes and formatters. Client-safe on purpose: the loader lives in
 * `lib/server.ts` because it touches the filesystem, and a client component
 * that imports a type from here must not drag `node:fs` into the browser
 * bundle with it.
 *
 * Every shape below is produced by `python/run_demo.py`. Nothing on this
 * dashboard is typed in by hand -- if the terminal and the dashboard ever
 * disagree, one of them is stale and the fix is to rerun the demo.
 */

export type Verdict =
  | "RECOVERED" | "CLEAN" | "MISSED" | "INFLATED"
  | "FALSE POSITIVE" | "NOT DETECTED";

export interface CalibrationRow {
  factor: string; label: string; true: number;
  naive: number; naive_lo: number; naive_hi: number; naive_verdict: Verdict;
  corrected: number; lo: number; hi: number;
  noise_floor: number; verdict: Verdict; why: string;
}

export interface Factor {
  id: string; group: string; label: string; tier: "A" | "B" | "C";
  how: string; source: string; cost: string; refresh: string;
  simulated: boolean; true_weight: number; note: string;
}

export interface SeparationRow {
  pipeline: string; n_queries: number; est_a: number; est_b: number;
  diff_lo: number; diff_hi: number; separated: boolean; n_required: number;
}

export interface Lab {
  calibration: {
    rows: CalibrationRow[];
    eta_true: number; eta_hat: number;
    bundle: { true: number; naive: number[]; corrected: number[] };
    n_queries: number; n_boot: number;
  };
  separation: {
    a: string; b: string; label_a: string; label_b: string;
    true_a: number; true_b: number;
    pilot: SeparationRow[];
    scale: null | {
      n_queries: number; required: number; est_a: number; est_b: number;
      diff_lo: number; diff_hi: number; separated: boolean;
      rows: { factor: string; true: number; est: number; lo: number;
              hi: number; verdict: Verdict }[];
    };
  };
  heads: {
    rows: { model: string; ndcg10: number; pair_acc: number; kendall_tau: number }[];
    linear_ok: boolean; message: string;
  };
  clusters: {
    ari: number;
    rows: { cluster: string; factor: string; index: number; n_queries: number }[];
    headline: {
      factor: string; label: string; high: string; low: string;
      high_val: number; low_val: number;
      measured_ratio: number; true_ratio: number;
    };
    clusters: string[];
  };
  causal: {
    rows: { factor: string; true_weight: number; fe_positions: number;
            matched_positions: number; lo: number; hi: number }[];
    n_queries: number;
  };
  experiments: {
    runs: { true: number; est: number; lo: number; hi: number;
            significant: boolean; n_per_arm: number; days: number;
            mde: number | null }[];
    coverage: number;
    mde: { urls_per_arm: number; mde_positions: number | null }[];
  };
  meta: {
    seed: number; pilot_queries: number; scale_queries: number | null;
    runtime_s: number; numpy: string; pandas: string; lightgbm: boolean;
    n_factors: number; n_simulated: number; generated: string;
  };
  factors: Factor[];
  rejected: { factor: string; why: string }[];
  groups: Record<string, string>;
  tiers: Record<string, string>;
}

export const VERDICT_TONE: Record<Verdict, string> = {
  RECOVERED: "var(--verify)",
  CLEAN: "var(--verify)",
  MISSED: "var(--alert)",
  INFLATED: "var(--alert)",
  "FALSE POSITIVE": "var(--alert)",
  "NOT DETECTED": "var(--muted)",
};

export const fmt = (n: number | null | undefined, d = 2) =>
  n === null || n === undefined || Number.isNaN(n) ? "--" : n.toFixed(d);

export const int = (n: number) => n.toLocaleString("en-US");
