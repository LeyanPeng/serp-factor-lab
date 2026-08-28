"use client";

import { useState } from "react";
import type { CalibrationRow } from "@/lib/lab";
import { fmt } from "@/lib/lab";
import { ErrorBar, VerdictBadge } from "./ui";

type Mode = "naive" | "corrected";

export default function CalibrationTable({ rows }: { rows: CalibrationRow[] }) {
  const [mode, setMode] = useState<Mode>("naive");
  const sorted = [...rows].sort((a, b) => b.true - a.true);
  const max = 1.2;

  const pick = (r: CalibrationRow) =>
    mode === "naive"
      ? { est: r.naive, lo: r.naive_lo, hi: r.naive_hi, v: r.naive_verdict }
      : { est: r.corrected, lo: r.lo, hi: r.hi, v: r.verdict };

  return (
    <div>
      <div className="flex flex-wrap items-center gap-x-1 gap-y-3 mb-6">
        {(["naive", "corrected"] as Mode[]).map((m) => (
          <button
            key={m}
            onClick={() => setMode(m)}
            className="mono text-[10.5px] tracking-[0.13em] uppercase px-4 py-2
                       border transition-colors"
            style={{
              color: mode === m ? "var(--ink)" : "var(--paper-dim)",
              background: mode === m ? "var(--signal)" : "transparent",
              borderColor: mode === m ? "var(--signal)" : "var(--rule)",
            }}
          >
            {m === "naive" ? "uncorrected" : "corrected"}
          </button>
        ))}
        <span
          className="text-[13px] ml-4"
          style={{ color: "var(--paper-dim)" }}
        >
          {mode === "naive"
            ? "Naive within-query fit. No position-bias correction, no measurement-error adjustment. This is the standard build."
            : "The same fit plus IPW on the click signal and an errors-in-variables adjustment on content depth."}
        </span>
      </div>

      <div className="scroll-x">
        <table className="lab min-w-[820px]">
          <thead>
            <tr>
              <th style={{ width: 250 }}>Factor</th>
              <th className="text-right" style={{ width: 62 }}>Truth</th>
              <th className="text-right" style={{ width: 62 }}>Estimate</th>
              <th style={{ width: 300 }}>Interval vs truth</th>
              <th style={{ width: 130 }}>Verdict</th>
            </tr>
          </thead>
          <tbody>
            {sorted.map((r) => {
              const p = pick(r);
              const off = Math.abs(p.est - r.true);
              return (
                <tr key={r.factor}>
                  <td>
                    <div className="text-[13.5px]">{r.label}</div>
                    <div
                      className="mono text-[10px] mt-0.5"
                      style={{ color: "var(--paper-faint)" }}
                    >
                      {r.factor}
                    </div>
                  </td>
                  <td className="n text-[13px]" style={{ color: "var(--truth)" }}>
                    {fmt(r.true)}
                  </td>
                  <td
                    className="n text-[13px]"
                    style={{
                      color: off > 0.25 ? "var(--alert)" : "var(--paper)",
                    }}
                  >
                    {fmt(p.est)}
                  </td>
                  <td>
                    <ErrorBar
                      lo={p.lo}
                      hi={p.hi}
                      point={p.est}
                      truth={r.true}
                      max={max}
                      tone={
                        p.v === "RECOVERED" || p.v === "CLEAN"
                          ? "var(--verify)"
                          : p.v === "NOT DETECTED"
                          ? "var(--muted)"
                          : "var(--alert)"
                      }
                    />
                  </td>
                  <td>
                    <VerdictBadge v={p.v} />
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
      </div>

      <div
        className="mono text-[10.5px] mt-4 flex flex-wrap gap-x-6 gap-y-1"
        style={{ color: "var(--paper-faint)" }}
      >
        <span>scale 0 &ndash; {fmt(max, 1)}, importance relative to the strongest factor</span>
        <span>blue rule = the weight we set</span>
        <span>bar = 95% cluster-bootstrap interval</span>
      </div>
    </div>
  );
}
