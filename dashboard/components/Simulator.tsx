"use client";

import { useMemo, useState } from "react";
import { fmt } from "@/lib/lab";

export interface Lever {
  factor: string;
  label: string;
  matched_positions: number;   // positions per +1 SD, negative = moves up
  lo: number;
  hi: number;
  effort: number;
  effort_label: string;
  verdict: string;             // calibration verdict for this factor
  true_weight: number;
}

const TRUSTED = new Set(["RECOVERED", "CLEAN"]);

export default function Simulator({ levers }: { levers: Lever[] }) {
  const [moves, setMoves] = useState<Record<string, number>>({});
  const [startRank, setStartRank] = useState(8);

  const set = (f: string, v: number) =>
    setMoves((m) => ({ ...m, [f]: v }));

  const result = useMemo(() => {
    let delta = 0, lo = 0, hi = 0, effort = 0, untrusted = 0;
    for (const l of levers) {
      const sd = moves[l.factor] ?? 0;
      if (!sd) continue;
      delta += l.matched_positions * sd;
      // intervals added in quadrature: the levers are largely independent
      // and summing the bounds directly would be needlessly pessimistic
      const half = ((l.hi - l.lo) / 2) * Math.abs(sd);
      lo += half * half;
      effort += l.effort * Math.abs(sd);
      if (!TRUSTED.has(l.verdict)) untrusted += 1;
    }
    const halfWidth = Math.sqrt(lo);
    hi = delta + 1.96 * halfWidth / 1.96;   // half-widths are already 95%
    lo = delta - halfWidth;
    hi = delta + halfWidth;
    const clamp = (r: number) => Math.max(1, Math.min(10, r));
    return {
      delta, lo, hi, effort, untrusted,
      end: clamp(startRank + delta),
      endLo: clamp(startRank + hi),
      endHi: clamp(startRank + lo),
      active: levers.filter((l) => moves[l.factor]),
    };
  }, [moves, levers, startRank]);

  const anyMove = result.active.length > 0;

  return (
    <div className="grid gap-px lg:grid-cols-[1fr_380px]"
         style={{ background: "var(--rule)" }}>
      {/* ---------------------------------------------------------- levers */}
      <div className="p-7" style={{ background: "var(--ink-2)" }}>
        <div className="flex items-baseline justify-between mb-6 gap-4 flex-wrap">
          <div className="eyebrow">Move a factor, in standard deviations</div>
          <button
            onClick={() => setMoves({})}
            className="mono text-[10px] tracking-[0.12em] uppercase px-3 py-1.5
                       border transition-colors hover:text-[var(--signal)]"
            style={{ borderColor: "var(--rule)", color: "var(--paper-faint)" }}
          >
            reset
          </button>
        </div>

        <div className="space-y-5">
          {levers.map((l) => {
            const v = moves[l.factor] ?? 0;
            const trusted = TRUSTED.has(l.verdict);
            return (
              <div key={l.factor}>
                <div className="flex items-baseline justify-between gap-3 mb-1">
                  <span className="text-[13.5px] flex items-baseline gap-2">
                    {l.label}
                    {!trusted && (
                      <span className="mono text-[9px] px-1 border"
                            style={{ color: "var(--alert)",
                                     borderColor: "var(--alert)" }}
                            title={`Calibration verdict: ${l.verdict}. This lever's estimate is known to be wrong in a measured way.`}>
                        {l.verdict}
                      </span>
                    )}
                  </span>
                  <span className="num text-[12px] shrink-0"
                        style={{ color: v ? "var(--signal)" : "var(--paper-faint)" }}>
                    {v > 0 ? "+" : ""}{fmt(v, 1)} SD
                  </span>
                </div>
                <input
                  type="range" min={-2} max={2} step={0.25} value={v}
                  onChange={(e) => set(l.factor, Number(e.target.value))}
                  aria-label={l.label}
                />
                <div className="mono text-[9.5px] flex justify-between"
                     style={{ color: "var(--paper-faint)" }}>
                  <span>{fmt(l.matched_positions)} pos / SD</span>
                  <span>effort: {l.effort_label}</span>
                </div>
              </div>
            );
          })}
        </div>
      </div>

      {/* --------------------------------------------------------- readout */}
      <div className="p-7 flex flex-col gap-7"
           style={{ background: "var(--ink-3)" }}>
        <div>
          <div className="eyebrow mb-3">Starting position</div>
          <input
            type="range" min={2} max={10} step={1} value={startRank}
            onChange={(e) => setStartRank(Number(e.target.value))}
            aria-label="starting position"
          />
          <div className="num text-[13px] mt-1">rank {startRank}</div>
        </div>

        <div>
          <div className="eyebrow mb-3">Predicted movement</div>
          <div className="num text-[46px] leading-none"
               style={{ color: anyMove
                 ? (result.delta < 0 ? "var(--verify)" : "var(--alert)")
                 : "var(--paper-faint)" }}>
            {result.delta > 0 ? "+" : ""}{fmt(result.delta)}
          </div>
          <div className="text-[13px] mt-2" style={{ color: "var(--paper-dim)" }}>
            positions {result.delta < 0 ? "up" : result.delta > 0 ? "down" : ""}
          </div>
          {anyMove && (
            <div className="mono text-[11px] mt-3"
                 style={{ color: "var(--paper-faint)" }}>
              95% interval {fmt(result.lo)} to {fmt(result.hi)}
            </div>
          )}
        </div>

        {anyMove && (
          <div>
            <div className="eyebrow mb-3">
              Rank {startRank} &rarr; {fmt(result.end, 1)}
            </div>
            <div className="relative h-[52px]"
                 style={{ background: "var(--ink)" }}>
              {[...Array(10)].map((_, i) => (
                <div key={i} className="absolute top-0 bottom-0"
                     style={{ left: `${(i / 9) * 100}%`, width: 1,
                              background: "var(--rule-soft)" }} />
              ))}
              {/* uncertainty band */}
              <div className="absolute top-[18px] h-[16px]"
                   style={{
                     left: `${((result.endLo - 1) / 9) * 100}%`,
                     width: `${(Math.abs(result.endHi - result.endLo) / 9) * 100}%`,
                     background: "var(--signal)", opacity: 0.22,
                   }} />
              {/* start */}
              <div className="absolute top-[10px] h-[32px] w-[2px]"
                   style={{ left: `${((startRank - 1) / 9) * 100}%`,
                            background: "var(--paper-faint)" }} />
              {/* end */}
              <div className="absolute top-[6px] h-[40px] w-[3px]"
                   style={{ left: `${((result.end - 1) / 9) * 100}%`,
                            background: "var(--signal)" }} />
            </div>
            <div className="mono text-[9.5px] flex justify-between mt-1.5"
                 style={{ color: "var(--paper-faint)" }}>
              <span>1</span><span>5</span><span>10</span>
            </div>
          </div>
        )}

        {anyMove && (
          <div className="pt-5 border-t space-y-3"
               style={{ borderColor: "var(--rule)" }}>
            <div className="flex justify-between text-[13px]">
              <span style={{ color: "var(--paper-dim)" }}>effort score</span>
              <span className="num">{fmt(result.effort, 1)}</span>
            </div>
            <div className="flex justify-between text-[13px]">
              <span style={{ color: "var(--paper-dim)" }}>positions per effort</span>
              <span className="num">
                {fmt(result.effort ? result.delta / result.effort : 0, 3)}
              </span>
            </div>
            {result.untrusted > 0 && (
              <p className="text-[12.5px] leading-relaxed pt-2"
                 style={{ color: "var(--alert)" }}>
                {result.untrusted} of the levers you moved failed calibration.
                Their coefficients are wrong by a known amount and this
                projection inherits that error. Under our own rule they cannot
                go in a client plan until a split test clears them.
              </p>
            )}
          </div>
        )}

        {!anyMove && (
          <p className="text-[13px] leading-relaxed"
             style={{ color: "var(--paper-faint)" }}>
            Move any slider. Estimates come from within-SERP near-twin
            matching, so they are what a page with an otherwise identical
            profile actually achieves &mdash; not what a correlation says it
            should.
          </p>
        )}
      </div>
    </div>
  );
}
