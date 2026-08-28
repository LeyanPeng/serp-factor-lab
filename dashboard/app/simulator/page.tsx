import { getLab } from "@/lib/server";
import { fmt, int } from "@/lib/lab";
import Simulator, { type Lever } from "@/components/Simulator";
import { Note, SectionTitle } from "@/components/ui";

export default function SimulatorPage() {
  const lab = getLab();
  const verdicts = new Map(lab.calibration.rows.map((r) => [r.factor, r.verdict]));

  const levers: Lever[] = lab.causal.rows
    .map((r) => ({
      factor: r.factor,
      label: (r as unknown as { label?: string }).label ?? r.factor,
      matched_positions: r.matched_positions,
      lo: r.lo,
      hi: r.hi,
      effort: (r as unknown as { effort?: number }).effort ?? 3,
      effort_label:
        (r as unknown as { effort_label?: string }).effort_label ?? "weeks",
      verdict: verdicts.get(r.factor) ?? "NOT DETECTED",
      true_weight: r.true_weight,
    }))
    .sort((a, b) => a.matched_positions - b.matched_positions);

  const roi = [...levers]
    .map((l) => ({ ...l, per: l.matched_positions / l.effort }))
    .sort((a, b) => a.per - b.per)
    .slice(0, 5);

  return (
    <div className="space-y-14">
      <SectionTitle
        eyebrow="04 / Simulator"
        title="What does the work actually buy?"
        lede={`Weights describe. Positions decide. Every coefficient below comes from
               comparing pages on the same SERP that are near-identical on every
               other factor, across ${int(lab.causal.n_queries)} queries, so it
               answers the question a client asks rather than the one a model
               prefers to answer.`}
      />

      <Simulator levers={levers} />

      <Note>
        <strong style={{ color: "var(--paper)" }}>
          Every lever here is observational, and observational estimates
          inherit whatever is wrong with the instrument.
        </strong>{" "}
        The keyword-density lever moves pages in this model even though its
        true weight is exactly zero, because it is measured cleanly and sits
        next to something that works. That is not a bug in the simulator, it is
        the reason the simulator is not allowed to authorise spending on its
        own. A lever gets into a client plan after a split test, not before.
      </Note>

      {/* -------------------------------------------------------------- */}
      <section>
        <div className="flex items-baseline justify-between flex-wrap gap-3 mb-2">
          <h3 className="display text-[24px]">Return per unit of effort</h3>
          <span className="mono text-[10px]" style={{ color: "var(--alert)" }}>
            effort is a judgement, not a measurement
          </span>
        </div>
        <p className="text-[14.5px] max-w-[72ch] mb-6"
           style={{ color: "var(--paper-dim)" }}>
          Effect size alone ranks the work backwards. A title rewrite worth a
          fifth of a position beats a year of link acquisition worth half of
          one, and no statistic will tell you that &mdash; it needs delivery
          data, which an agency has and a tool vendor does not.
        </p>

        <div className="scroll-x">
          <table className="lab min-w-[720px]">
            <thead>
              <tr>
                <th style={{ width: 240 }}>Factor</th>
                <th className="text-right" style={{ width: 100 }}>Positions / SD</th>
                <th style={{ width: 130 }}>Effort</th>
                <th className="text-right" style={{ width: 110 }}>Per effort</th>
                <th style={{ width: 200 }}>Calibration</th>
              </tr>
            </thead>
            <tbody>
              {roi.map((l, i) => {
                const bad = !["RECOVERED", "CLEAN"].includes(l.verdict);
                return (
                  <tr key={l.factor}>
                    <td>
                      <span className="num text-[10px] mr-2"
                            style={{ color: "var(--signal)" }}>
                        {String(i + 1).padStart(2, "0")}
                      </span>
                      <span className="text-[13.5px]">{l.label}</span>
                    </td>
                    <td className="n text-[13px]">{fmt(l.matched_positions)}</td>
                    <td className="mono text-[11px]"
                        style={{ color: "var(--paper-dim)" }}>
                      {l.effort_label}
                    </td>
                    <td className="n text-[13px]"
                        style={{ color: "var(--signal)" }}>{fmt(l.per, 3)}</td>
                    <td className="text-[12.5px]"
                        style={{ color: bad ? "var(--alert)" : "var(--verify)" }}>
                      {bad
                        ? `${l.verdict} -- blocked until a split test clears it`
                        : "passes calibration"}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>

        {roi[0] && roi[0].true_weight === 0 && (
          <p className="text-[14.5px] mt-6 max-w-[74ch]"
             style={{ color: "var(--alert)" }}>
            The factor at the top of that table has a true weight of exactly
            zero. It wins on return-per-effort because it is cheap to change
            and correlated with something that works. An agency that ranked its
            roadmap this way would spend a quarter shipping nothing, and every
            number on the page would agree with it the whole time.
          </p>
        )}
      </section>
    </div>
  );
}
