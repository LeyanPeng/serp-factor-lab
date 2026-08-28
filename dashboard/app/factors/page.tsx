import { getLab } from "@/lib/server";
import { fmt } from "@/lib/lab";
import { SectionTitle, TierBadge } from "@/components/ui";

export default function Factors() {
  const lab = getLab();
  const byGroup = Object.entries(lab.groups).map(([key, label]) => ({
    key, label, items: lab.factors.filter((f) => f.group === key),
  }));
  const tierCount = (t: string) => lab.factors.filter((f) => f.tier === t).length;

  return (
    <div className="space-y-14">
      <SectionTitle
        eyebrow="02 / Factor bench"
        title={`${lab.meta.n_factors} factors, sorted by how much we actually know`}
        lede="Not two hundred. The list is short on purpose: each entry has a
              measurement method, a data source, a marginal cost and a refresh
              cadence, because a factor you cannot measure weekly at a price you
              can defend is not a factor, it is an opinion."
      />

      <section className="grid gap-px sm:grid-cols-3"
               style={{ background: "var(--rule)" }}>
        {(["A", "B", "C"] as const).map((t) => (
          <div key={t} className="p-6" style={{ background: "var(--ink-2)" }}>
            <div className="flex items-center gap-3 mb-3">
              <TierBadge tier={t} />
              <span className="num text-[26px]">
                {t === "C" ? lab.rejected.length : tierCount(t)}
              </span>
            </div>
            <div className="text-[14px] mb-1.5">{lab.tiers[t]}</div>
            <p className="text-[12.5px] leading-relaxed"
               style={{ color: "var(--paper-dim)" }}>
              {t === "A"
                ? "Stated in Google's own documentation, or by a Google witness under oath."
                : t === "B"
                ? "Pandu Nayak's DOJ testimony confirming click data as a core signal, plus the March 2024 Content Warehouse schema leak."
                : "Correlational folklore. Listed below so the omission is a decision, not an oversight."}
            </p>
          </div>
        ))}
      </section>

      {byGroup.map((g) => (
        <section key={g.key}>
          <div className="flex items-baseline gap-4 mb-5 pb-3 border-b"
               style={{ borderColor: "var(--rule)" }}>
            <h3 className="display text-[24px]">{g.label}</h3>
            <span className="mono text-[10.5px]"
                  style={{ color: "var(--paper-faint)" }}>
              {g.items.length} factors
            </span>
          </div>

          <div className="scroll-x">
            <table className="lab min-w-[900px]">
              <thead>
                <tr>
                  <th style={{ width: 30 }} />
                  <th style={{ width: 220 }}>Factor</th>
                  <th style={{ width: 300 }}>How we measure it</th>
                  <th style={{ width: 170 }}>Source</th>
                  <th className="text-right" style={{ width: 90 }}>Cost / 1k</th>
                  <th style={{ width: 100 }}>Refresh</th>
                </tr>
              </thead>
              <tbody>
                {g.items.map((f) => (
                  <tr key={f.id}>
                    <td><TierBadge tier={f.tier} /></td>
                    <td>
                      <div className="text-[13.5px] flex items-baseline gap-2">
                        {f.label}
                        {f.simulated && (
                          <span className="mono text-[9px] px-1 border"
                                style={{ color: "var(--signal)",
                                         borderColor: "var(--signal)" }}
                                title={`Carried into the calibration harness with a true weight of ${fmt(f.true_weight, 3)}`}>
                            HARNESS
                          </span>
                        )}
                      </div>
                      {f.note && (
                        <div className="text-[12px] mt-1 leading-snug max-w-[30ch]"
                             style={{ color: "var(--paper-faint)" }}>
                          {f.note}
                        </div>
                      )}
                    </td>
                    <td className="text-[12.5px] leading-snug"
                        style={{ color: "var(--paper-dim)" }}>{f.how}</td>
                    <td className="mono text-[11px]"
                        style={{ color: "var(--paper-dim)" }}>{f.source}</td>
                    <td className="n text-[12px]">{f.cost}</td>
                    <td className="mono text-[11px]"
                        style={{ color: "var(--paper-faint)" }}>{f.refresh}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </section>
      ))}

      {/* -------------------------------------------------------------- */}
      <section>
        <div className="flex items-baseline gap-4 mb-5 pb-3 border-b"
             style={{ borderColor: "var(--alert)" }}>
          <h3 className="display text-[24px]" style={{ color: "var(--alert)" }}>
            What we refuse to track
          </h3>
          <span className="mono text-[10.5px]"
                style={{ color: "var(--paper-faint)" }}>
            {lab.rejected.length} entries
          </span>
        </div>
        <p className="text-[14.5px] max-w-[70ch] mb-6"
           style={{ color: "var(--paper-dim)" }}>
          Half of a factor list is the half you leave out. Every one of these
          appears in someone&rsquo;s ranking-factor report, and every one of
          them will make a model look better while making its advice worse.
        </p>
        <div className="grid gap-px sm:grid-cols-2"
             style={{ background: "var(--rule)" }}>
          {lab.rejected.map((r) => (
            <div key={r.factor} className="p-5"
                 style={{ background: "var(--ink-2)" }}>
              <div className="text-[14.5px] mb-1.5"
                   style={{ color: "var(--alert)" }}>{r.factor}</div>
              <p className="text-[13px] leading-relaxed"
                 style={{ color: "var(--paper-dim)" }}>{r.why}</p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
