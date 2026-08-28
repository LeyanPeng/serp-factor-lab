import { getLab } from "@/lib/server";
import { fmt, int } from "@/lib/lab";
import { Note, SectionTitle, Stat } from "@/components/ui";

const CLUSTER_TONE: Record<string, string> = {
  "ymyl-finance": "var(--signal)",
  ecommerce: "var(--truth)",
  "local-service": "var(--alert)",
  informational: "var(--verify)",
};

export default function Clusters() {
  const lab = getLab();
  const { rows, clusters, headline, ari } = lab.clusters;

  const byFactor = new Map<string, Record<string, number>>();
  for (const r of rows) {
    const e = byFactor.get(r.factor) ?? {};
    e[r.cluster] = r.index;
    byFactor.set(r.factor, e);
  }
  const table = [...byFactor.entries()]
    .map(([factor, vals]) => {
      const v = clusters.map((c) => vals[c] ?? 0);
      const hi = Math.max(...v), lo = Math.min(...v);
      return { factor, vals, spread: hi - lo, ratio: hi / Math.max(lo, 1e-6) };
    })
    .sort((a, b) => b.spread - a.spread);

  const labelOf = (id: string) =>
    lab.factors.find((f) => f.id === id)?.label ??
    (id === "kw_density" ? "Keyword density (decoy)" : id);

  const maxVal = Math.max(...rows.map((r) => r.index));

  return (
    <div className="space-y-14">
      <SectionTitle
        eyebrow="03 / Intent clusters"
        title="One weight table is the wrong deliverable"
        lede="A single vector averaged over finance, e-commerce, local services and
              plain informational queries describes none of them. For an agency this
              is not a statistical nicety, it is the product: where a wealth manager
              should spend and where a plumber should spend are different answers,
              and a global table cannot give you either."
      />

      <section className="grid gap-8 sm:grid-cols-3">
        <Stat
          label="Clusters found without labels"
          value={fmt(ari, 3)}
          sub="adjusted Rand index against the true grouping, from SERP shape alone"
          tone={ari > 0.8 ? "var(--verify)" : "var(--signal)"}
        />
        <Stat
          label={`${labelOf(headline.factor)} spread`}
          value={`${fmt(headline.measured_ratio, 1)}x`}
          sub={`${headline.high} ${fmt(headline.high_val)} vs ${headline.low} ${fmt(headline.low_val)}`}
          tone="var(--signal)"
          big
        />
        <Stat
          label="True spread"
          value={`${fmt(headline.true_ratio, 1)}x`}
          sub="we under-report it, and we say so rather than rounding up"
          tone="var(--truth)"
        />
      </section>

      <Note>
        Clusters are <strong style={{ color: "var(--paper)" }}>discovered,
        never assumed</strong>. The fingerprint is built from the shape of the
        result set &mdash; what the winning pages are made of &mdash; and never
        from the words in the query. Two queries with nothing lexically in
        common belong together when Google treats them the same way, and that
        is the grouping the weights follow.
      </Note>

      {/* -------------------------------------------------------------- */}
      <section>
        <div className="flex items-baseline justify-between flex-wrap gap-3 mb-5">
          <h3 className="display text-[24px]">Weight by intent, sorted by disagreement</h3>
          <div className="flex flex-wrap gap-x-5 gap-y-1 mono text-[10px]">
            {clusters.map((c) => (
              <span key={c} className="flex items-center gap-1.5"
                    style={{ color: "var(--paper-faint)" }}>
                <span style={{ width: 9, height: 9,
                               background: CLUSTER_TONE[c] ?? "var(--paper-dim)" }} />
                {c}
              </span>
            ))}
          </div>
        </div>

        <div className="space-y-px" style={{ background: "var(--rule)" }}>
          {table.map((row) => (
            <div key={row.factor}
                 className="grid grid-cols-[minmax(150px,220px)_1fr_64px] gap-5
                            items-center px-5 py-4"
                 style={{ background: "var(--ink-2)" }}>
              <div>
                <div className="text-[13.5px] leading-snug">
                  {labelOf(row.factor)}
                </div>
                <div className="mono text-[10px] mt-0.5"
                     style={{ color: "var(--paper-faint)" }}>
                  {row.factor}
                </div>
              </div>

              <div className="space-y-1.5">
                {clusters.map((c) => (
                  <div key={c} className="flex items-center gap-3">
                    <span className="mono text-[9.5px] w-[92px] shrink-0
                                     text-right"
                          style={{ color: "var(--paper-faint)" }}>{c}</span>
                    <div className="flex-1 h-[9px] relative"
                         style={{ background: "var(--ink-3)" }}>
                      <div className="absolute inset-y-0 left-0"
                           style={{
                             width: `${((row.vals[c] ?? 0) / maxVal) * 100}%`,
                             background: CLUSTER_TONE[c] ?? "var(--paper-dim)",
                             opacity: 0.85,
                           }} />
                    </div>
                    <span className="num text-[11px] w-[36px] text-right">
                      {fmt(row.vals[c] ?? 0)}
                    </span>
                  </div>
                ))}
              </div>

              <div className="text-right">
                <div className="num text-[19px]" style={{ color: "var(--signal)" }}>
                  {fmt(row.ratio, 1)}x
                </div>
                <div className="mono text-[9px]"
                     style={{ color: "var(--paper-faint)" }}>spread</div>
              </div>
            </div>
          ))}
        </div>
      </section>

      {/* -------------------------------------------------------------- */}
      <section className="grid gap-px lg:grid-cols-2"
               style={{ background: "var(--rule)" }}>
        <div className="p-7" style={{ background: "var(--ink-2)" }}>
          <div className="eyebrow mb-4">What an account manager does with this</div>
          <p className="text-[14.5px] leading-relaxed mb-4">
            {labelOf(headline.factor)} carries{" "}
            <span className="num" style={{ color: "var(--signal)" }}>
              {fmt(headline.measured_ratio, 1)}x
            </span>{" "}
            the weight in <strong>{headline.high}</strong> that it carries in{" "}
            <strong>{headline.low}</strong>. Same agency, same month, opposite
            recommendation.
          </p>
          <p className="text-[14.5px] leading-relaxed"
             style={{ color: "var(--paper-dim)" }}>
            A link-building retainer sold at a flat rate across a client book
            is being overcharged to some of those clients and undercharged to
            the rest. This table is the version of the argument that survives
            the client asking why.
          </p>
        </div>
        <div className="p-7" style={{ background: "var(--ink-2)" }}>
          <div className="eyebrow mb-4">The honest caveat</div>
          <p className="text-[14.5px] leading-relaxed mb-4"
             style={{ color: "var(--paper-dim)" }}>
            Measured spread is{" "}
            <span className="num">{fmt(headline.measured_ratio, 1)}x</span>;
            the truth is{" "}
            <span className="num" style={{ color: "var(--truth)" }}>
              {fmt(headline.true_ratio, 1)}x
            </span>
            . We understate the difference, because only the top ten results
            are ever observed and that truncation compresses every estimate
            toward the middle.
          </p>
          <p className="text-[14.5px] leading-relaxed">
            So the direction is trustworthy and the magnitude is a floor. That
            is a usable finding as long as it is stated that way, and a
            liability as soon as someone rounds it into a promise.
          </p>
        </div>
      </section>

      <p className="mono text-[10.5px]" style={{ color: "var(--paper-faint)" }}>
        fitted on {int(lab.causal.n_queries)} queries, one model per cluster,
        minimum 150 queries per cluster to fit
      </p>
    </div>
  );
}
