import { getLab } from "@/lib/server";
import { fmt, int } from "@/lib/lab";
import { Note, SectionTitle, Stat } from "@/components/ui";

export default function Experiments() {
  const lab = getLab();
  const e = lab.experiments;
  const heads = lab.heads;
  const maxMde = Math.max(...e.mde.map((m) => m.mde_positions ?? 0));

  return (
    <div className="space-y-14">
      <SectionTitle
        eyebrow="05 / Experiments"
        title="The only rung that produces a number you can sign"
        lede="Two cohorts of comparable client URLs, one treated, both riding the same
              market-wide shocks. Difference-in-differences against the control arm.
              This is the only place in the whole stack where a badly measured
              instrument cannot fool us, because the thing being changed is the real
              thing and not a proxy for it."
      />

      {/* -------------------------------------------------------------- */}
      <section>
        <div className="eyebrow mb-5">Two runs of the same test</div>
        <div className="grid gap-px md:grid-cols-2"
             style={{ background: "var(--rule)" }}>
          {e.runs.map((r) => {
            const good = r.significant;
            return (
              <div key={r.n_per_arm} className="p-7"
                   style={{ background: "var(--ink-2)" }}>
                <div className="flex items-baseline justify-between mb-6">
                  <span className="num text-[15px]">
                    {int(r.n_per_arm)} URLs per arm
                  </span>
                  <span className="mono text-[10px] tracking-[0.12em] uppercase
                                   px-2 py-1 border"
                        style={{ color: good ? "var(--verify)" : "var(--alert)",
                                 borderColor: good ? "var(--verify)" : "var(--alert)" }}>
                    {good ? "detected" : "inconclusive"}
                  </span>
                </div>

                {/* effect axis: -1.2 .. +0.4 positions */}
                <div className="relative h-[64px] mb-3"
                     style={{ background: "var(--ink)" }}>
                  {(() => {
                    const min = -1.3, max = 0.5;
                    const pc = (v: number) =>
                      `${((v - min) / (max - min)) * 100}%`;
                    return (
                      <>
                        <div className="absolute top-0 bottom-0"
                             style={{ left: pc(0), width: 1,
                                      background: "var(--rule)" }} />
                        <div className="absolute top-[26px] h-[12px]"
                             style={{ left: pc(r.lo),
                                      width: `calc(${pc(r.hi)} - ${pc(r.lo)})`,
                                      background: good ? "var(--verify)" : "var(--alert)",
                                      opacity: 0.28 }} />
                        <div className="absolute top-[20px] h-[24px] w-[2.5px]"
                             style={{ left: pc(r.est),
                                      background: good ? "var(--verify)" : "var(--alert)" }} />
                        <div className="absolute top-[12px] h-[40px] w-[1px]"
                             style={{ left: pc(r.true),
                                      background: "var(--truth)" }} />
                      </>
                    );
                  })()}
                </div>
                <div className="mono text-[9.5px] flex justify-between mb-6"
                     style={{ color: "var(--paper-faint)" }}>
                  <span>-1.3</span>
                  <span>no effect</span>
                  <span>+0.5</span>
                </div>

                <div className="grid grid-cols-2 gap-5 text-[13px]">
                  <div>
                    <div className="eyebrow mb-1.5">Estimate</div>
                    <div className="num text-[18px]">
                      {fmt(r.est)}{" "}
                      <span className="text-[11px]"
                            style={{ color: "var(--paper-faint)" }}>
                        [{fmt(r.lo)}, {fmt(r.hi)}]
                      </span>
                    </div>
                  </div>
                  <div>
                    <div className="eyebrow mb-1.5">True effect</div>
                    <div className="num text-[18px]" style={{ color: "var(--truth)" }}>
                      {fmt(r.true)}
                    </div>
                  </div>
                </div>

                <p className="text-[13px] leading-relaxed mt-5 pt-5 border-t"
                   style={{ borderColor: "var(--rule)",
                            color: good ? "var(--paper-dim)" : "var(--alert)" }}>
                  {r.mde !== null && r.mde > Math.abs(r.true)
                    ? `Underpowered by construction: this arm size resolves ${fmt(r.mde)} positions and the real effect is ${fmt(Math.abs(r.true))}. The test did not fail. It was never able to succeed.`
                    : `Resolves ${fmt(r.mde ?? 0)} positions, comfortably below the ${fmt(Math.abs(r.true))} on offer.`}
                </p>
              </div>
            );
          })}
        </div>
      </section>

      <Note>
        <strong style={{ color: "var(--paper)" }}>
          &ldquo;The change did nothing&rdquo; and &ldquo;we could never have
          seen it&rdquo; look identical in a report and mean opposite things.
        </strong>{" "}
        Working out the minimum detectable effect before the test runs is the
        only thing that separates them, and it is a five-minute calculation
        that almost nobody does.
      </Note>

      {/* -------------------------------------------------------------- */}
      <section>
        <div className="flex items-baseline justify-between flex-wrap gap-3 mb-6">
          <h3 className="display text-[24px]">How big does the test have to be?</h3>
          <span className="mono text-[10px]" style={{ color: "var(--paper-faint)" }}>
            measured by simulation, 80% power, 28+28 days
          </span>
        </div>

        <div className="space-y-px" style={{ background: "var(--rule)" }}>
          {e.mde.map((m) => (
            <div key={m.urls_per_arm}
                 className="grid grid-cols-[110px_1fr_92px] items-center gap-5
                            px-5 py-3.5"
                 style={{ background: "var(--ink-2)" }}>
              <span className="num text-[13px]">
                {int(m.urls_per_arm)} URLs
              </span>
              <div className="h-[10px] relative" style={{ background: "var(--ink-3)" }}>
                <div className="absolute inset-y-0 left-0"
                     style={{ width: `${((m.mde_positions ?? 0) / maxMde) * 100}%`,
                              background: "var(--signal)", opacity: 0.8 }} />
              </div>
              <span className="num text-[13px] text-right">
                {fmt(m.mde_positions ?? 0)} pos
              </span>
            </div>
          ))}
        </div>

        <p className="text-[14.5px] max-w-[74ch] mt-6"
           style={{ color: "var(--paper-dim)" }}>
          The number that sets this table is not day-to-day rank wobble, which
          averages away if you run longer. It is per-URL drift between the two
          periods &mdash; a competitor moved, a link died, the page got
          re-crawled. That does not average out over days. It only averages out
          over URLs. A team that sizes a test off daily variance concludes that
          two weeks and thirty URLs will do, runs it, and gets a confident
          wrong answer.
        </p>
      </section>

      {/* -------------------------------------------------------------- */}
      <section className="grid gap-px lg:grid-cols-2"
               style={{ background: "var(--rule)" }}>
        <div className="p-7" style={{ background: "var(--ink-2)" }}>
          <div className="eyebrow mb-5">Are our intervals honest?</div>
          <Stat
            label="95% interval coverage, 400 replications"
            value={`${(e.coverage * 100).toFixed(1)}%`}
            tone={Math.abs(e.coverage - 0.95) < 0.03 ? "var(--verify)" : "var(--alert)"}
            sub="nominal 95%"
            big
          />
          <p className="text-[13.5px] leading-relaxed mt-6"
             style={{ color: "var(--paper-dim)" }}>
            We ran the test four hundred times against a known truth and
            counted how often the interval contained it. An interval that is
            right seventy percent of the time is worse than no interval at all,
            because it converts uncertainty into false confidence &mdash; which
            is the failure this whole project exists to prevent. Validating the
            validator is not optional.
          </p>
        </div>

        <div className="p-7" style={{ background: "var(--ink-2)" }}>
          <div className="eyebrow mb-5">L1 &mdash; does the model rank like Google?</div>
          <div className="space-y-2.5">
            {heads.rows.map((r) => {
              const isHead = r.model.startsWith("head");
              return (
                <div key={r.model} className="flex items-center gap-3">
                  <span className="text-[12.5px] w-[168px] shrink-0 leading-snug"
                        style={{ color: isHead ? "var(--paper)" : "var(--paper-dim)" }}>
                    {r.model}
                  </span>
                  <div className="flex-1 h-[9px]" style={{ background: "var(--ink-3)" }}>
                    <div className="h-full"
                         style={{ width: `${((r.ndcg10 - 0.55) / 0.3) * 100}%`,
                                  background: isHead ? "var(--signal)" : "var(--muted)" }} />
                  </div>
                  <span className="num text-[12px] w-[42px] text-right">
                    {fmt(r.ndcg10, 3)}
                  </span>
                </div>
              );
            })}
          </div>
          <p className="text-[13px] leading-relaxed mt-5 pt-5 border-t"
             style={{ borderColor: "var(--rule)", color: "var(--paper-dim)" }}>
            NDCG@10 on queries the model has never seen, held out by query and
            never by row. Baselines included because a model is only as
            impressive as the baseline it beats, and beating random is not an
            achievement anyone should pay for.
          </p>
          <p className="text-[13px] leading-relaxed mt-4"
             style={{ color: heads.linear_ok ? "var(--verify)" : "var(--alert)" }}>
            {heads.message}
          </p>
        </div>
      </section>
    </div>
  );
}
