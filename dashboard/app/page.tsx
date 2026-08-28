import Link from "next/link";
import { getLab } from "@/lib/server";
import { fmt, int } from "@/lib/lab";
import { ErrorBar, Legend, Note, Stat } from "@/components/ui";

export default function Verdict() {
  const lab = getLab();
  const s = lab.separation;
  const naive = s.pilot.find((p) => p.pipeline === "naive")!;
  const corrected = s.pilot.find((p) => p.pipeline === "corrected")!;

  return (
    <div className="space-y-16">
      {/* ---------------------------------------------------------------- */}
      <section className="rise">
        <div className="eyebrow mb-4">The question, answered</div>
        <h1 className="display text-[clamp(34px,5.4vw,60px)] max-w-[20ch] mb-7">
          &ldquo;This factor is 0.84 and this one is 0.73. Is that accurate?
          How would we test it?&rdquo;
        </h1>
        <p className="text-[17px] max-w-[68ch] leading-relaxed"
           style={{ color: "var(--paper-dim)" }}>
          You cannot check a weight against Google, because Google will not
          show you its weights. You can check it against a search engine whose
          weights you set yourself. So this lab builds one, runs the entire
          pipeline against it, and reports where the pipeline gets the answer
          wrong. Whatever error it makes on a world it cannot cheat at is the
          floor on the error it makes on the real one.
        </p>
        <p className="text-[17px] max-w-[68ch] leading-relaxed mt-4">
          In that synthetic world,{" "}
          <span style={{ color: "var(--truth)" }}>
            those two numbers are literally the truth
          </span>
          : content depth is set to {fmt(s.true_a)} and site authority to{" "}
          {fmt(s.true_b)}. Here is what two pipelines say about them.
        </p>
      </section>

      {/* ---------------------------------------------------------------- */}
      <section className="grid gap-px md:grid-cols-2"
               style={{ background: "var(--rule)" }}>
        {[
          {
            k: "The pipeline every SEO tool ships",
            tone: "var(--alert)",
            row: naive,
            verdict: naive.separated
              ? `Reports a clear winner. Says ${int(naive.n_required)} queries were enough.`
              : "Reports no difference.",
            kicker:
              "It is not merely wrong about the sizes. It has the ranking " +
              "backwards, and it says so with a tight interval. Confidently " +
              "wrong is worse than unsure.",
          },
          {
            k: "The pipeline after three corrections",
            tone: "var(--verify)",
            row: corrected,
            verdict: corrected.separated
              ? "Separates them."
              : `Refuses to call it. Prices the answer at ~${int(corrected.n_required)} queries.`,
            kicker:
              "Position-bias correction, an errors-in-variables adjustment, " +
              "and intervals bootstrapped over whole SERPs rather than rows.",
          },
        ].map((c) => (
          <div key={c.k} className="p-7" style={{ background: "var(--ink-2)" }}>
            <div className="eyebrow mb-5">{c.k}</div>
            <div className="space-y-5">
              {[
                { label: s.label_a, est: c.row.est_a, truth: s.true_a },
                { label: s.label_b, est: c.row.est_b, truth: s.true_b },
              ].map((f) => (
                <div key={f.label}>
                  <div className="flex items-baseline justify-between mb-1.5">
                    <span className="text-[13.5px]">{f.label}</span>
                    <span className="num text-[19px]" style={{ color: c.tone }}>
                      {fmt(f.est)}
                    </span>
                  </div>
                  <ErrorBar
                    lo={f.est} hi={f.est} point={f.est} truth={f.truth}
                    tone={c.tone} max={1.15} height={16}
                  />
                  <div className="mono text-[10px] mt-1"
                       style={{ color: "var(--paper-faint)" }}>
                    truth {fmt(f.truth)}
                  </div>
                </div>
              ))}
            </div>
            <div className="mt-6 pt-5 border-t" style={{ borderColor: "var(--rule)" }}>
              <div className="text-[14px] mb-2" style={{ color: c.tone }}>
                {c.verdict}
              </div>
              <p className="text-[13px] leading-relaxed"
                 style={{ color: "var(--paper-dim)" }}>
                {c.kicker}
              </p>
            </div>
          </div>
        ))}
      </section>

      {/* ---------------------------------------------------------------- */}
      <section>
        <Note>
          <strong style={{ color: "var(--paper)" }}>
            The most valuable output of the second pipeline is the sentence
            &ldquo;I cannot tell yet, and here is what telling would cost.&rdquo;
          </strong>{" "}
          Every competitor prints a number. Being the only firm that says when
          the number is not yet a finding &mdash; and quotes the panel size
          that would make it one &mdash; is a position no one else is
          occupying.
        </Note>
      </section>

      {/* ---------------------------------------------------------------- */}
      {s.scale && (
        <section>
          <div className="eyebrow mb-4">
            The prediction, tested
          </div>
          <div className="panel p-7">
            <p className="text-[15px] max-w-[70ch] mb-7"
               style={{ color: "var(--paper-dim)" }}>
              At {int(corrected.n_queries)} queries the corrected pipeline said
              it needed about {int(s.scale.required)} to make the call. So we
              ran it at {int(s.scale.n_queries)}.
            </p>
            <div className="grid gap-8 sm:grid-cols-3">
              <Stat
                label={s.label_a}
                value={fmt(s.scale.est_a)}
                sub={`truth ${fmt(s.true_a)}`}
                tone="var(--signal)"
              />
              <Stat
                label={s.label_b}
                value={fmt(s.scale.est_b)}
                sub={`truth ${fmt(s.true_b)}`}
                tone="var(--signal)"
              />
              <Stat
                label="difference, 95% interval"
                value={`${s.scale.diff_lo > 0 ? "+" : ""}${fmt(s.scale.diff_lo)} to ${fmt(s.scale.diff_hi)}`}
                sub={
                  s.scale.separated
                    ? "clears zero -- the two factors are now separable"
                    : "still spans zero -- the pilot estimate was optimistic"
                }
                tone={s.scale.separated ? "var(--verify)" : "var(--alert)"}
              />
            </div>
          </div>
        </section>
      )}

      {/* ---------------------------------------------------------------- */}
      <section>
        <div className="eyebrow mb-5">
          Three things people mean by &ldquo;importance&rdquo;
        </div>
        <div className="grid gap-px sm:grid-cols-3"
             style={{ background: "var(--rule)" }}>
          {[
            ["Predictive", "How much does knowing this help me reproduce Google's order?",
             "Permutation importance, bootstrapped. What 0.84 usually means."],
            ["Causal", "If I change this on a page, how many positions does it move?",
             "Needs matching, or better, an experiment. Usually a smaller number."],
            ["Actionable", "Effect size divided by what it costs to change.",
             "The only one of the three that belongs in a client deck."],
          ].map(([t, q, a], i) => (
            <div key={t} className="p-6" style={{ background: "var(--ink-2)" }}>
              <div className="num text-[11px] mb-3"
                   style={{ color: "var(--signal)" }}>
                0{i + 1}
              </div>
              <div className="text-[15px] mb-2">{t}</div>
              <p className="text-[13px] leading-relaxed mb-3"
                 style={{ color: "var(--paper-dim)" }}>{q}</p>
              <p className="text-[12.5px] leading-relaxed"
                 style={{ color: "var(--paper-faint)" }}>{a}</p>
            </div>
          ))}
        </div>
        <p className="text-[14px] mt-5 max-w-[70ch]"
           style={{ color: "var(--paper-dim)" }}>
          A tool that prints 0.84 without saying which of the three it means
          is selling a horoscope. The three routinely disagree with each other
          by a factor of two, and only the last one has a budget attached.
        </p>
      </section>

      {/* ---------------------------------------------------------------- */}
      <section>
        <div className="flex items-baseline justify-between mb-5 flex-wrap gap-3">
          <div className="eyebrow">Where to look next</div>
          <Legend />
        </div>
        <div className="grid gap-px sm:grid-cols-2 lg:grid-cols-3"
             style={{ background: "var(--rule)" }}>
          {[
            ["/calibration", "01", "Calibration",
             "Every factor, its true weight, and what the pipeline recovered. Including the two it gets wrong."],
            ["/factors", "02", "Factor bench",
             `All ${lab.meta.n_factors} factors with evidence tier and unit cost, plus the ones we refuse to track.`],
            ["/clusters", "03", "Intent clusters",
             `Backlinks matter ${fmt(lab.clusters.headline.measured_ratio, 1)}x more in one vertical than another. A global table hides that.`],
            ["/simulator", "04", "Simulator",
             "Move a factor, see the predicted rank change with its uncertainty band."],
            ["/experiments", "05", "Experiments",
             "Split tests, minimum detectable effect, and whether the intervals are honest."],
          ].map(([href, n, title, desc]) => (
            <Link key={href} href={href}
                  className="p-6 group transition-colors hover:bg-[var(--ink-3)]"
                  style={{ background: "var(--ink-2)" }}>
              <div className="flex items-baseline gap-2 mb-3">
                <span className="num text-[10px]"
                      style={{ color: "var(--signal)" }}>{n}</span>
                <span className="text-[16px] group-hover:text-[var(--signal)]
                                 transition-colors">{title}</span>
              </div>
              <p className="text-[13px] leading-relaxed"
                 style={{ color: "var(--paper-dim)" }}>{desc}</p>
            </Link>
          ))}
        </div>
      </section>
    </div>
  );
}
