import { getLab } from "@/lib/server";
import { fmt, int } from "@/lib/lab";
import CalibrationTable from "@/components/CalibrationTable";
import { Note, SectionTitle, Stat } from "@/components/ui";

export default function Calibration() {
  const lab = getLab();
  const c = lab.calibration;
  const decoy = c.rows.find((r) => r.factor === "kw_density")!;
  const depth = c.rows.find((r) => r.factor === "content_depth_rel")!;
  const ctr = c.rows.find((r) => r.factor === "residual_ctr")!;

  return (
    <div className="space-y-14">
      <SectionTitle
        eyebrow="01 / Calibration"
        title="Does the estimator recover weights we set ourselves?"
        lede={`A synthetic search engine ranks ${int(c.n_queries)} SERPs using a weight
               table we wrote. It is deliberately unkind: it is a pipeline rather than
               a function, most of its score comes from signals we cannot measure, the
               click data is only visible through position bias, and content depth is
               measured badly while a worthless decoy beside it is measured cleanly.
               Then we run the whole estimation stack at it and mark our own work.`}
      />

      <section className="grid gap-8 sm:grid-cols-3">
        <Stat
          label="Position-bias exponent"
          value={fmt(c.eta_hat, 3)}
          sub={`true value ${fmt(c.eta_true, 2)}, recovered from a 5% randomised slice`}
          tone="var(--verify)"
        />
        <Stat
          label="Bootstrap resamples"
          value={int(c.n_boot)}
          sub="resampling whole SERPs, never individual rows"
        />
        <Stat
          label="Decoy, uncorrected"
          value={fmt(decoy.naive)}
          sub="true weight is exactly zero"
          tone="var(--alert)"
        />
      </section>

      <section>
        <CalibrationTable rows={c.rows} />
      </section>

      {/* -------------------------------------------------------------- */}
      <section className="grid gap-px lg:grid-cols-2"
               style={{ background: "var(--rule)" }}>
        <div className="p-7" style={{ background: "var(--ink-2)" }}>
          <div className="eyebrow mb-4">Failure one &mdash; the invented factor</div>
          <p className="text-[14.5px] leading-relaxed mb-4"
             style={{ color: "var(--paper-dim)" }}>
            Keyword density has a true weight of{" "}
            <span style={{ color: "var(--truth)" }}>exactly zero</span> and is
            0.85-correlated with content depth. Uncorrected, the model gives it{" "}
            <span className="num" style={{ color: "var(--alert)" }}>
              {fmt(decoy.naive)}
            </span>{" "}
            and gives real content depth only{" "}
            <span className="num" style={{ color: "var(--alert)" }}>
              {fmt(depth.naive)}
            </span>
            .
          </p>
          <p className="text-[14.5px] leading-relaxed mb-4">
            The mechanism is not collinearity, it is measurement. Depth is hard
            to extract cleanly &mdash; boilerplate, navigation, pagination. The
            decoy is trivial to measure. A regression credits whichever
            variable carries the cleanest signal, not whichever one causes the
            outcome.
          </p>
          <p className="text-[14.5px] leading-relaxed"
             style={{ color: "var(--paper-dim)" }}>
            After the errors-in-variables adjustment the decoy falls to{" "}
            <span className="num" style={{ color: "var(--verify)" }}>
              {fmt(decoy.corrected)}
            </span>{" "}
            and depth rises to{" "}
            <span className="num" style={{ color: "var(--verify)" }}>
              {fmt(depth.corrected)}
            </span>
            . The input that fixes it is a re-measurement audit: hand-extract
            three hundred pages, compare against the crawler, and the
            reliability of the instrument falls out. That is an afternoon of
            work and it is the highest-return item in this entire project.
          </p>
        </div>

        <div className="p-7" style={{ background: "var(--ink-2)" }}>
          <div className="eyebrow mb-4">Failure two &mdash; the vanished factor</div>
          <p className="text-[14.5px] leading-relaxed mb-4"
             style={{ color: "var(--paper-dim)" }}>
            The click signal has a true weight of{" "}
            <span className="num" style={{ color: "var(--truth)" }}>
              {fmt(ctr.true)}
            </span>
            . Uncorrected it reads{" "}
            <span className="num" style={{ color: "var(--alert)" }}>
              {fmt(ctr.naive)}
            </span>
            .
          </p>
          <p className="text-[14.5px] leading-relaxed mb-4">
            The usual construction &mdash; observed clicks minus the average at
            that position &mdash; looks like it removes position bias. It does
            the opposite. Position is <em>caused</em> by quality, so
            subtracting the positional mean subtracts the quality along with
            the bias.
          </p>
          <p className="text-[14.5px] leading-relaxed"
             style={{ color: "var(--paper-dim)" }}>
            Dividing by an estimated examination probability instead lifts it
            to{" "}
            <span className="num" style={{ color: "var(--signal)" }}>
              {fmt(ctr.corrected)}
            </span>{" "}
            &mdash; better, still short. That residual gap is honest: click
            noise is irreducible, and no amount of extra data closes it. We
            ship the factor with the gap attached rather than pretending it is
            not there.
          </p>
        </div>
      </section>

      {/* -------------------------------------------------------------- */}
      <section>
        <div className="eyebrow mb-4">
          What to report when two factors are entangled
        </div>
        <div className="panel p-7">
          <p className="text-[15px] max-w-[74ch] mb-7"
             style={{ color: "var(--paper-dim)" }}>
            Content depth and keyword density are two measurements of one
            underlying thing. Split between them the individual numbers are
            close to meaningless; added together they are stable. When the
            correlation inside a group crosses 0.7 the honest unit of
            reporting is the bundle, not its parts.
          </p>
          <div className="grid gap-8 sm:grid-cols-3">
            <Stat label="Truth" value={fmt(c.bundle.true)}
                  tone="var(--truth)" sub="the weight actually assigned" />
            <Stat label="Bundle, uncorrected"
                  value={fmt(c.bundle.naive[0])}
                  tone="var(--alert)"
                  sub={`95% interval ${fmt(c.bundle.naive[1])} to ${fmt(c.bundle.naive[2])}`} />
            <Stat label="Bundle, corrected"
                  value={fmt(c.bundle.corrected[0])}
                  tone="var(--verify)"
                  sub={`95% interval ${fmt(c.bundle.corrected[1])} to ${fmt(c.bundle.corrected[2])}`} />
          </div>
        </div>
      </section>

      <Note>
        <strong style={{ color: "var(--paper)" }}>
          More data does not fix any of this.
        </strong>{" "}
        Run the uncorrected pipeline at 1,200 queries and at 15,000 and it
        returns the same wrong answer with a narrower interval. Bias and
        variance are different problems and only one of them has a budget
        solution. Buying a bigger panel to fix a broken instrument is the most
        expensive way to stay wrong.
      </Note>
    </div>
  );
}
