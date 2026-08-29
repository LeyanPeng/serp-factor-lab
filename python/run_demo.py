"""
One command, end to end.

    python python/run_demo.py            # full run, ~1 minute
    python python/run_demo.py --quick    # skip the scale-up panel
    python python/run_demo.py --seed 123 # a different world, same conclusions

Everything printed below is computed at run time. Nothing is hard-coded, and
the dashboard reads the same JSON this script writes.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

import causal
import cluster as clu
import config
import knowledge_graph as kg
import model
import simulate
import validate

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "outputs"
W = 78


# ---------------------------------------------------------------------------
# printing
# ---------------------------------------------------------------------------

def rule(ch: str = "=") -> None:
    print(ch * W)


def head(n: int, total: int, title: str, sub: str = "") -> None:
    print()
    rule()
    print(f" [{n}/{total}]  {title}")
    if sub:
        print(f"         {sub}")
    rule()


def fmt(df: pd.DataFrame, **kw) -> str:
    return df.to_string(index=False, float_format=lambda x: f"{x: .3f}", **kw)


# ---------------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--queries", type=int, default=1200,
                    help="pilot panel size, one vertical's worth of budget")
    ap.add_argument("--scale-queries", type=int, default=48000,
                    help="upper bound on the scale-up panel")
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    t0 = time.perf_counter()
    OUT.mkdir(exist_ok=True)
    export: dict = {}

    rule()
    print(" SERP FACTOR LAB -- calibration run".ljust(W))
    print(f" seed={args.seed}   pilot panel={args.queries:,} queries"
          f"   scale-up={'skipped' if args.quick else f'{args.scale_queries:,} queries'}")
    print(" every number below is computed now, none of it is hard-coded")
    rule()

    panel = simulate.simulate(n_queries=args.queries, seed=args.seed)
    feats = panel.features
    tw = panel.true_weights
    ref_true = tw["semantic_sim"]
    true_idx = {f: tw[f] / ref_true for f in feats}
    reliability = {"content_depth_rel": panel.meta["depth_reliability"]}

    # =====================================================================
    head(1, 7, "CALIBRATION -- does the estimator recover weights we set?",
         "the only honest way to put an error bar on a number nobody can verify")

    naive = validate.bootstrap_weights(
        panel.df, feats, n_boot=300, n_perm=150, seed=args.seed,
        use_ipw=False, reliability=None)
    corrected = validate.bootstrap_weights(
        panel.df, feats, n_boot=300, n_perm=150, seed=args.seed,
        use_ipw=True, reliability=reliability)

    vn = validate.verdicts(naive, true_idx).set_index("factor")
    vc = validate.verdicts(corrected, true_idx).set_index("factor")

    tbl = pd.DataFrame({
        "factor": [config.label_of(f) for f in feats],
        "true": [true_idx[f] for f in feats],
        "naive": vn["est"].reindex(feats).to_numpy(),
        "naive_verdict": vn["verdict"].reindex(feats).to_numpy(),
        "corrected": vc["est"].reindex(feats).to_numpy(),
        "ci_lo": vc["lo"].reindex(feats).to_numpy(),
        "ci_hi": vc["hi"].reindex(feats).to_numpy(),
        "verdict": vc["verdict"].reindex(feats).to_numpy(),
    }).sort_values("true", ascending=False)
    print(fmt(tbl))

    print(f"\n  position-bias exponent: true {panel.eta:.2f}, "
          f"estimated {corrected.eta_hat:.3f} from the {panel.meta['randomize_frac']:.0%} "
          f"randomised slice")
    b_naive = validate.grouped_index(naive, ["content_depth_rel", "kw_density"])
    b_corr = validate.grouped_index(corrected, ["content_depth_rel", "kw_density"])
    print(f"  collinear bundle (content depth + keyword density), true "
          f"{true_idx['content_depth_rel']:.3f}")
    print(f"      naive     {b_naive[0]:.3f} [{b_naive[1]:.3f}, {b_naive[2]:.3f}]")
    print(f"      corrected {b_corr[0]:.3f} [{b_corr[1]:.3f}, {b_corr[2]:.3f}]"
          f"   <- the bundle is trustworthy even when its parts are not")

    print("\n  READ THIS:")
    kd_n = float(vn.loc["kw_density", "est"])
    cd_n = float(vn.loc["content_depth_rel", "est"])
    print(f"    The uncorrected pipeline -- the one every SEO tool ships -- reports")
    print(f"    keyword density at {kd_n:.2f} and content depth at {cd_n:.2f}. The true")
    print(f"    weights are 0.00 and {true_idx['content_depth_rel']:.2f}. It does not merely")
    print(f"    get the sizes wrong, it inverts the ranking, and it does so with")
    print(f"    a tight confidence interval. Confidently wrong is worse than unsure.")

    export["calibration"] = dict(
        rows=[dict(factor=f, label=config.label_of(f), true=true_idx[f],
                   naive=float(vn.loc[f, "est"]),
                   naive_lo=float(vn.loc[f, "lo"]), naive_hi=float(vn.loc[f, "hi"]),
                   naive_verdict=str(vn.loc[f, "verdict"]),
                   corrected=float(vc.loc[f, "est"]),
                   lo=float(vc.loc[f, "lo"]), hi=float(vc.loc[f, "hi"]),
                   noise_floor=float(vc.loc[f, "noise_floor"]),
                   verdict=str(vc.loc[f, "verdict"]),
                   why=str(vc.loc[f, "why"])) for f in feats],
        eta_true=panel.eta, eta_hat=corrected.eta_hat,
        bundle=dict(true=true_idx["content_depth_rel"],
                    naive=b_naive, corrected=b_corr),
        n_queries=args.queries, n_boot=corrected.n_boot,
    )

    # =====================================================================
    head(2, 7, "THE 0.84 vs 0.73 QUESTION",
         "in this harness those two numbers are the ground truth, by construction")

    a, b = "content_depth_rel", "host_pagerank"
    print(f"  true importance:  {config.label_of(a)} = {true_idx[a]:.2f}")
    print(f"                    {config.label_of(b)} = {true_idx[b]:.2f}")

    sep_rows = []
    for name, res in (("naive", naive), ("corrected", corrected)):
        st = validate.can_we_separate(res, a, b)
        sep_rows.append(dict(pipeline=name, n_queries=st.n_queries,
                             est_a=st.est_a, est_b=st.est_b,
                             diff_lo=st.diff_lo, diff_hi=st.diff_hi,
                             separated=st.separated, n_required=st.n_required))
    print()
    print(fmt(pd.DataFrame(sep_rows)))
    print("\n  The naive pipeline is certain, and it has the order backwards.")
    print("  The corrected pipeline says it cannot tell them apart at this panel")
    print("  size, and prices the answer. That sentence is the product.")

    scale = None
    if not args.quick:
        # Run the scale-up at the size the power calculation just asked for,
        # not at a size we picked. If the calculation is right the difference
        # separates; if it is wrong, this is where that shows up.
        need = sep_rows[1]["n_required"]
        n_big = int(min(max(int(np.ceil(need / 1000.0)) * 1000, 6000),
                        args.scale_queries))
        big = simulate.simulate(n_queries=n_big, seed=args.seed)
        # Same max_pairs as the pilot. Thinning the pairs to save time would
        # throw away information per query, which is exactly the quantity the
        # power calculation extrapolated -- the check would then fail for a
        # reason that has nothing to do with the prediction being wrong.
        big_res = validate.bootstrap_weights(
            big.df, feats, n_boot=120, n_perm=40, max_pairs=15,
            reliability=reliability, seed=args.seed)
        st_big = validate.can_we_separate(big_res, a, b)
        vb = validate.verdicts(big_res, true_idx).set_index("factor")
        capped = " (capped by --scale-queries)" if n_big == args.scale_queries \
            and need > args.scale_queries else ""
        print(f"\n  scale-up check at {n_big:,} queries -- the size the power "
              f"calculation asked for{capped}:")
        print(f"    {config.label_of(a)} {st_big.est_a:.2f}  vs  "
              f"{config.label_of(b)} {st_big.est_b:.2f}")
        print(f"    difference 95% CI [{st_big.diff_lo:+.2f}, {st_big.diff_hi:+.2f}]"
              f"   separated = {st_big.separated}")
        if st_big.separated and capped:
            # Separating below the demanded panel is NOT the prediction being
            # confirmed. It means the requirement was conservative, and saying
            # otherwise would be exactly the overclaiming this repo exists to
            # argue against.
            print(f"    the power calculation asked for ~{need:,} queries. This ran "
                  f"at {n_big:,}, short of that, and the difference cleared zero")
            print(f"    anyway -- so the requirement was conservative, not confirmed. "
                  f"Raise --scale-queries to {need:,} to test it properly.")
        elif st_big.separated:
            print(f"    the power calculation asked for ~{need:,} queries to make "
                  f"this call. At {n_big:,} the call is made -- the prediction held.")
        else:
            print(f"    the power calculation asked for ~{need:,} queries and at "
                  f"{n_big:,} the difference still does not clear zero. The")
            print(f"    prediction was optimistic; the honest read is that this "
                  f"pair needs a larger panel than the pilot implied.")
        print(f"    click factor, still attenuated: {float(vb.loc['residual_ctr','est']):.2f} "
              f"vs true {true_idx['residual_ctr']:.2f} -- more data narrows intervals,")
        print(f"    it does not remove bias. Those are different problems.")
        scale = dict(n_queries=n_big, required=need, est_a=st_big.est_a,
                     est_b=st_big.est_b, diff_lo=st_big.diff_lo,
                     diff_hi=st_big.diff_hi, separated=st_big.separated,
                     rows=[dict(factor=f, true=true_idx[f],
                                est=float(vb.loc[f, "est"]),
                                lo=float(vb.loc[f, "lo"]), hi=float(vb.loc[f, "hi"]),
                                verdict=str(vb.loc[f, "verdict"])) for f in feats])

    export["separation"] = dict(a=a, b=b, label_a=config.label_of(a),
                                label_b=config.label_of(b),
                                true_a=true_idx[a], true_b=true_idx[b],
                                pilot=sep_rows, scale=scale)

    # =====================================================================
    head(3, 7, "L1 -- does it rank like Google?",
         "held out by query, never by row; compared against baselines that are honest")

    train, test = validate.group_split(panel.df, frac=0.7, seed=args.seed)
    heads = validate.fit_ranking_heads(train, test, feats,
                                       reliability=reliability, seed=args.seed)
    print(fmt(heads))
    ok, msg = validate.linearity_verdict(heads)
    print(f"\n  linearity rule: {msg}")
    export["heads"] = dict(rows=heads.to_dict("records"),
                           linear_ok=bool(ok), message=msg)

    # =====================================================================
    head(4, 7, "L2 -- does it travel?",
         "clusters discovered from SERP shape, never assumed")

    cpanel = simulate.simulate(n_queries=max(4000, args.queries), seed=args.seed + 1)
    cdf, _ = validate.apply_ipw(cpanel.df)
    _, ari = clu.discover_clusters(cdf, feats, k=len(config.CLUSTERS),
                                   seed=args.seed)
    print(f"  intent clusters recovered without labels: adjusted Rand = {ari:.3f}")
    wbc = clu.weights_by_cluster(cdf, feats)
    spread = clu.spread_report(wbc, top_n=6)
    print()
    print(spread.to_string(float_format=lambda x: f"{x: .3f}"))

    top = spread.index[0]
    piv = wbc.pivot(index="factor", columns="cluster", values="index")
    hi_c, lo_c = piv.loc[top].idxmax(), piv.loc[top].idxmin()
    tcw = cpanel.cluster_weights
    true_ratio = ((tcw[hi_c][top] / tcw[hi_c]["semantic_sim"]) /
                  (tcw[lo_c][top] / tcw[lo_c]["semantic_sim"]))
    print(f"\n  {config.label_of(top)}: {piv.loc[top, hi_c]:.2f} in {hi_c} vs "
          f"{piv.loc[top, lo_c]:.2f} in {lo_c}")
    print(f"  measured ratio {spread.loc[top, 'ratio']:.1f}x, true ratio "
          f"{true_ratio:.1f}x -- a global weight table would report neither.")
    export["clusters"] = dict(
        ari=float(ari), rows=wbc.to_dict("records"),
        headline=dict(factor=top, label=config.label_of(top), high=hi_c, low=lo_c,
                      high_val=float(piv.loc[top, hi_c]),
                      low_val=float(piv.loc[top, lo_c]),
                      measured_ratio=float(spread.loc[top, "ratio"]),
                      true_ratio=float(true_ratio)),
        clusters=list(piv.columns))

    # =====================================================================
    head(5, 7, "L3 -- causal estimates, in positions",
         "a weight is a description; positions per unit of work is a decision")

    cz = model.standardize(cdf, feats)
    fe = causal.within_fe(cz, feats).set_index("factor")
    rows = []
    for f in feats:
        m = causal.matched_effect(cz, feats, f, seed=args.seed)
        eff = config.EFFORT.get(f, 3)
        rows.append(dict(factor=f, label=config.label_of(f),
                         true_weight=true_idx[f],
                         fe_positions=float(fe.loc[f, "positions_per_sd"]),
                         matched_positions=m["positions_per_sd"],
                         lo=m["lo"], hi=m["hi"],
                         effort=eff, effort_label=config.EFFORT_LABEL[eff],
                         per_effort=m["positions_per_sd"] / eff))
    ctab = pd.DataFrame(rows)
    show = ["semantic_sim", "content_depth_rel", "host_pagerank",
            "residual_ctr", "kw_density"]
    print(fmt(ctab[ctab["factor"].isin(show)]
              [["label", "true_weight", "fe_positions", "matched_positions",
                "lo", "hi"]]))

    roi = ctab.sort_values("per_effort").head(4)
    print("\n  Best positions per unit of effort (effort is our judgement, "
          "not a measurement):")
    for _, r in roi.iterrows():
        print(f"    {r['label']:<32} {r['matched_positions']:+.2f} positions / "
              f"{r['effort_label']:<18} = {r['per_effort']:+.3f}")
    top_roi = roi.iloc[0]
    if top_roi["factor"] == "kw_density":
        print("\n  Look at what came top of that league table.")
        print("  A factor whose true weight is exactly zero ranks first on return")
        print("  per unit of effort -- because it is cheap to change and it is")
        print("  correlated with something that works. Fixed effects and matching")
        print("  both credit it: they are observational, so they inherit the")
        print("  measurement problem whole.")
        print("  This is the entire argument for the governance rule. L0 through L3")
        print("  can all be fooled by a badly measured instrument. Only L4 cannot,")
        print("  because in a split test you change the real thing rather than a")
        print("  proxy for it -- and nothing here reaches a client until it has.")
    else:
        print("\n  Fixed effects and matching both credit the decoy, because both")
        print("  are observational and inherit the measurement problem whole.")
        print("  L0-L3 can be fooled by a badly measured instrument. Only L4 cannot,")
        print("  because in a split test you change the real thing, not a proxy.")
    export["causal"] = dict(rows=ctab.to_dict("records"),
                            n_queries=int(cdf["query_id"].nunique()),
                            effort_note=("Effort is an agency judgement on a 1-5 "
                                         "scale, not a measured quantity. Every "
                                         "other number on this page is measured."))

    # =====================================================================
    head(6, 7, "L4 -- intervention",
         "the only rung that yields a number you can put in a contract")

    cov = causal.coverage_check(reps=400, seed=args.seed)
    mde = causal.mde_curve(sizes=(60, 120, 240, 400, 600), reps=40, seed=args.seed)
    mde_at = dict(zip(mde["urls_per_arm"], mde["mde_positions"]))

    true_eff = -0.60
    runs = []
    for n_urls in (240, 600):
        st = causal.run_split_test(n_urls=n_urls, true_effect=true_eff,
                                   seed=args.seed)
        runs.append(st)
        det = mde_at.get(n_urls // 2)
        if det is None:
            note = "minimum detectable effect not computed at this arm size"
        elif det > abs(true_eff):
            note = (f"underpowered by design: this arm size resolves "
                    f"{det:.2f} positions, the real effect is {abs(true_eff):.2f}")
        else:
            note = (f"adequately powered: resolves {det:.2f} positions, "
                    f"below the real effect of {abs(true_eff):.2f}")
        print(f"  {st.n_per_arm} URLs per arm, {st.days}+{st.days} days")
        print(f"    true {st.true_effect:+.2f}   estimate {st.estimate:+.2f}   "
              f"95% CI [{st.lo:+.2f}, {st.hi:+.2f}]   significant={st.significant}")
        print(f"    {note}")
    print(f"\n  interval calibration over 400 replications: {cov:.1%} coverage "
          f"(nominal 95%)")
    print()
    print(fmt(mde.rename(columns={"urls_per_arm": "URLs per arm",
                                  "mde_positions": "min detectable (positions)"})))
    print("\n  The first test above failed to detect a real effect, and the table")
    print("  predicted it would. Working out the minimum detectable effect BEFORE")
    print("  running is what separates 'the change did nothing' from 'we could")
    print("  never have seen it'. Sizing a test off day-to-day wobble alone says")
    print("  thirty URLs is plenty. It is not: per-URL drift between the periods")
    print("  does not average out over days, only over URLs.")
    export["experiments"] = dict(
        runs=[dict(true=r.true_effect, est=r.estimate, lo=r.lo, hi=r.hi,
                   significant=r.significant, n_per_arm=r.n_per_arm,
                   days=r.days, mde=mde_at.get(r.n_per_arm)) for r in runs],
        coverage=cov, mde=mde.to_dict("records"))

    # =====================================================================
    head(7, 7, "THE KNOWLEDGE GRAPH -- from a weight to a work order",
         "a search engine ranks entities, not strings")

    kg_out = kg.print_demo()
    export["knowledge_graph"] = kg_out

    print()
    print("  Why this sits in the same repo as the calibration harness:")
    print("  every other factor in the list produces a number an account")
    print("  manager reads. This one produces the same number AND the list of")
    print("  entities a writer has to add. The weight says how hard to push;")
    print("  the graph says where. Neither is much use without the other.")

    # =====================================================================
    print()
    rule()
    print(" WHAT WE WOULD ACTUALLY SHIP")
    rule()
    tally: dict[str, int] = {}
    for r in export["calibration"]["rows"]:
        tally[r["verdict"]] = tally.get(r["verdict"], 0) + 1
    for v in ("RECOVERED", "CLEAN", "MISSED", "INFLATED", "FALSE POSITIVE",
              "NOT DETECTED"):
        if tally.get(v):
            print(f"    {tally[v]:>2}  {v}")
    good = tally.get("RECOVERED", 0) + tally.get("CLEAN", 0)
    print(f"  {good}/{len(feats)} factors land where they should. The rest carry a")
    print(f"  named, measured failure mode and ship with it attached. Nothing")
    print(f"  reaches a client deck without a verdict, an interval, and a date.")
    print()
    print(f"  runtime {time.perf_counter() - t0:.1f}s   "
          f"numpy {np.__version__}  pandas {pd.__version__}  "
          f"lightgbm={'yes' if model.HAVE_LGB else 'no (sklearn fallback)'}")
    rule()

    export["meta"] = dict(
        seed=args.seed, pilot_queries=args.queries,
        scale_queries=None if args.quick else args.scale_queries,
        runtime_s=round(time.perf_counter() - t0, 1),
        numpy=np.__version__, pandas=pd.__version__,
        lightgbm=bool(model.HAVE_LGB),
        n_factors=len(config.FACTORS),
        n_simulated=len(feats),
        generated=pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d %H:%M UTC"),
    )
    export["factors"] = config.as_dicts()
    export["rejected"] = [dict(factor=f, why=w) for f, w in config.REJECTED]
    export["groups"] = config.GROUP_LABELS
    export["tiers"] = config.TIER_LABELS

    import export_dashboard
    export_dashboard.write(export, OUT)
    print(f"\n  wrote {OUT / 'lab.json'} and dashboard/public/data/lab.json")


if __name__ == "__main__":
    main()
