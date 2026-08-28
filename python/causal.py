"""
L3 and L4 -- the part that turns a weight into a decision.

A weight of 0.84 answers "how much does knowing this help me reproduce
Google's order". A client does not care. A client cares about "if I spend
three weeks on this, how many positions do I move". Those are different
questions with different answers, and conflating them is the single most
expensive mistake in this industry.

L3  observational causal estimates
    - within-query fixed effects: the same query, so query-level confounding
      is differenced away
    - near-twin matching: pages that are similar on everything else, so the
      remaining rank gap is attributable to the one factor we varied
    - difference-in-differences across a core update: which factor's weight
      actually moved, rather than which client happened to complain loudest

L4  intervention
    - a proper SEO split test on client URLs, analysed against a control
      cohort, with the minimum detectable effect worked out in advance

The split test is the only rung that produces a number you can put in a
contract. Everything above it is evidence for what to test next.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

import model


# ---------------------------------------------------------------------------
# L3a -- within-query fixed effects
# ---------------------------------------------------------------------------

def within_fe(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    Regress rank on features after subtracting each query's own mean.

    The within transformation removes every query-level confounder there is,
    observed or not: how commercial the query is, how many ads it carries,
    how authoritative the whole result set happens to be. What it cannot
    remove is confounding between two documents on the same SERP -- for that
    we need the matching below.

    Standard errors are clustered on query, because the ten rows from one
    SERP are one observation, not ten.
    """
    d = df.sort_values(["query_id", "rank"], kind="stable")
    X = d[features].to_numpy(dtype=float)
    y = d["rank"].to_numpy(dtype=float)
    q = d["query_id"].to_numpy()

    n_q = len(np.unique(q))
    D = len(d) // n_q
    Xg = X.reshape(n_q, D, -1)
    yg = y.reshape(n_q, D)
    Xw = (Xg - Xg.mean(axis=1, keepdims=True)).reshape(-1, X.shape[1])
    yw = (yg - yg.mean(axis=1, keepdims=True)).ravel()

    XtX_inv = np.linalg.pinv(Xw.T @ Xw)
    beta = XtX_inv @ (Xw.T @ yw)
    resid = yw - Xw @ beta

    # cluster-robust (query-clustered) sandwich
    g = (Xw * resid[:, None]).reshape(n_q, D, -1).sum(axis=1)
    meat = g.T @ g
    V = XtX_inv @ meat @ XtX_inv
    se = np.sqrt(np.diag(V))

    return pd.DataFrame({
        "factor": features,
        "positions_per_sd": beta,      # negative = moves the page UP
        "se": se,
        "lo": beta - 1.96 * se,
        "hi": beta + 1.96 * se,
    })


# ---------------------------------------------------------------------------
# L3b -- near-twin matching
# ---------------------------------------------------------------------------

def matched_effect(df: pd.DataFrame, features: list[str], target: str,
                   n_keep: float = 0.15, seed: int = 0) -> dict:
    """
    Compare pages on the same SERP that are near-identical on every factor
    except the one we care about, then read off the rank gap.

    This is the closest an observational study gets to an experiment, and it
    is what we run before spending a client's budget on an actual test. The
    matching is exact on query by construction -- we only ever compare two
    results from the same SERP.
    """
    others = [f for f in features if f != target]
    d = df.sort_values(["query_id", "rank"], kind="stable")
    n_q = d["query_id"].nunique()
    D = len(d) // n_q

    Xo = d[others].to_numpy(dtype=float).reshape(n_q, D, -1)
    xt = d[target].to_numpy(dtype=float).reshape(n_q, D)
    rk = d["rank"].to_numpy(dtype=float).reshape(n_q, D)

    ii, jj = np.triu_indices(D, k=1)
    dist = np.linalg.norm(Xo[:, ii, :] - Xo[:, jj, :], axis=2).ravel()
    dx = (xt[:, ii] - xt[:, jj]).ravel()
    dr = (rk[:, ii] - rk[:, jj]).ravel()

    cut = np.quantile(dist, n_keep)
    m = (dist <= cut) & (np.abs(dx) > 1e-9)
    dx_m, dr_m = dx[m], dr[m]

    slope = float(np.sum(dx_m * dr_m) / np.sum(dx_m ** 2))

    rng = np.random.default_rng(seed)
    boots = []
    for _ in range(300):
        s = rng.integers(0, dx_m.size, dx_m.size)
        boots.append(np.sum(dx_m[s] * dr_m[s]) / np.sum(dx_m[s] ** 2))
    lo, hi = np.percentile(boots, [2.5, 97.5])

    return dict(factor=target, positions_per_sd=slope, lo=float(lo),
                hi=float(hi), n_matched_pairs=int(dx_m.size),
                match_radius=float(cut))


# ---------------------------------------------------------------------------
# L3c -- difference-in-differences across a core update
# ---------------------------------------------------------------------------

def did_update(before: pd.DataFrame, after: pd.DataFrame, features: list[str],
               n_boot: int = 200, seed: int = 0) -> pd.DataFrame:
    """
    Fit the same model either side of an update and ask which weights moved.

    This is how an update gets diagnosed in hours instead of by reading three
    weeks of forum speculation. The bootstrap is what stops us from
    announcing a change every time the panel wobbles.
    """
    rng = np.random.default_rng(seed)
    out = {}
    boots = {}
    for name, d in (("before", before), ("after", after)):
        s = model.standardize(d, features)
        pairs = model.build_pairs(s, features, max_pairs=15, seed=seed)
        beta = model.fit_pairwise(pairs.X, ridge=1e-6)
        ref = features.index("semantic_sim")
        out[name] = np.abs(beta) / max(abs(beta[ref]), 1e-12)
        bs = np.empty((n_boot, len(features)))
        for b in range(n_boot):
            idx = rng.integers(0, pairs.n_queries, pairs.n_queries)
            bb = model.fit_pairwise(pairs.resample(idx), ridge=1e-6,
                                    beta0=beta, max_iter=3)
            bs[b] = np.abs(bb) / max(abs(bb[ref]), 1e-12)
        boots[name] = bs

    diff = boots["after"] - boots["before"]
    lo, hi = np.percentile(diff, [2.5, 97.5], axis=0)
    res = pd.DataFrame({
        "factor": features,
        "before": out["before"],
        "after": out["after"],
        "change": out["after"] - out["before"],
        "lo": lo, "hi": hi,
    })
    res["moved"] = (res["lo"] > 0) | (res["hi"] < 0)
    return res.sort_values("change", key=np.abs, ascending=False)


# ---------------------------------------------------------------------------
# L4 -- the split test
# ---------------------------------------------------------------------------

@dataclass
class SplitTestResult:
    true_effect: float
    estimate: float
    lo: float
    hi: float
    covered: bool
    significant: bool
    n_per_arm: int
    days: int


def run_split_test(n_urls: int = 240, days: int = 28, true_effect: float = -0.60,
                   url_sd: float = 1.8, day_sd: float = 0.55,
                   noise_sd: float = 0.9, url_drift_sd: float = 1.75,
                   effect_het_sd: float = 0.45, seed: int = 0) -> SplitTestResult:
    """
    A page-level SEO A/B test, the way it actually has to be run.

    Two cohorts of comparable URLs, one treated. Both cohorts ride the same
    market-wide daily shocks -- seasonality, competitor launches, core
    updates -- which is exactly why a before/after comparison on the treated
    cohort alone is worthless and a control cohort is not optional.

    The estimator is difference-in-differences with URL and day fixed
    effects. That is the same identification a Bayesian structural time
    series gets you; BSTS buys tighter intervals from better priors, not a
    different assumption.

    Two noise terms decide how big this test has to be, and only one of them
    is the obvious one:

      noise_sd      day-to-day rank wobble. Averages away over the window,
                    so running longer helps.
      url_drift_sd  each URL's own shift between the two periods, for its own
                    reasons -- a competitor moved, a link died, the page got
                    re-crawled. This does NOT average away over days. More
                    days buy you nothing against it. Only more URLs do.

    Teams that size a split test off daily variance alone conclude they need
    two weeks and thirty URLs, run it, and get a confident wrong answer.
    """
    rng = np.random.default_rng(seed)
    n_half = n_urls // 2
    treat = np.zeros(n_urls, dtype=bool)
    treat[:n_half] = True

    url_fe = rng.normal(0, url_sd, n_urls)[:, None]
    day_fe = rng.normal(0, day_sd, 2 * days)[None, :]        # hits both arms
    post = np.zeros(2 * days, dtype=bool)
    post[days:] = True

    # each URL drifts by its own amount between the periods
    drift = rng.normal(0, url_drift_sd, n_urls)[:, None] * post[None, :]
    # and responds to the treatment by its own amount
    per_url_effect = (true_effect + rng.normal(0, effect_het_sd, n_urls))[:, None]

    effect = per_url_effect * (treat[:, None] & post[None, :])
    y = (8.0 + url_fe + day_fe + drift + effect
         + rng.normal(0, noise_sd, (n_urls, 2 * days)))

    pre_t = y[treat][:, ~post].mean(axis=1)
    post_t = y[treat][:, post].mean(axis=1)
    pre_c = y[~treat][:, ~post].mean(axis=1)
    post_c = y[~treat][:, post].mean(axis=1)

    d_t, d_c = post_t - pre_t, post_c - pre_c            # per-URL differences
    est = d_t.mean() - d_c.mean()
    se = np.sqrt(d_t.var(ddof=1) / d_t.size + d_c.var(ddof=1) / d_c.size)
    lo, hi = est - 1.96 * se, est + 1.96 * se

    return SplitTestResult(true_effect=true_effect, estimate=float(est),
                           lo=float(lo), hi=float(hi),
                           covered=bool(lo <= true_effect <= hi),
                           significant=bool(hi < 0 or lo > 0),
                           n_per_arm=n_half, days=days)


def coverage_check(reps: int = 400, seed: int = 0, **kw) -> float:
    """
    Does the 95% interval actually contain the truth 95% of the time?

    Validating the validator. An interval that is only right 70% of the time
    is worse than no interval, because it converts uncertainty into false
    confidence, which is the failure mode this entire lab exists to prevent.
    """
    hits = sum(run_split_test(seed=seed + r, **kw).covered for r in range(reps))
    return hits / reps


def mde_curve(sizes=(40, 80, 120, 200, 320, 500), days: int = 28,
              reps: int = 60, seed: int = 0) -> pd.DataFrame:
    """
    Minimum detectable effect by cohort size, measured rather than asserted.

    We simulate the test at a range of effect sizes and read off the smallest
    one detected at least 80% of the time. This table is what stops us
    promising a client a test that was never going to resolve.
    """
    rows = []
    for n in sizes:
        found = None
        for eff in np.arange(-0.05, -2.51, -0.05):
            hits = sum(run_split_test(n_urls=n, days=days, true_effect=float(eff),
                                      seed=seed + r).significant
                       for r in range(reps))
            if hits / reps >= 0.80:
                found = abs(float(eff))
                break
        rows.append(dict(urls_per_arm=n // 2, mde_positions=found))
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# L4b -- forward prediction
# ---------------------------------------------------------------------------

def forward_test(train: pd.DataFrame, future: pd.DataFrame,
                 features: list[str]) -> dict:
    """
    Freeze the model, then score a panel it has never seen from a later
    period. A model that only describes today is a report; a model that
    survives this is a forecast, and only a forecast is worth acting on.
    """
    tr = model.standardize(train, features)
    pairs = model.build_pairs(tr, features, max_pairs=15)
    beta = model.fit_pairwise(pairs.X, ridge=1e-6)

    fu = model.standardize(future, features)
    fu["s"] = fu[features].to_numpy() @ beta
    return dict(ndcg10=model.ndcg_at_k(fu, "s"),
                pair_acc=model.pairwise_accuracy(fu, "s"),
                n_queries=int(fu["query_id"].nunique()))
