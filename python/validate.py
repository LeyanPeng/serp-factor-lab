"""
The validation ladder. This is the part that answers "is 0.84 correct?".

L0  Is the number stable?     cluster bootstrap + a permutation noise floor
L1  Does it rank like Google? group-held-out NDCG@10 against real baselines
L2  Does it travel?           unseen clusters, unseen time window   (run_demo)
L3  Is it causal?             fixed effects, matching, DiD          (causal.py)
L4  Does it survive contact?  forward prediction and split tests    (causal.py)

Two things here are non-negotiable and almost never done in this industry:

  * The bootstrap resamples QUERIES, not rows. Ten results from one SERP are
    not ten independent observations, and treating them as such shrinks every
    confidence interval by roughly the square root of ten.

  * Every point estimate is compared against a permutation noise floor. Fit
    the same model to deliberately scrambled orderings and you still get
    non-zero weights. Any factor that does not clear that floor is not a
    finding, it is the floor.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.stats import norm

import model


# ---------------------------------------------------------------------------
# Corrections that turn the naive pipeline into the corrected one
# ---------------------------------------------------------------------------

def apply_ipw(df: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    """Replace the naive residual-CTR column with the IPW-corrected signal."""
    eta_hat, _ = model.estimate_examination_curve(df)
    out = df.copy()
    out["residual_ctr"] = model.ipw_click_feature(df, eta_hat)
    return out, eta_hat


def eiv_correct(beta: np.ndarray, X: np.ndarray,
                reliability: dict[str, float], features: list[str]) -> np.ndarray:
    """
    Errors-in-variables disattenuation.

    A feature measured with noise gets its credit stolen by whatever
    correlated feature happens to be measured cleanly. That is not a
    hypothetical: it is how a factor with a true weight of exactly zero ends
    up outranking the factor that actually causes the ranking.

    With observed x = x* + u and independent u,  Sigma_x = Sigma_x* + Sigma_u,
    so  beta* = (Sigma_x - Sigma_u)^-1 Sigma_x beta_obs.

    Sigma_u comes from a re-measurement audit: hand-extract the main content
    of a few hundred pages, compare against the automatic extractor, and the
    reliability of each instrument falls out. That audit costs an afternoon
    and is the highest-return thing on this entire list.
    """
    if not reliability:
        return beta
    Sx = np.cov(X, rowvar=False)
    Su = np.zeros_like(Sx)
    for f, rel in reliability.items():
        if f in features:
            i = features.index(f)
            Su[i, i] = (1.0 - rel) * Sx[i, i]
    return np.linalg.solve(Sx - Su, Sx @ beta)


# ---------------------------------------------------------------------------
# L0 -- stability
# ---------------------------------------------------------------------------

@dataclass
class BootResult:
    features: list[str]
    point: pd.DataFrame          # factor, coef, share, index
    boot_index: np.ndarray       # (n_boot, K)
    boot_share: np.ndarray       # (n_boot, K)
    null_coef: np.ndarray        # (n_perm, K), absolute coefficient scale
    ref: int                     # factor the index is measured against
    n_queries: int
    n_boot: int
    eta_hat: float | None = None
    meta: dict = field(default_factory=dict)

    def ci(self, level: float = 0.95, which: str = "index"):
        a = (1 - level) / 2 * 100
        arr = self.boot_index if which == "index" else self.boot_share
        return np.percentile(arr, a, axis=0), np.percentile(arr, 100 - a, axis=0)

    def null_threshold(self, pct: float = 95.0) -> np.ndarray:
        """
        Noise floor on the RAW coefficient scale. It has to be raw: under a
        permutation every coefficient is noise, so one of them is always the
        largest, and a max-normalised floor would sit at 1.0 for every factor
        and reject nothing.
        """
        return np.percentile(self.null_coef, pct, axis=0)

    def clears_noise(self, pct: float = 95.0) -> np.ndarray:
        return np.abs(self.point["coef"].to_numpy()) > self.null_threshold(pct)


def bootstrap_weights(
    df: pd.DataFrame,
    features: list[str],
    n_boot: int = 400,
    n_perm: int = 200,
    max_pairs: int | None = 15,
    reliability: dict[str, float] | None = None,
    use_ipw: bool = True,
    ridge: float = 1e-6,
    ref_factor: str = "semantic_sim",
    seed: int = 0,
) -> BootResult:
    rng = np.random.default_rng(seed)
    eta_hat = None
    if use_ipw:
        df, eta_hat = apply_ipw(df)

    d = model.standardize(df, features)
    X = d[features].to_numpy()
    pairs = model.build_pairs(d, features, max_pairs=max_pairs, seed=seed)

    beta = model.fit_pairwise(pairs.X, ridge=ridge)
    beta = eiv_correct(beta, X, reliability or {}, features)
    # The denominator is a NAMED factor, not whichever one happens to come out
    # largest. Letting argmax pick it means the reference can change between a
    # corrected and an uncorrected run, or between two seeds, and the factor
    # that wins is pinned to exactly 1.0 with a zero-width interval -- an
    # artefact that reads like certainty.
    ref = features.index(ref_factor)
    point = model.importances(beta, features, ref=ref)

    # --- cluster bootstrap: resample whole SERPs, never individual rows -----
    boot_i = np.empty((n_boot, len(features)))
    boot_s = np.empty((n_boot, len(features)))
    for b in range(n_boot):
        idx = rng.integers(0, pairs.n_queries, pairs.n_queries)
        bb = model.fit_pairwise(pairs.resample(idx), ridge=ridge,
                                beta0=beta, max_iter=3)
        bb = eiv_correct(bb, X, reliability or {}, features)
        a = np.abs(bb)
        boot_i[b] = a / max(a[ref], 1e-12)
        boot_s[b] = a / a.sum()

    # --- permutation noise floor -------------------------------------------
    # Randomly flipping which document in a pair is "better" is exactly a
    # random re-ordering inside each SERP. Whatever weight survives that is
    # the weight the method invents out of nothing.
    null_c = np.empty((n_perm, len(features)))
    for b in range(n_perm):
        signs = rng.choice([-1.0, 1.0], size=pairs.X.shape[0])
        bb = model.fit_pairwise(pairs.X * signs[:, None], ridge=ridge, max_iter=6)
        null_c[b] = np.abs(bb)

    return BootResult(features=features, point=point, boot_index=boot_i,
                      boot_share=boot_s, null_coef=null_c, ref=ref,
                      n_queries=pairs.n_queries, n_boot=n_boot, eta_hat=eta_hat,
                      meta=dict(max_pairs=max_pairs,
                                pairs_per_query=pairs.pairs_per_query))


# ---------------------------------------------------------------------------
# Verdicts
# ---------------------------------------------------------------------------

def verdicts(res: BootResult, true_index: dict[str, float]) -> pd.DataFrame:
    lo, hi = res.ci()
    thr = res.null_threshold()
    clears = res.clears_noise()
    rows = []
    for k, f in enumerate(res.features):
        t = true_index.get(f, np.nan)
        est = res.point["index"].iloc[k]
        above_noise = bool(clears[k])
        if t == 0:
            verdict = "FALSE POSITIVE" if above_noise else "CLEAN"
            why = ("estimator invented weight for a factor with a true weight "
                   "of zero" if above_noise else "correctly found nothing")
        elif not above_noise:
            verdict = "NOT DETECTED"
            why = "point estimate does not clear the permutation noise floor"
        elif lo[k] <= t <= hi[k]:
            verdict = "RECOVERED"
            why = "95% interval covers the truth"
        elif hi[k] < t:
            verdict = "MISSED"
            why = "understated -- interval sits entirely below the truth"
        else:
            verdict = "INFLATED"
            why = "overstated -- interval sits entirely above the truth"
        rows.append(dict(factor=f, true=t, est=est, lo=lo[k], hi=hi[k],
                         coef=res.point["coef"].iloc[k], noise_floor=thr[k],
                         verdict=verdict, why=why))
    return pd.DataFrame(rows)


def grouped_index(res: BootResult, bundle: list[str]) -> tuple[float, float, float]:
    """
    Summed importance of a correlated bundle, with its own interval.

    When two factors are collinear, their individual coefficients are close to
    meaningless but their sum is stable. Reporting the bundle is the honest
    move, and it is what the dashboard shows when the correlation inside a
    group crosses 0.7.
    """
    idx = [res.features.index(f) for f in bundle]
    tot = res.boot_index[:, idx].sum(axis=1)
    return (float(res.point["index"].iloc[idx].sum()),
            float(np.percentile(tot, 2.5)), float(np.percentile(tot, 97.5)))


# ---------------------------------------------------------------------------
# Can we tell 0.84 from 0.73?
# ---------------------------------------------------------------------------

@dataclass
class SeparationTest:
    a: str
    b: str
    est_a: float
    est_b: float
    diff: float
    diff_lo: float
    diff_hi: float
    overlap: float
    separated: bool
    n_queries: int
    n_required: int


def can_we_separate(res: BootResult, a: str, b: str,
                    power: float = 0.80, alpha: float = 0.05) -> SeparationTest:
    """
    The literal question that was asked: factor A scores 0.84 and factor B
    scores 0.73 -- is that a finding?

    We answer it on the bootstrap distribution of the DIFFERENCE, not by
    eyeballing whether two intervals happen to overlap (two overlapping
    intervals can still be a significant difference, and two non-overlapping
    ones are not automatically one).

    When the difference is not significant we report the panel size that would
    make it significant, since standard errors fall as 1/sqrt(queries). That
    number is the actual deliverable: it converts "we do not know" into a
    line item with a price on it.
    """
    ia, ib = res.features.index(a), res.features.index(b)
    da = res.boot_index[:, ia] - res.boot_index[:, ib]
    lo, hi = np.percentile(da, [2.5, 97.5])
    sep = lo > 0 or hi < 0

    ea = float(res.point["index"].iloc[ia])
    eb = float(res.point["index"].iloc[ib])
    delta = ea - eb
    se = float(da.std(ddof=1))

    z = norm.ppf(1 - alpha / 2) + norm.ppf(power)
    need = int(np.ceil(res.n_queries * (se * z / delta) ** 2)) if delta else -1

    # share of the two bootstrap distributions that overlap
    qa = np.percentile(res.boot_index[:, ia], [2.5, 97.5])
    qb = np.percentile(res.boot_index[:, ib], [2.5, 97.5])
    inter = max(0.0, min(qa[1], qb[1]) - max(qa[0], qb[0]))
    union = max(qa[1], qb[1]) - min(qa[0], qb[0])
    overlap = inter / union if union > 0 else 1.0

    return SeparationTest(a=a, b=b, est_a=ea, est_b=eb, diff=delta,
                          diff_lo=float(lo), diff_hi=float(hi), overlap=overlap,
                          separated=bool(sep), n_queries=res.n_queries,
                          n_required=need)


# ---------------------------------------------------------------------------
# L1 -- does it rank like Google?
# ---------------------------------------------------------------------------

def group_split(df: pd.DataFrame, frac: float = 0.7, seed: int = 0):
    """Split by QUERY. Splitting by row leaks the same SERP into both sides."""
    rng = np.random.default_rng(seed)
    q = df["query_id"].unique()
    rng.shuffle(q)
    cut = int(len(q) * frac)
    tr, te = set(q[:cut]), set(q[cut:])
    return (df[df["query_id"].isin(tr)].copy(),
            df[df["query_id"].isin(te)].copy())


def fit_ranking_heads(train: pd.DataFrame, test: pd.DataFrame,
                      features: list[str], reliability=None, use_ipw=True,
                      seed: int = 0) -> pd.DataFrame:
    """
    Both heads plus the baselines anyone would try first. A model is only as
    impressive as the baseline it beats, and "better than random" is not a
    baseline anybody should be paid for.
    """
    if use_ipw:
        eta_hat, _ = model.estimate_examination_curve(train)
        train = train.copy(); test = test.copy()
        train["residual_ctr"] = model.ipw_click_feature(train, eta_hat)
        test["residual_ctr"] = model.ipw_click_feature(test, eta_hat)

    tr = model.standardize(train, features)
    te = model.standardize(test, features)

    pairs = model.build_pairs(tr, features)
    beta = model.fit_pairwise(pairs.X, ridge=1e-6)
    beta = eiv_correct(beta, tr[features].to_numpy(), reliability or {}, features)

    gbm, gbm_name = model.fit_gbm(tr, features, seed=seed)

    rng = np.random.default_rng(seed)
    te["s_random"] = rng.standard_normal(len(te))
    te["s_semantic"] = te["semantic_sim"]
    te["s_links"] = te["ref_domains_log"]
    te["s_linear"] = te[features].to_numpy() @ beta
    te["s_gbm"] = gbm.predict(te[features])

    rows = []
    for col, name in [("s_random", "random ordering"),
                      ("s_semantic", "semantic similarity only"),
                      ("s_links", "referring domains only"),
                      ("s_linear", "head 1: pairwise logistic"),
                      ("s_gbm", f"head 2: {gbm_name}")]:
        rows.append(dict(
            model=name,
            ndcg10=model.ndcg_at_k(te, col, 10),
            pair_acc=model.pairwise_accuracy(te, col),
        ))
    out = pd.DataFrame(rows)
    out["kendall_tau"] = [model.kendall_tau(te.head(4000), c) for c in
                          ["s_random", "s_semantic", "s_links", "s_linear", "s_gbm"]]
    return out


def linearity_verdict(heads: pd.DataFrame, tol: float = 0.03) -> tuple[bool, str]:
    """The rule that decides whether quoting a scalar weight is legitimate."""
    lin = float(heads.loc[heads["model"].str.startswith("head 1"), "ndcg10"].iloc[0])
    gbm = float(heads.loc[heads["model"].str.startswith("head 2"), "ndcg10"].iloc[0])
    gap = (gbm - lin) / gbm
    ok = gap <= tol
    if gap < 0:
        msg = (f"the linear head BEATS the boosted head by {-gap * 100:.1f}% -- "
               f"there is no non-linearity here for trees to find, so scalar "
               f"weights are not just quotable, they are the better model")
    elif ok:
        msg = (f"gap {gap * 100:.1f}% <= {tol * 100:.0f}% -- linear weights "
               f"describe the world, scalars may be quoted")
    else:
        msg = (f"gap {gap * 100:.1f}% > {tol * 100:.0f}% -- the world is "
               f"non-linear, report partial dependence curves, not scalars")
    return ok, msg
