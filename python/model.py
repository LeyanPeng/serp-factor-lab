"""
Two model heads, on purpose, plus the position-bias correction.

Head 1 -- interpretable. A Bradley-Terry / RankNet pairwise logistic fitted on
          WITHIN-QUERY feature differences. Differencing inside a query is
          query fixed effects for free: everything about the query itself
          (how competitive it is, how commercial, how many ads) cancels.
          Its standardised coefficients are the numbers you are allowed to
          quote as "0.84".

Head 2 -- performance. A gradient-boosted ranker. It will out-rank head 1
          whenever the world is non-linear.

The decision rule that connects them:

    if NDCG(head 2) - NDCG(head 1) <= 3% of NDCG(head 2):
        the linear weights describe the world; scalars may be quoted
    else:
        the world is non-linear; report partial dependence curves, not numbers

Nobody in this industry states that rule, which is why they all quote scalars
for models that do not have any.

The IRLS solver is hand-rolled rather than sklearn's because the bootstrap
refits it a few hundred times on a few hundred thousand pairs, and a 12x12
Newton step is roughly two orders of magnitude cheaper than a general-purpose
lbfgs call.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

try:                                  # optional accelerator, never required
    import lightgbm as lgb
    HAVE_LGB = True
except ImportError:                   # pragma: no cover
    lgb = None
    HAVE_LGB = False

from sklearn.ensemble import HistGradientBoostingRegressor


# ---------------------------------------------------------------------------
# Pair construction
# ---------------------------------------------------------------------------

@dataclass
class PairBlocks:
    """
    Within-query feature differences, laid out so that every query owns the
    same number of consecutive rows. That makes a cluster bootstrap a reshape
    and a fancy-index instead of a groupby, which is what makes 1,000
    resamples affordable.
    """
    X: np.ndarray          # (n_queries * pairs_per_query, K)
    n_queries: int
    pairs_per_query: int
    features: list[str]
    query_ids: np.ndarray  # (n_queries,)

    def resample(self, idx: np.ndarray) -> np.ndarray:
        X3 = self.X.reshape(self.n_queries, self.pairs_per_query, -1)
        return X3[idx].reshape(-1, X3.shape[2])


def build_pairs(df: pd.DataFrame, features: list[str],
                max_pairs: int | None = None, seed: int = 0) -> PairBlocks:
    """
    One row per ordered pair (better-ranked doc, worse-ranked doc), value =
    x_better - x_worse. Only pairs with different relevance grades carry
    information, so ties are dropped.
    """
    rng = np.random.default_rng(seed)
    df = df.sort_values(["query_id", "rank"], kind="stable")
    qids = df["query_id"].to_numpy()
    uq, counts = np.unique(qids, return_counts=True)
    D = int(counts[0])
    if not np.all(counts == D):
        raise ValueError("build_pairs expects a balanced panel")

    X = df[features].to_numpy(dtype=np.float64).reshape(len(uq), D, -1)
    grades = df["grade"].to_numpy().reshape(len(uq), D)[0]

    ii, jj = np.triu_indices(D, k=1)
    keep = grades[ii] != grades[jj]
    ii, jj = ii[keep], jj[keep]

    if max_pairs is not None and len(ii) > max_pairs:
        sel = rng.choice(len(ii), size=max_pairs, replace=False)
        ii, jj = ii[sel], jj[sel]

    diffs = X[:, ii, :] - X[:, jj, :]          # better minus worse
    return PairBlocks(
        X=diffs.reshape(-1, diffs.shape[2]),
        n_queries=len(uq),
        pairs_per_query=len(ii),
        features=features,
        query_ids=uq,
    )


# ---------------------------------------------------------------------------
# Head 1: pairwise logistic, hand-rolled IRLS
# ---------------------------------------------------------------------------

def fit_pairwise(X: np.ndarray, ridge: float = 1e-3, beta0=None,
                 max_iter: int = 25, tol: float = 1e-8) -> np.ndarray:
    """
    Maximise the Bradley-Terry log-likelihood  sum log sigma(X @ beta).

    Every row is a difference oriented so the better document comes first, so
    the label is 1 everywhere and the model carries no intercept -- which is
    exactly what we want, because an intercept would be a claim about
    absolute rank, and we are only ever modelling order within a query.
    """
    n, k = X.shape
    beta = np.zeros(k) if beta0 is None else beta0.astype(np.float64).copy()
    I = np.eye(k) * ridge * n
    for _ in range(max_iter):
        eta = X @ beta
        p = _sigmoid(eta)
        grad = X.T @ (1.0 - p) - ridge * n * beta
        w = p * (1.0 - p)
        H = (X * w[:, None]).T @ X + I
        step = np.linalg.solve(H, grad)
        beta += step
        if np.max(np.abs(step)) < tol:
            break
    return beta


def _sigmoid(x: np.ndarray) -> np.ndarray:
    out = np.empty_like(x)
    pos = x >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-x[pos]))
    ex = np.exp(x[~pos])
    out[~pos] = ex / (1.0 + ex)
    return out


def importances(beta: np.ndarray, features: list[str],
                ref: int | None = None) -> pd.DataFrame:
    """
    Two normalisations, because they answer different questions and people
    confuse them constantly.

      share  -- |b_k| / sum|b|.  Comparable to a weight table that sums to 1.
      index  -- |b_k| / |b_ref|. The 0-to-1 scale an SEO tool prints, and the
                scale on which "0.84 versus 0.73" lives.

    `ref` is pinned to one factor rather than recomputed as the per-fit
    maximum. Re-picking the maximum inside every bootstrap draw makes the
    denominator random, which quietly forces whichever factor happens to win
    that draw to exactly 1.0 and destroys the interval.
    """
    a = np.abs(beta)
    if ref is None:
        ref = int(np.argmax(a))
    denom = max(a[ref], 1e-12)
    return pd.DataFrame({
        "factor": features,
        "coef": beta,
        "share": a / a.sum(),
        "index": a / denom,
    })


# ---------------------------------------------------------------------------
# Head 2: gradient-boosted ranker
# ---------------------------------------------------------------------------

def fit_gbm(train: pd.DataFrame, features: list[str], seed: int = 0):
    """LightGBM lambdarank when it is installed, sklearn otherwise."""
    if HAVE_LGB:
        train = train.sort_values("query_id", kind="stable")
        group = train.groupby("query_id", sort=True).size().to_numpy()
        m = lgb.LGBMRanker(objective="lambdarank", n_estimators=250,
                           learning_rate=0.06, num_leaves=31, verbose=-1,
                           random_state=seed)
        m.fit(train[features], train["grade"], group=group)
        return m, "LightGBM lambdarank"
    m = HistGradientBoostingRegressor(max_iter=250, learning_rate=0.06,
                                      max_leaf_nodes=31, random_state=seed)
    m.fit(train[features], train["grade"])
    return m, "sklearn HistGradientBoosting"


# ---------------------------------------------------------------------------
# Ranking metrics
# ---------------------------------------------------------------------------

def ndcg_at_k(df: pd.DataFrame, score_col: str, k: int = 10) -> float:
    """Mean NDCG@k over queries, gain = 2^grade - 1."""
    d = df.sort_values(["query_id", score_col], ascending=[True, False],
                       kind="stable")
    n_q = d["query_id"].nunique()
    D = len(d) // n_q
    g = d["grade"].to_numpy().reshape(n_q, D)[:, :k]
    gain = np.exp2(g) - 1.0
    disc = 1.0 / np.log2(np.arange(2, gain.shape[1] + 2))
    dcg = (gain * disc).sum(axis=1)
    ideal = np.sort(d["grade"].to_numpy().reshape(n_q, D), axis=1)[:, ::-1][:, :k]
    idcg = ((np.exp2(ideal) - 1.0) * disc).sum(axis=1)
    return float(np.mean(np.divide(dcg, idcg, out=np.zeros_like(dcg),
                                   where=idcg > 0)))


def kendall_tau(df: pd.DataFrame, score_col: str) -> float:
    """Mean within-query Kendall tau-b between our order and Google's."""
    from scipy.stats import kendalltau
    taus = []
    for _, g in df.groupby("query_id", sort=False):
        t = kendalltau(-g[score_col].to_numpy(), g["rank"].to_numpy()).statistic
        if np.isfinite(t):
            taus.append(t)
    return float(np.mean(taus))


def pairwise_accuracy(df: pd.DataFrame, score_col: str) -> float:
    """Share of graded pairs we order the same way Google did."""
    n_q = df["query_id"].nunique()
    d = df.sort_values(["query_id", "rank"], kind="stable")
    D = len(d) // n_q
    s = d[score_col].to_numpy().reshape(n_q, D)
    g = d["grade"].to_numpy().reshape(n_q, D)[0]
    ii, jj = np.triu_indices(D, k=1)
    keep = g[ii] != g[jj]
    ii, jj = ii[keep], jj[keep]
    return float((s[:, ii] > s[:, jj]).mean())


# ---------------------------------------------------------------------------
# Position-bias correction (Joachims-style, propensities from randomisation)
# ---------------------------------------------------------------------------

def estimate_examination_curve(df: pd.DataFrame) -> tuple[float, np.ndarray]:
    """
    Estimate P(examine | position) from the randomised slice only.

    In randomised queries the displayed order is independent of quality, so
    average click volume by position is proportional to examination and
    nothing else. Regressing log(clicks) on log(position) gives the exponent.

    An agency without a randomisation slice can get the same estimate from
    naturally occurring rank churn: the same URL observed at several positions
    within a short window identifies the curve, at the cost of a much larger
    panel.
    """
    rnd = df[df["is_randomized"]]
    if len(rnd) < 200:
        raise ValueError("randomised slice too small to identify propensities")
    m = rnd.groupby("shown_rank")["obs_clicks"].mean()
    x = np.log(m.index.to_numpy(dtype=float))
    y = np.log(m.to_numpy())
    A = np.column_stack([np.ones_like(x), x])
    coef, *_ = np.linalg.lstsq(A, y, rcond=None)
    eta_hat = -coef[1]
    ranks = np.arange(1, int(df["shown_rank"].max()) + 1)
    return float(eta_hat), np.power(ranks, -eta_hat)


def ipw_click_feature(df: pd.DataFrame, eta_hat: float) -> np.ndarray:
    """
    Inverse-propensity-weighted click signal.

    The naive construction -- observed clicks minus the average at that
    position -- looks like it removes position bias. It does not. Position is
    caused by quality, so subtracting the positional mean subtracts the
    quality signal along with the bias. Dividing by the estimated examination
    probability removes the bias and keeps the signal.
    """
    prop = np.power(df["shown_rank"].to_numpy(dtype=float), -eta_hat)
    v = np.log(df["obs_clicks"].to_numpy() / prop)
    return (v - v.mean()) / v.std()


# ---------------------------------------------------------------------------

def standardize(df: pd.DataFrame, features: list[str],
                within: bool = True) -> pd.DataFrame:
    """
    Scale features to the variation the model actually uses.

    This looks like a housekeeping detail and is not. Every estimator here
    works on differences WITHIN a query, so the only variation that reaches
    it is within-query variation. Standardising by the global standard
    deviation folds in between-query variation the model never sees -- and
    between-query variation is large, because a finance SERP and a plumber
    SERP are made of completely different documents.

    Divide by a standard deviation inflated with variance the model cannot
    use and the coefficient has to grow to compensate. In a first pass here
    that inflated site authority by roughly a third and flipped its verdict
    from RECOVERED to INFLATED, for purely clerical reasons.

    So: centre each feature inside its own query, and scale by the pooled
    within-query standard deviation.
    """
    out = df.copy()
    if not within:
        for f in features:
            v = out[f].to_numpy(dtype=float)
            sd = v.std()
            out[f] = (v - v.mean()) / (sd if sd > 1e-12 else 1.0)
        return out

    g = out.groupby("query_id")
    for f in features:
        v = out[f].to_numpy(dtype=float)
        dev = v - g[f].transform("mean").to_numpy()
        sd = dev.std()
        out[f] = dev / (sd if sd > 1e-12 else 1.0)
    return out
