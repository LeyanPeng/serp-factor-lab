"""
A synthetic Google, whose weights we know.

This is the load-bearing idea of the whole lab. We can never see Google's real
weights, so "is 0.84 correct?" is unanswerable head-on. What IS answerable:
given a ranker whose weights we set ourselves, does our estimation pipeline
recover them? Whatever error it makes here is the floor on the error it makes
on real SERPs.

The fake Google is deliberately unkind to us. It reproduces four things that
make real SERP data hard, and each one is something our estimator is allowed
to fail on:

  1. It is a pipeline, not a function.
     retrieval -> base scoring -> twiddler re-ranking -> truncation to top 10.

  2. Weights differ by query intent cluster.
     A single global weight table is the wrong model of the world.

  3. Most of the score comes from things we cannot measure.
     `unobserved_sd` caps the achievable NDCG at a realistic level. A harness
     where you can reach NDCG 0.99 is a harness that has taught you nothing.

  4. Our instruments are imperfect.
     - The click signal is observed through position bias, so the obvious
       "residual CTR" construction destroys the signal it is meant to capture.
     - Content depth is measured with error (boilerplate, nav chrome,
       pagination), while the worthless decoy next to it is measured cleanly.
       That is how a factor with a true weight of exactly zero earns a
       confident non-zero coefficient.
"""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------

# How each intent cluster bends the global weight vector. These multipliers are
# the ground truth behind the "backlinks matter 3x more in finance than in
# local services" headline -- the pipeline has to rediscover them.
CLUSTER_MODIFIERS: dict[str, dict[str, float]] = {
    "ymyl-finance": {
        "host_pagerank": 2.40, "ref_domains_log": 2.00, "eeat_score": 2.20,
        "content_depth_rel": 1.10, "brand_demand": 1.30, "residual_ctr": 0.70,
        "kg_entity_coverage": 1.45,
    },
    "ecommerce": {
        "schema_coverage": 3.00, "residual_ctr": 1.30, "cwv_inp": 2.00,
        "host_pagerank": 0.90, "eeat_score": 0.70, "content_depth_rel": 0.70,
        "kg_entity_coverage": 0.65,
    },
    "local-service": {
        "host_pagerank": 0.55, "ref_domains_log": 0.50, "brand_demand": 1.60,
        "eeat_score": 0.60, "residual_ctr": 1.40, "content_depth_rel": 0.60,
        "kg_entity_coverage": 0.50,
    },
    "informational": {
        "semantic_sim": 1.20, "content_depth_rel": 1.30, "freshness_gap": 1.50,
        "eeat_score": 1.20, "schema_coverage": 0.50,
        "kg_entity_coverage": 1.70,
    },
}


# Clusters differ in what their SERPs are MADE of, not only in how the parts
# are weighted. A finance SERP is a wall of aged, heavily-linked domains; a
# local-services SERP is not. These are pure level shifts applied to every
# document on the query, so they cancel exactly in the within-query
# differencing the estimator uses -- they change what a SERP looks like
# without touching the weights the estimator has to recover.
CLUSTER_FEATURE_SHIFTS: dict[str, dict[str, float]] = {
    "ymyl-finance": {
        "host_pagerank": 1.20, "ref_domains_log": 1.05, "eeat_score": 0.95,
        "content_depth_rel": 0.55, "host_age": 0.0, "brand_demand": 0.60,
    },
    "ecommerce": {
        "schema_coverage": 1.40, "residual_ctr": 0.35, "content_depth_rel": -0.70,
        "cwv_inp": -0.45, "eeat_score": -0.55,
    },
    "local-service": {
        "host_pagerank": -1.10, "ref_domains_log": -0.95, "brand_demand": -0.50,
        "content_depth_rel": -0.60, "eeat_score": -0.70,
    },
    "informational": {
        "content_depth_rel": 0.85, "freshness_gap": 0.55, "schema_coverage": -0.60,
        "semantic_sim": 0.30,
    },
}


def cluster_weight_table() -> dict[str, dict[str, float]]:
    """Per-cluster ground-truth weights, each renormalised to sum to 1."""
    base = config.true_weights()
    out: dict[str, dict[str, float]] = {}
    for cluster in config.CLUSTERS:
        mods = CLUSTER_MODIFIERS.get(cluster, {})
        w = {k: v * mods.get(k, 1.0) for k, v in base.items()}
        total = sum(w.values())
        out[cluster] = {k: v / total for k, v in w.items()}
    return out


@dataclass
class SerpPanel:
    """One simulated SERP panel plus the truth used to build it."""
    df: pd.DataFrame
    features: list[str]
    true_weights: dict[str, float]
    cluster_weights: dict[str, dict[str, float]]
    eta: float                       # position-bias exponent actually used
    meta: dict = field(default_factory=dict)

    @property
    def n_queries(self) -> int:
        return int(self.df["query_id"].nunique())


# ---------------------------------------------------------------------------
# The simulator
# ---------------------------------------------------------------------------

def simulate(
    n_queries: int = 1_200,
    docs_per_query: int = 10,
    candidates_per_query: int = 24,
    retrieval_keep: int = 16,
    seed: int = 7,
    eta: float = 1.0,               # position-bias exponent, P(examine) ~ 1/r^eta
    randomize_frac: float = 0.05,   # share of queries shown in random order
    depth_reliability: float = 0.45,  # how well we measure content depth
    decoy_corr: float = 0.85,       # correlation of the decoy with true depth
    unobserved_sd: float = 0.40,    # score mass we can never measure
    crowding_demotion: float = 0.25,
) -> SerpPanel:
    rng = np.random.default_rng(seed)

    feats = config.tracked_ids()
    K = len(feats)
    i_depth = feats.index("content_depth_rel")
    i_ctr = feats.index("residual_ctr")
    i_decoy = feats.index("kw_density")
    i_crowd_proxy = feats.index("brand_demand")  # drives host identity below

    cw = cluster_weight_table()
    clusters = rng.choice(config.CLUSTERS, size=n_queries)

    C = candidates_per_query
    # --- latent document features, standard normal, one block per query -----
    Z = rng.standard_normal((n_queries, C, K))

    # The decoy is not independent: it is a clean measurement of the same
    # underlying thing that content depth measures noisily.
    true_depth = Z[:, :, i_depth].copy()
    Z[:, :, i_decoy] = (decoy_corr * true_depth
                        + np.sqrt(1 - decoy_corr ** 2)
                        * rng.standard_normal((n_queries, C)))

    # Cluster-level composition shifts. Applied to every candidate on the
    # query, so they are invisible to a within-query estimator and visible to
    # anything that looks at the shape of the SERP as a whole -- which is
    # exactly how cluster.py is able to find the clusters without being told.
    for cluster, shifts in CLUSTER_FEATURE_SHIFTS.items():
        qmask = clusters == cluster
        if not qmask.any():
            continue
        for fname, delta in shifts.items():
            if fname in feats:
                Z[qmask, :, feats.index(fname)] += delta

    # Latent click propensity is what the fake Google's twiddler reads. The
    # analyst never sees it directly.
    latent_ctr = Z[:, :, i_ctr].copy()

    # --- per-query weight vector -------------------------------------------
    # The click term is deliberately NOT in this vector. NavBoost is a
    # re-ranker, so the click signal enters at stage C below, with exactly its
    # nominal weight. Adding it in both places would silently make the click
    # factor several times more powerful than the weight table claims.
    W = np.zeros((n_queries, K))
    for ci, cluster in enumerate(config.CLUSTERS):
        mask = clusters == cluster
        W[mask] = np.array([cw[cluster][f] for f in feats])
    W_ctr = W[:, i_ctr].copy()
    W[:, i_ctr] = 0.0

    # --- stage A: retrieval -------------------------------------------------
    # Index selection is mostly topical and quite noisy. Documents that never
    # clear this stage are never observed by us at all: our panel is a
    # truncated sample, which is itself a source of bias worth naming.
    i_sem = feats.index("semantic_sim")
    retrieval = Z[:, :, i_sem] + 0.8 * rng.standard_normal((n_queries, C))
    keep = np.argsort(-retrieval, axis=1)[:, :retrieval_keep]
    rows = np.arange(n_queries)[:, None]
    Z = Z[rows, keep]                       # (n_queries, retrieval_keep, K)
    latent_ctr = latent_ctr[rows, keep]
    true_depth = true_depth[rows, keep]

    # --- stage B: base score ------------------------------------------------
    base = np.einsum("qdk,qk->qd", Z, W)
    unobserved = unobserved_sd * rng.standard_normal(base.shape)
    base = base + unobserved

    # --- stage C: twiddlers -------------------------------------------------
    # A NavBoost-style re-rank driven by the click signal, plus a one-sided
    # host-crowding demotion. Twiddlers are why a single additive weight
    # vector is the wrong functional form for Google in the first place.
    host_id = np.argsort(Z[:, :, i_crowd_proxy], axis=1) % 6
    final = base + W_ctr[:, None] * latent_ctr
    for h in range(6):
        same = host_id == h
        seen = np.cumsum(same, axis=1) * same
        final = final - crowding_demotion * np.maximum(seen - 1, 0)  # never a boost

    # --- realized SERP ------------------------------------------------------
    order = np.argsort(-final, axis=1)[:, :docs_per_query]
    Z = Z[rows, order]
    latent_ctr = latent_ctr[rows, order]
    true_depth = true_depth[rows, order]
    final_score = final[rows, order]
    D = docs_per_query
    rank = np.tile(np.arange(1, D + 1), (n_queries, 1))

    # --- what the analyst actually gets to measure --------------------------
    # (a) content depth is measured with error
    lam = np.sqrt(depth_reliability)
    Z[:, :, i_depth] = (lam * true_depth
                        + np.sqrt(1 - lam ** 2)
                        * rng.standard_normal((n_queries, D)))

    # (b) a randomised slice, which is the only clean way to estimate the
    #     examination curve. Real agencies approximate this with naturally
    #     occurring rank volatility; a platform would run it deliberately.
    is_rand = rng.random(n_queries) < randomize_frac
    shown_rank = rank.copy()
    if is_rand.any():
        idx = np.where(is_rand)[0]
        perm = np.argsort(rng.random((idx.size, D)), axis=1)
        shown_rank[idx] = perm + 1

    # (c) clicks are observed through position bias
    examination = 1.0 / np.power(shown_rank, eta)
    click_noise = np.exp(0.35 * rng.standard_normal((n_queries, D)))
    obs_clicks = examination * np.exp(0.55 * latent_ctr) * click_noise

    # --- assemble -----------------------------------------------------------
    df = pd.DataFrame({
        "query_id": np.repeat(np.arange(n_queries), D),
        "cluster": np.repeat(clusters, D),
        "doc_id": np.arange(n_queries * D),
        "rank": rank.ravel(),
        "shown_rank": shown_rank.ravel(),
        "is_randomized": np.repeat(is_rand, D),
        "google_score": final_score.ravel(),
        "obs_clicks": obs_clicks.ravel(),
        "_latent_ctr": latent_ctr.ravel(),
        "_examination": examination.ravel(),
    })
    for k, name in enumerate(feats):
        df[name] = Z[:, :, k].ravel()

    # The naive residual-CTR construction every SEO team reaches for first:
    # divide out the average click volume at that position. Because rank is
    # caused by click quality, this removes the signal along with the bias.
    pos_mean = df.groupby("shown_rank")["obs_clicks"].transform("mean")
    df["residual_ctr"] = _z(np.log(df["obs_clicks"] / pos_mean))

    # Graded relevance for NDCG: we are learning to reproduce Google's order,
    # so Google's order is the label.
    grade = np.zeros(D, dtype=float)
    grade[0] = 3.0
    grade[1:3] = 2.0
    grade[3:6] = 1.0
    df["grade"] = np.tile(grade, n_queries)

    return SerpPanel(
        df=df,
        features=feats,
        true_weights=config.true_weights(),
        cluster_weights=cw,
        eta=eta,
        meta=dict(seed=seed, n_queries=n_queries, docs_per_query=D,
                  unobserved_sd=unobserved_sd, depth_reliability=depth_reliability,
                  decoy_corr=decoy_corr, randomize_frac=randomize_frac),
    )


def simulate_update(n_queries: int = 1_500, seed: int = 7,
                    factor: str = "eeat_score", multiplier: float = 2.2):
    """
    Two panels either side of a core update, where exactly one factor's weight
    changes. causal.py has to detect which one moved, without being told.
    """
    before = simulate(n_queries=n_queries, seed=seed)

    original = {c: dict(m) for c, m in CLUSTER_MODIFIERS.items()}
    try:
        for c in CLUSTER_MODIFIERS:
            CLUSTER_MODIFIERS[c][factor] = (
                CLUSTER_MODIFIERS[c].get(factor, 1.0) * multiplier)
        after = simulate(n_queries=n_queries, seed=seed + 1)
    finally:
        CLUSTER_MODIFIERS.clear()
        CLUSTER_MODIFIERS.update(original)

    before.df["period"] = 0
    after.df["period"] = 1
    return before, after, factor


# ---------------------------------------------------------------------------

def _z(x) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    sd = x.std()
    return (x - x.mean()) / (sd if sd > 1e-12 else 1.0)


if __name__ == "__main__":
    p = simulate(n_queries=800, seed=7)
    print(f"panel: {len(p.df):,} rows, {p.n_queries:,} queries, "
          f"{len(p.features)} features")
    print(f"randomised queries: {p.df['is_randomized'].sum() // 10}")
    print("\nground-truth weights (global):")
    for k, v in sorted(p.true_weights.items(), key=lambda kv: -kv[1]):
        print(f"  {k:<20} {v:.3f}")
    print("\nbacklink weight by cluster:")
    for c in config.CLUSTERS:
        print(f"  {c:<16} host_pagerank {p.cluster_weights[c]['host_pagerank']:.3f}")
