"""
Intent clustering -- why one global weight table is the wrong deliverable.

A single weight vector averaged over finance, e-commerce, local services and
plain informational queries describes none of them. For an agency this is not
a statistical nicety, it is the product: the answer to "where should this
client spend" is different for a wealth manager and a plumber, and a global
table cannot tell you that.

Clusters are DISCOVERED from SERP shape, not assumed. Two questions are
answered here:

  1. Can we recover intent groups without being told them?
     (adjusted Rand index against the truth)
  2. Do the weights actually differ across the groups we found?
     (per-cluster fits, compared against the per-cluster truth)
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

import model


def serp_fingerprint(df: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    """
    One row per query describing the SHAPE of its result set.

    Deliberately built from the SERP, never from the query string. Two queries
    with no words in common belong together when Google treats them the same
    way, and that is the grouping the weights follow. In production the same
    vector also carries SERP-feature flags -- ad slots, AI Overview presence,
    local pack, video carousel -- which are the strongest signals of all.
    """
    g = df.groupby("query_id")
    parts = [g[f].agg(["mean", "std"]).add_prefix(f"{f}_") for f in features]
    top = df[df["rank"] == 1].set_index("query_id")[features].add_prefix("win_")
    fp = pd.concat(parts + [top], axis=1).fillna(0.0)
    return (fp - fp.mean()) / fp.std().replace(0, 1)


def discover_clusters(df: pd.DataFrame, features: list[str], k: int = 4,
                      seed: int = 0) -> tuple[pd.Series, float]:
    """Cluster queries by SERP shape; score against the truth if we have it."""
    fp = serp_fingerprint(df, features)
    km = KMeans(n_clusters=k, n_init=10, random_state=seed).fit(fp.to_numpy())
    found = pd.Series(km.labels_, index=fp.index, name="found_cluster")

    ari = float("nan")
    if "cluster" in df.columns:
        truth = df.groupby("query_id")["cluster"].first().reindex(fp.index)
        ari = adjusted_rand_score(truth, found)
    return found, ari


def weights_by_cluster(df: pd.DataFrame, features: list[str],
                       label_col: str = "cluster",
                       min_queries: int = 150) -> pd.DataFrame:
    """Fit the interpretable head separately inside each cluster."""
    ref = features.index("semantic_sim")
    rows = []
    for name, part in df.groupby(label_col):
        if part["query_id"].nunique() < min_queries:
            continue
        d = model.standardize(part, features)
        pairs = model.build_pairs(d, features, max_pairs=15)
        beta = model.fit_pairwise(pairs.X, ridge=1e-6)
        idx = np.abs(beta) / max(abs(beta[ref]), 1e-12)
        for f, v in zip(features, idx):
            rows.append(dict(cluster=name, factor=f, index=float(v),
                             n_queries=int(part["query_id"].nunique())))
    return pd.DataFrame(rows)


def spread_report(wbc: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    Which factors vary most across clusters -- the actual finding.

    A factor with a flat profile is a company-wide rule. A factor with a wide
    profile is a per-client budgeting decision, and those are the ones worth
    putting in front of an account manager.
    """
    p = wbc.pivot(index="factor", columns="cluster", values="index")
    p["ratio"] = p.max(axis=1) / p.min(axis=1).clip(lower=1e-6)
    p["spread"] = p.drop(columns="ratio").max(axis=1) - p.drop(columns="ratio").min(axis=1)
    out = p.sort_values("spread", ascending=False).head(top_n)
    # clear axis names: pandas prints them as a stray header row that
    # breaks the alignment of the table in the terminal report
    return out.rename_axis(index=None, columns=None)
