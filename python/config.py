"""
Factor registry -- the single source of truth for the whole lab.

Both the Python pipeline and the Next.js dashboard read this file (the
dashboard reads the JSON that `export_dashboard.py` derives from it), so a
factor is defined exactly once.

Evidence tiers
--------------
A  Confirmed by Google in public documentation or by a Google witness.
B  Established by the DOJ v. Google trial record and/or the March 2024
   Content Warehouse API documentation leak. Strong, but second-hand.
C  Industry folklore. Correlational at best. We deliberately DO NOT track
   these -- see REJECTED below. Listing what you refuse to model is part
   of the model.
"""

from dataclasses import dataclass, asdict


@dataclass(frozen=True)
class Factor:
    id: str
    group: str
    label: str
    tier: str                 # "A" | "B" | "C"
    how: str                  # how we actually measure it
    source: str               # where the data comes from
    cost: str                 # marginal cost per 1k URLs
    refresh: str              # how often it is worth re-measuring
    simulated: bool = False   # carried into the calibration harness
    true_weight: float = 0.0  # ground truth, only meaningful when simulated
    note: str = ""


# ---------------------------------------------------------------------------
# The 34 factors we track. Six groups.
# ---------------------------------------------------------------------------

FACTORS: list[Factor] = [
    # -- 1. Query-document relevance ----------------------------------------
    Factor("semantic_sim", "relevance", "Dense query-document similarity", "A",
           "bi-encoder cosine between query vector and page vector",
           "local sentence-transformer", "$0 (CPU)", "on crawl",
           simulated=True, true_weight=0.220,
           note="The strongest single factor in every cluster we have modelled."),
    Factor("passage_max_sim", "relevance", "Passage-level max similarity", "A",
           "chunk page into 90-word passages, take max cosine against query",
           "local sentence-transformer", "$0 (CPU)", "on crawl",
           note="Predicts AI Overview citation far better than page-level "
                "similarity does."),
    Factor("bm25_body", "relevance", "BM25 lexical match (body)", "A",
           "classic BM25 over tokenised body text against query terms",
           "own index", "$0", "on crawl"),
    Factor("title_match", "relevance", "Title-query match", "B",
           "normalised token overlap plus order penalty, title against query",
           "SERP + crawl", "$0", "on crawl",
           simulated=True, true_weight=0.055,
           note="The leak exposes a titlematchScore attribute."),
    Factor("topic_coverage_gap", "relevance", "Topic coverage gap vs SERP", "B",
           "entities present in the SERP centroid but missing from this page",
           "crawl + NER", "$0.40", "monthly",
           note="Actionable by construction: it names the missing subtopics."),
    Factor("kg_entity_coverage", "relevance", "Knowledge-graph entity coverage",
           "B",
           "PageRank over the entity graph built from the top-10 pages, then "
           "the share of that salience mass this page carries",
           "Cloud NL analyzeEntities + Wikidata", "$1.00 / 1k pages", "monthly",
           simulated=True, true_weight=0.145,
           note="A search engine ranks entities, not strings. This is the only "
                "factor in the list whose by-product is a work order rather "
                "than a score -- the same graph names the entities the page "
                "is missing. See knowledge_graph.py."),
    Factor("intent_match", "relevance", "Query intent match", "A",
           "classify query intent, classify page type, score the agreement",
           "SERP features + crawl", "$0", "on crawl"),

    # -- 2. Site-level authority --------------------------------------------
    Factor("host_pagerank", "authority", "Host-graph PageRank", "B",
           "power iteration on our own host graph (scipy.sparse)",
           "Common Crawl host graph", "$0 after build", "quarterly",
           simulated=True, true_weight=0.160,
           note="The leak exposes siteAuthority. We compute our own rather "
                "than buying a vendor's black box."),
    Factor("ref_domains_log", "authority", "Referring root domains (log)", "A",
           "log1p of distinct linking root domains",
           "Ahrefs / Majestic / Common Crawl", "$1.10", "monthly",
           simulated=True, true_weight=0.090),
    Factor("topical_link_share", "authority", "Topical relevance of links", "B",
           "share of referring domains inside the same topic cluster",
           "link data + clustering", "$0.20", "monthly",
           note="Separates 1,000 irrelevant links from 50 relevant ones."),
    Factor("host_age", "authority", "Host age", "B",
           "years since the host was first observed in a crawl",
           "Wayback / Common Crawl history", "$0", "yearly",
           note="The leak exposes hostAge, used mainly in a spam context."),
    Factor("brand_demand", "authority", "Branded search demand", "B",
           "monthly search volume for brand terms, log-scaled",
           "keyword API", "$0.05", "monthly",
           simulated=True, true_weight=0.040,
           note="Cheap, strong, and almost nobody tracks it. Our best single "
                "proxy for the site-level trust term."),

    # -- 3. Page-level quality ----------------------------------------------
    Factor("content_depth_rel", "quality", "Content depth vs SERP median", "A",
           "page word count divided by the median of the top 10 for the query",
           "crawl", "$0", "on crawl",
           simulated=True, true_weight=0.185,
           note="RELATIVE, not absolute. Absolute word count is on the "
                "rejected list; the ratio is one of the strongest factors."),
    Factor("info_gain", "quality", "Information gain / originality", "B",
           "1 minus max cosine between this page and the other top-10 pages",
           "crawl + embeddings", "$0", "on crawl",
           note="The leak exposes OriginalContentScore."),
    Factor("eeat_score", "quality", "E-E-A-T rubric score", "A",
           "an LLM scores the page against Google's Quality Rater Guidelines, "
           "calibrated against human raters with Krippendorff alpha reported",
           "LLM API + human sample", "$3.00", "quarterly",
           simulated=True, true_weight=0.075,
           note="Never shipped without its inter-rater reliability number."),
    Factor("author_entity", "quality", "Author entity resolution", "A",
           "byline present AND resolvable to a Knowledge Graph entity",
           "crawl + entity API", "$0.30", "quarterly"),
    Factor("freshness_gap", "quality", "Date honesty gap", "B",
           "disagreement between the visible byline date, the markup date and "
           "the date inferred from the content itself",
           "crawl", "$0", "on crawl",
           simulated=True, true_weight=0.025,
           note="The leak exposes bylineDate, syntacticDate and semanticDate "
                "as three separate fields -- faked freshness is detectable."),
    Factor("schema_coverage", "quality", "Structured data coverage", "A",
           "count and validity of schema.org types relevant to the page type",
           "crawl", "$0", "on crawl",
           simulated=True, true_weight=0.012),
    Factor("ad_density_atf", "quality", "Above-the-fold ad density", "A",
           "share of the first viewport occupied by ads or interstitials",
           "headless render", "$1.80", "quarterly"),

    # -- 4. User-interaction proxies (the NavBoost family) ------------------
    Factor("residual_ctr", "interaction", "Residual CTR", "B",
           "observed CTR minus the expected CTR for that position",
           "GSC (own sites) / clickstream panel", "$0", "weekly",
           simulated=True, true_weight=0.130,
           note="The closest public proxy for the leaked goodClicks. Position "
                "bias makes the naive version badly biased -- this is the "
                "factor the calibration harness catches us getting wrong."),
    Factor("pogo_rate", "interaction", "Pogo-sticking rate", "B",
           "share of sessions returning to the SERP within 15 seconds",
           "clickstream panel", "$4.00", "monthly",
           note="Proxy for the leaked badClicks."),
    Factor("last_click_share", "interaction", "Last-longest-click share", "B",
           "share of query sessions where this URL is the terminal click",
           "clickstream panel", "$4.00", "monthly",
           note="Proxy for lastLongestClicks, the strongest of the NavBoost "
                "click families in the leaked schema."),
    Factor("url_nav_demand", "interaction", "Navigational demand for URL", "B",
           "volume of navigational queries that resolve to this URL",
           "keyword API", "$0.05", "monthly"),
    Factor("return_visit_rate", "interaction", "Return visit rate", "B",
           "share of visitors returning within 30 days",
           "GSC / GA, own sites only", "$0", "monthly",
           note="Only available for client sites. That asymmetry is exactly "
                "why an agency can model this and a tool vendor cannot."),

    # -- 5. Technical delivery ----------------------------------------------
    Factor("cwv_lcp", "technical", "LCP (field)", "A",
           "75th percentile Largest Contentful Paint, field data",
           "CrUX API", "$0", "monthly"),
    Factor("cwv_inp", "technical", "INP (field)", "A",
           "75th percentile Interaction to Next Paint, field data",
           "CrUX API", "$0", "monthly",
           simulated=True, true_weight=0.008,
           note="Replaced FID in March 2024. Small but real, and a tie-breaker "
                "in crowded SERPs."),
    Factor("cwv_cls", "technical", "CLS (field)", "A",
           "75th percentile Cumulative Layout Shift, field data",
           "CrUX API", "$0", "monthly"),
    Factor("indexability", "technical", "Indexability & canonical health", "A",
           "robots, noindex, canonical conflicts, soft-404 detection",
           "crawl + GSC", "$0", "weekly",
           note="A binary gate, not a weight. A page that fails this has no "
                "ranking at all, so it is excluded from the model rather "
                "than scored by it."),
    Factor("mobile_parity", "technical", "Mobile content parity", "A",
           "content diff between the mobile and desktop render",
           "headless render", "$1.80", "quarterly"),
    Factor("ttfb", "technical", "Time to first byte", "A",
           "server response time measured from three geographies",
           "own probes", "$0", "weekly"),

    # -- 6. SERP competitive context ----------------------------------------
    Factor("serp_volatility", "serp", "SERP volatility", "A",
           "mean rank churn of the top 10 over a trailing 30 days",
           "own SERP history", "$0", "daily",
           note="Tells the client which keywords are winnable at all."),
    Factor("organic_compression", "serp", "Organic space compression", "A",
           "ad slots plus AI Overview plus other features above result one",
           "SERP API", "$0", "daily",
           note="Not a ranking factor -- a traffic factor. Position 3 is worth "
                "very different amounts on two different SERPs."),
    Factor("result_type_bias", "serp", "Preferred result type", "A",
           "distribution of page types Google already rewards for this query",
           "SERP API + classifier", "$0", "weekly"),
    Factor("host_crowding", "serp", "Host crowding", "B",
           "number of results from the same host already in the top 10",
           "SERP API", "$0", "daily",
           note="A one-sided demotion, not a symmetric weight."),
    Factor("site_serp_presence", "serp", "Own site's other placements", "A",
           "whether this site appears elsewhere on the same SERP",
           "SERP API", "$0", "daily"),
]


# The decoy: planted with a true weight of exactly zero and correlated with
# content_depth_rel. It is not part of the tracked set. It is the control
# that tells us when our estimator has started hallucinating.
DECOY = Factor("kw_density", "rejected", "Keyword density", "C",
               "keyword occurrences divided by total words",
               "crawl", "$0", "not tracked",
               simulated=True, true_weight=0.000,
               note="True weight is exactly zero and it is 0.85-correlated "
                    "with content depth. If our estimator assigns it weight, "
                    "our estimator is broken and we need to know that.")


# ---------------------------------------------------------------------------
# What we refuse to track, and why. This list is part of the deliverable.
# ---------------------------------------------------------------------------

REJECTED = [
    ("Keyword density",
     "No evidence in any tier. Collinear with content depth, so it looks "
     "predictive in any naive model. We use it as our decoy."),
    ("Absolute word count",
     "Predictive only because it proxies depth-versus-competitors. Use the "
     "ratio; the raw number leads clients to pad pages."),
    ("Bounce rate from Analytics",
     "Google does not see your GA. Not a signal, and our own measurement of "
     "it is mostly a session-definition artefact."),
    ("Domain Authority as a lever",
     "Fine as a covariate, fatal as a target. It is a vendor's model output, "
     "not a Google input -- optimising it optimises Moz's model, not Google's."),
    ("LSI keywords",
     "Latent Semantic Indexing is a 1988 technique Google has never claimed "
     "to use. Dense retrieval superseded the idea entirely."),
    ("Meta keywords and exact-match domains",
     "Dead since 2009 and 2012 respectively; the leak shows an explicit "
     "exactMatchDomainDemotion attribute."),
]


# ---------------------------------------------------------------------------
# Cost of moving a factor, on a 1-5 scale.
#
# This is the one table on the page that is NOT a measurement. It is an
# agency's judgement about how much work one standard deviation of change
# costs, and it is labelled as such everywhere it is used. It matters because
# effect size alone ranks factors the wrong way round: a title rewrite worth
# a fifth of a position beats a year of link building worth half of one, and
# no purely statistical quantity will ever tell you that.
#
# Replace these with your own delivery data. Yours will be better than ours.
# ---------------------------------------------------------------------------

EFFORT: dict[str, int] = {
    "title_match": 1,        # a title tag edit, shipped this afternoon
    "freshness_gap": 1,      # stop faking dates; a template fix
    "kw_density": 1,         # trivial, and worthless -- kept for contrast
    "schema_coverage": 2,    # one template change across a page type
    "residual_ctr": 2,       # snippet and title work, iterated
    "semantic_sim": 3,       # genuine rewriting against query intent
    "content_depth_rel": 3,  # commissioning real additional content
    "cwv_inp": 3,            # front-end engineering, competing for sprint time
    "eeat_score": 4,         # bylines, credentials, citations, review process
    "host_pagerank": 5,      # link acquisition -- months, and never guaranteed
    "ref_domains_log": 5,    # same
    "brand_demand": 5,       # brand marketing; not an SEO lever at all
}

EFFORT_LABEL = {1: "hours", 2: "days", 3: "weeks",
                4: "a quarter", 5: "months, uncertain"}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

GROUP_LABELS = {
    "relevance":   "Query-document relevance",
    "authority":   "Site-level authority",
    "quality":     "Page-level quality",
    "interaction": "User-interaction proxies",
    "technical":   "Technical delivery",
    "serp":        "SERP competitive context",
}

TIER_LABELS = {
    "A": "Confirmed by Google",
    "B": "DOJ trial record / 2024 API leak",
    "C": "Industry folklore -- not tracked",
}

# Intent clusters used by cluster.py and by the simulator.
CLUSTERS = ["ymyl-finance", "ecommerce", "local-service", "informational"]


def simulated_factors() -> list[Factor]:
    """Factors carried through the calibration harness, decoy last."""
    return [f for f in FACTORS if f.simulated] + [DECOY]


def tracked_ids() -> list[str]:
    return [f.id for f in simulated_factors()]


def true_weights() -> dict[str, float]:
    return {f.id: f.true_weight for f in simulated_factors()}


def label_of(fid: str) -> str:
    for f in simulated_factors():
        if f.id == fid:
            return f.label
    return fid


def as_dicts() -> list[dict]:
    return [asdict(f) for f in FACTORS]


if __name__ == "__main__":
    print(f"{len(FACTORS)} tracked factors across {len(GROUP_LABELS)} groups")
    for g, label in GROUP_LABELS.items():
        n = sum(1 for f in FACTORS if f.group == g)
        print(f"  {label:<34} {n}")
    tiers = {t: sum(1 for f in FACTORS if f.tier == t) for t in "AB"}
    print(f"\n  tier A (confirmed)  {tiers['A']}")
    print(f"  tier B (trial/leak) {tiers['B']}")
    sim = simulated_factors()
    print(f"\n{len(sim)} carried into the calibration harness, "
          f"true weights sum to {sum(f.true_weight for f in sim):.3f}")
    print(f"{len(REJECTED)} factors on the rejected list")
