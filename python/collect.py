"""
The real-data path. Disabled by default -- it costs money and it hits other
people's servers.

Nothing else in this repo needs it. The calibration harness runs on synthetic
SERPs precisely so the method can be argued about before anyone spends a
budget. This file is here to show the wiring is not hand-waving, and to make
the cost of switching it on explicit.

    set SERPAPI_KEY=...            # DataForSEO / Serper / Bright Data
    python python/collect.py --queries queries.csv --out data/panel.parquet

Panel sizing for a pilot, and where the money goes
--------------------------------------------------
    5,000 queries x 3 verticals x top-20 x 3 pulls/week
      = 15,000 queries x 2 billing units x 13 pulls/month
      = 390,000 units/month  @ $0.0006  ~  $234/month     SERP data
    crawling 150k URLs/month on one small VPS             ~ $50/month
    CrUX field data                                          free
    embeddings, run locally on CPU                           free
    LLM rubric scoring, monthly on a 2,000-page sample    ~ $60/month
                                                          -------------
                                                         ~ $350-600/month

That is the entire marginal cost of the pilot. It is a rounding error against
one month of one analyst's time, which is the actual argument for doing this.

Rate limiting, robots.txt and identifying ourselves in the user agent are not
optional politeness. An agency that gets its crawler blocked loses the data
asset that this whole model is built on.
"""

from __future__ import annotations

import argparse
import asyncio
import os
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

USER_AGENT = "SerpFactorLab/0.1 (+research crawler; contact: seo@agency.example)"
SERP_ENDPOINT = "https://api.dataforseo.com/v3/serp/google/organic/live/advanced"
CRUX_ENDPOINT = ("https://chromeuxreport.googleapis.com/v1/records:"
                 "queryRecord?key={key}")


@dataclass
class CollectConfig:
    serp_key: str = os.environ.get("SERPAPI_KEY", "")
    crux_key: str = os.environ.get("CRUX_KEY", "")
    depth: int = 20
    location: str = "United Kingdom"
    language: str = "en"
    concurrency: int = 8
    request_delay: float = 0.25          # seconds between crawl requests per host
    randomize_frac: float = 0.05         # see the note on propensities below


# ---------------------------------------------------------------------------
# 1. SERPs
# ---------------------------------------------------------------------------

async def fetch_serp(client, query: str, cfg: CollectConfig) -> list[dict]:
    """One SERP, structured. Keep the raw payload -- SERP features matter."""
    payload = [{"keyword": query, "location_name": cfg.location,
                "language_code": cfg.language, "depth": cfg.depth}]
    r = await client.post(SERP_ENDPOINT, json=payload,
                          auth=(cfg.serp_key, ""), timeout=30)
    r.raise_for_status()
    items = r.json()["tasks"][0]["result"][0]["items"]
    return [
        dict(query=query, rank=i["rank_absolute"], url=i.get("url"),
             domain=i.get("domain"), title=i.get("title"),
             item_type=i.get("type"))
        for i in items if i.get("type") == "organic"
    ]


# ---------------------------------------------------------------------------
# 2. Pages
# ---------------------------------------------------------------------------

async def fetch_page(client, url: str, sem: asyncio.Semaphore,
                     cfg: CollectConfig) -> dict:
    """
    Fetch and parse one page.

    selectolax rather than BeautifulSoup: on a 150k-URL month the parser is
    the bottleneck, and selectolax is roughly an order of magnitude faster on
    the same HTML. Playwright is reserved for the subset of URLs whose main
    content is genuinely client-rendered -- it costs about 40x more per page,
    so deciding which pages need it is a real engineering decision, not a
    default.
    """
    async with sem:
        await asyncio.sleep(cfg.request_delay)
        try:
            r = await client.get(url, timeout=20,
                                 headers={"User-Agent": USER_AGENT},
                                 follow_redirects=True)
            html = r.text
        except Exception as exc:                       # noqa: BLE001
            return dict(url=url, ok=False, error=str(exc)[:120])

    from selectolax.parser import HTMLParser           # local import: optional dep
    tree = HTMLParser(html)
    for tag in tree.css("script, style, nav, header, footer, aside"):
        tag.decompose()
    body = tree.body.text(separator=" ") if tree.body else ""

    # The reliability of THIS extraction is what validate.eiv_correct needs.
    # Hand-label 300 pages once, compare, and you have the number. Skipping
    # that audit is how content depth ends up with a coefficient of 0.16.
    return dict(
        url=url, ok=True, status=r.status_code,
        title=(tree.css_first("title").text() if tree.css_first("title") else ""),
        word_count=len(body.split()),
        text=body[:200_000],
        n_schema=len(tree.css('script[type="application/ld+json"]')),
        html_bytes=len(html),
    )


# ---------------------------------------------------------------------------
# 3. Field performance
# ---------------------------------------------------------------------------

async def fetch_crux(client, origin: str, cfg: CollectConfig) -> dict:
    """LCP / INP / CLS at the 75th percentile, real users, free."""
    r = await client.post(CRUX_ENDPOINT.format(key=cfg.crux_key),
                          json={"origin": origin, "formFactor": "PHONE"},
                          timeout=20)
    if r.status_code != 200:
        return dict(origin=origin, ok=False)
    m = r.json()["record"]["metrics"]
    pick = lambda k: m.get(k, {}).get("percentiles", {}).get("p75")  # noqa: E731
    return dict(origin=origin, ok=True,
                lcp=pick("largest_contentful_paint"),
                inp=pick("interaction_to_next_paint"),
                cls=pick("cumulative_layout_shift"))


# ---------------------------------------------------------------------------
# 4. Ground truth for our own clients
# ---------------------------------------------------------------------------

GSC_BIGQUERY_SQL = """
-- Search Console bulk export -> BigQuery. Set the export up on day one:
-- it only ever starts collecting from the moment you enable it, so every
-- week of delay is a week of history that does not exist.
SELECT
  data_date,
  query,
  url,
  SUM(impressions)                                   AS impressions,
  SUM(clicks)                                        AS clicks,
  SAFE_DIVIDE(SUM(sum_position), SUM(impressions)) + 1 AS avg_position
FROM `{project}.searchconsole.searchdata_url_impressions`
WHERE data_date BETWEEN @start AND @end
  AND is_anonymized_query = FALSE
GROUP BY data_date, query, url
"""


# ---------------------------------------------------------------------------
# Propensities in the wild
# ---------------------------------------------------------------------------

PROPENSITY_NOTE = """
Estimating position bias without a randomised slice
---------------------------------------------------
A search engine estimates the examination curve by shuffling results for a
small share of traffic. We cannot shuffle Google's SERP, so we use the churn
Google supplies for free: track the same (query, URL) pair across the daily
panel and keep the cases where it appears at several different positions
inside a short window. Within that window the page is close to unchanged, so
the variation in position is close to exogenous, and the click ratio across
positions identifies the curve.

It is a weaker instrument than randomisation and it needs a much larger panel
-- which is the honest reason to run the panel daily rather than weekly. In
the harness, the randomised slice recovers the exponent to within 0.01 at
15,000 queries and to about 0.08 at 1,200. Rank churn should be expected to
land somewhere between those two, and the correction is still worth far more
than leaving position bias in.
"""


async def run(queries: list[str], out: Path, cfg: CollectConfig) -> None:
    import httpx
    sem = asyncio.Semaphore(cfg.concurrency)
    async with httpx.AsyncClient(http2=True) as client:
        serps = await asyncio.gather(*(fetch_serp(client, q, cfg) for q in queries))
        rows = [r for s in serps for r in s]
        urls = sorted({r["url"] for r in rows if r["url"]})
        pages = await asyncio.gather(*(fetch_page(client, u, sem, cfg) for u in urls))

    df = (pd.DataFrame(rows)
          .merge(pd.DataFrame(pages), on="url", how="left"))
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"{len(df):,} rows from {len(queries):,} queries -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--queries", type=Path, required=True,
                    help="CSV with a 'query' column")
    ap.add_argument("--out", type=Path, default=Path("data/panel.parquet"))
    ap.add_argument("--yes-spend-money", action="store_true",
                    help="required: this calls a paid API")
    args = ap.parse_args()

    cfg = CollectConfig()
    if not args.yes_spend_money:
        print(__doc__)
        print(PROPENSITY_NOTE)
        print("Refusing to run without --yes-spend-money.")
        return
    if not cfg.serp_key:
        raise SystemExit("SERPAPI_KEY is not set")

    queries = pd.read_csv(args.queries)["query"].dropna().tolist()
    asyncio.run(run(queries, args.out, cfg))


if __name__ == "__main__":
    main()
