# SERP Factor Lab

A ranking-factor model that reports how wrong it is.

You asked whether a factor scoring 0.84 is really more important than one
scoring 0.73, and how you would test that. You cannot test it against Google,
because Google will not show you its weights. You *can* test it against a
search engine whose weights you set yourself — so this builds one, runs the
whole estimation pipeline at it, and marks its own work.

Whatever error the pipeline makes on a world it cannot cheat at is the floor
on the error it makes on the real one.

```bash
python python/run_demo.py          # ~80s, no install needed beyond numpy/pandas/scipy/sklearn
cd dashboard && npm install && npm run dev
```

Nothing below is hard-coded. Every number is printed by that command, and the
dashboard reads the same JSON the command writes — so a screenshot of the
terminal and a screenshot of the dashboard always agree.

---

## Results

### 1. Your two numbers, run through two pipelines

In the harness those numbers *are* the truth: content depth is set to **0.84**
and site authority to **0.73**.

| pipeline | content depth | site authority | verdict | says it needs |
|---|---|---|---|---|
| naive — what every SEO tool ships | **0.13** | 0.57 | "clear winner" | 115 queries |
| corrected — IPW + errors-in-variables + cluster bootstrap | **0.65** | 0.56 | "cannot separate these yet" | 36,230 queries |

The naive pipeline does not merely get the sizes wrong. **It has the ranking
backwards, and it reports a tight confidence interval while doing it.**
Confidently wrong is worse than unsure.

Then we ran the corrected pipeline at the size it asked for:

```
scale-up at 37,000 queries
  content depth 0.76   site authority 0.59
  difference 95% CI [+0.10, +0.24]   separated = True
```

The power calculation predicted what it would take, and it was right. That
loop — *quote a price for certainty, pay it, get certainty* — is the product.

### 2. Calibration: where the estimator fails, and by how much

| factor | truth | naive | corrected | verdict |
|---|---|---|---|---|
| Dense query-doc similarity | 1.00 | 1.00 | 1.00 | RECOVERED |
| Content depth vs SERP median | 0.84 | 0.13 | 0.65 | RECOVERED |
| Host-graph PageRank | 0.73 | 0.57 | 0.56 | MISSED |
| **KG entity coverage** | 0.66 | 0.50 | 0.50 | MISSED |
| Residual CTR | 0.59 | 0.16 | 0.45 | MISSED |
| Referring root domains | 0.41 | 0.34 | 0.33 | MISSED |
| E-E-A-T rubric | 0.34 | 0.28 | 0.28 | RECOVERED |
| **Keyword density** | **0.00** | **0.33** | **0.04** | FALSE POSITIVE → CLEAN |

Two failures are worth more than the eight that landed:

**The invented factor.** Keyword density has a true weight of exactly zero.
Uncorrected, the model gives it 0.33 and gives real content depth 0.13 — it
would tell a client that keyword density matters twice as much as content.
The cause is not collinearity, it is *measurement*: depth is hard to extract
cleanly, the decoy is trivial to extract, and a regression credits whichever
variable carries the cleanest signal rather than whichever one causes the
outcome. The fix is a re-measurement audit — hand-extract 300 pages, compare
against the crawler, feed the reliability into an errors-in-variables
correction. One afternoon of work, and the highest-return item in the project.

**The vanished factor.** The click signal reads 0.16 against a truth of 0.59.
The usual construction — observed CTR minus the average CTR at that position —
looks like it removes position bias and does the opposite: position is *caused
by* quality, so subtracting the positional mean subtracts the quality too.
Dividing by an examination probability estimated from a randomised slice
recovers most of it. The residual gap is irreducible click noise, and we ship
the factor with the gap attached.

**More data fixes neither.** Run the naive pipeline at 1,200 and at 15,000
queries and it returns the same wrong answer with a narrower interval. Bias
and variance are different problems and only one of them has a budget
solution.

### 3. Weights are not global

Clusters are discovered from SERP shape, never assumed — **adjusted Rand index
0.96** against the true grouping, with no labels.

```
Host-graph PageRank:  ymyl-finance 1.26   vs   local-service 0.37     = 3.4x
                      (true ratio 4.4x — we understate it, and say so)
```

A link-building retainer priced flat across a client book is overcharging some
of those clients and undercharging the rest. A global weight table cannot show
you which is which.

### 4. The uncomfortable one

Ranked by positions gained per unit of effort, the winner is **keyword
density** — a factor with a true weight of exactly zero. It wins because it is
cheap to change and correlated with something that works, and both the
fixed-effects and the matching estimator credit it, because both are
observational and inherit the measurement problem whole.

This is the entire argument for the governance rule: **L0 through L3 can all
be fooled by a badly measured instrument. Only L4 cannot**, because in a split
test you change the real thing rather than a proxy for it.

### 5. The knowledge graph — from a weight to a work order

A search engine ranks entities, not strings. `knowledge_graph.py` builds the
topic's entity graph out of the pages Google itself chose, and runs PageRank
over it — the same power iteration as the host graph, pointed at entities.

```
query: "business bank account uk"   9 pages -> 18 entities, 16 typed edges

entity linking -- one node, several surface forms:
  Banking licence  <- "banking licence", "full banking licence",
                      "authorised uk bank"

typed edges, not just adjacency:
  Banking licence --PROTECTED_BY-> FSCS

entity coverage, page ranked #2 : 0.84
entity coverage, thin page      : 0.14

brief generated for the thin page:
  Arranged overdraft, Cash deposit charge, Monthly account fee,
  Current Account Switch Service, FSCS, Banking licence, QuickBooks, Xero

client brand in this topic graph: absent
```

That is what makes it a knowledge graph rather than a word graph: **linking**
(three surface forms collapse to one node), **typing** (nodes carry ORG /
SCHEME / REGULATION / SOFTWARE), and **typed relations** rather than "these
words appeared near each other".

It is also a modelled factor — `kg_entity_coverage`, true weight 0.66 — and it
swings **3.2x** between clusters (0.74 informational, 0.23 local service).

**The weight says how hard to push; the graph says where.** Every other factor
produces a number an account manager reads. This one produces that number *and*
the list of entities a writer has to add. And when the client brand is absent
from the graph entirely, on-page work cannot fix it — that is a schema.org,
`sameAs`, Wikidata and citations problem, on a different budget.

In production: Cloud Natural Language `analyzeEntities` returns Google's own
entities plus a `salience` score (~$1/1k pages, free monthly tier), reconciled
to Wikidata — which is Google's own recommendation when you need a graph of
connected entities rather than single lookups.

---

## Technology

| area | what is used | where |
|---|---|---|
| Information retrieval | BM25, NDCG@k, MAP, Kendall τ, pairwise accuracy | `model.py` `validate.py` |
| Learning to rank | Bradley-Terry / RankNet **pairwise logistic on within-query differences** (hand-rolled IRLS); gradient-boosted ranker; LightGBM `lambdarank` when available | `model.py` |
| Unbiased LTR | **inverse-propensity weighting**, propensities identified from a randomised slice (Joachims' counterfactual framework) | `model.py` |
| Measurement error | multivariate **errors-in-variables** disattenuation from an instrument-reliability audit | `validate.py` |
| Statistical inference | **cluster bootstrap by query**, permutation noise floor, cluster-robust sandwich SEs, power analysis | `validate.py` |
| Causal inference | within-query **fixed effects**, near-twin matching, **difference-in-differences** across updates, DiD split tests with measured coverage | `causal.py` |
| Clustering | SERP-shape fingerprint → KMeans, scored by adjusted Rand index | `cluster.py` |
| Graph | host-graph **PageRank** by sparse power iteration | design in `config.py`, `collect.py` |
| Knowledge graph | entity linking across surface forms, typed relation extraction, **PageRank over the entity graph**, coverage and gap scoring | `knowledge_graph.py` |
| NLP | bi-encoder similarity, passage-level max similarity, LLM-as-judge against the Quality Rater Guidelines with Krippendorff α | `config.py`, `collect.py` |
| Data engineering | `asyncio` + `httpx` concurrency, `selectolax` parsing, Parquet, GSC → BigQuery | `collect.py` |
| Frontend | **Next.js 16 App Router, TypeScript, Tailwind 4**, hand-drawn SVG error bars | `dashboard/` |

**Python for everything that thinks, Next.js for everything that gets looked
at.** React versus Vue is not a modelling decision. Next.js specifically,
because every output of this system ends up as a URL somebody sends to a
client, and server rendering plus route-level caching is what makes that
cheap.

The Python side deliberately depends only on numpy, pandas, scipy,
scikit-learn, statsmodels and matplotlib. LightGBM is used if it is installed
and silently skipped if not. `run_demo.py` has to run on a laptop with no
setup, because a demo that needs a working afternoon to install is not a demo.

---

## Factors

35 factors in 6 groups, sorted by **evidence tier** rather than by folklore:

- **Tier A (20) — confirmed by Google.** Core Web Vitals (INP replaced FID in
  March 2024), HTTPS, mobile usability, spam policies, the helpful-content
  system folded into core in 2024, freshness, links.
- **Tier B (14) — trial record and leak.** Pandu Nayak's DOJ testimony
  confirming click data as a core signal and NavBoost's 13-month rolling
  window; the March 2024 Content Warehouse schema exposing `siteAuthority`,
  `hostAge`, `titlematchScore`, `OriginalContentScore`, and three separate
  date fields (`bylineDate` / `syntacticDate` / `semanticDate`) that make
  faked freshness detectable.
- **Tier C — refused.** Keyword density, absolute word count, GA bounce rate,
  Domain Authority as a lever, "LSI keywords", meta keywords. Listed
  explicitly, with reasons, because half of a factor list is the half you
  leave out.

Every entry carries a measurement method, a data source, a marginal cost and a
refresh cadence. A factor you cannot measure weekly at a price you can defend
is not a factor, it is an opinion.

Full table: `python/config.py`, or the **Factor bench** page in the dashboard.

---

## Method, in brief

**Google is a pipeline, not a function.** Modelling it as one weight vector is
the root error. Four stages, modelled separately:

```
candidate retrieval  →  base ranking  →  twiddler re-ranking  →  SERP assembly
   (classification)       (LTR)          (incl. one-sided          + AI Overview
                                          demotions)                 citation
                                                                  (binary model)
```

**Two heads, and a rule that connects them.**

- *Interpretable head*: pairwise logistic on within-query differences.
  Differencing inside a query gives you query fixed effects for free —
  everything about the query itself cancels. Its coefficients are the numbers
  you are allowed to quote as "0.84".
- *Performance head*: gradient-boosted ranker.
- **The rule**: if the boosted head beats the linear head by ≤3% NDCG, the
  linear weights describe the world and scalars may be quoted. Otherwise the
  world is non-linear and you must report partial dependence curves instead of
  numbers. Current run: the linear head wins by 0.8%, so scalars are
  legitimate here.

**Five rungs of validation.**

| | question | method |
|---|---|---|
| L0 | is the number stable? | cluster bootstrap **by query**, permutation noise floor |
| L1 | does it rank like Google? | group-held-out NDCG@10 vs honest baselines |
| L2 | does it travel? | unseen clusters and time windows; weight drift doubles as an update detector |
| L3 | is it causal? | within-query fixed effects, near-twin matching, DiD across updates |
| L4 | does it survive contact? | frozen-model forward prediction, and split tests on client URLs |

**Governance rule: nothing reaches a client deck without a verdict, an
interval, and a date it was last checked.** No factor that has not cleared L3
or L4 may drive a recommendation.

**Three different things people call "importance"** — predictive (reproduces
the order), causal (moves the page), actionable (moves the page per unit of
cost). They routinely disagree by a factor of two and only the third has a
budget attached. A tool that prints 0.84 without saying which it means is
selling a horoscope.

**Split tests have to be sized first.** The binding constraint is not
day-to-day rank wobble, which averages away over a longer window. It is
per-URL drift between the two periods — a competitor moved, a link died, a
page got re-crawled — which averages out only over URLs. Measured minimum
detectable effect at 28+28 days:

| URLs per arm | 30 | 60 | 120 | 200 | 300 |
|---|---|---|---|---|---|
| min detectable | 1.30 | 1.00 | 0.70 | 0.50 | 0.40 |

and our 95% intervals contain the truth **94.5% of the time** over 400
replications — the validator, validated.

---

## Cost of the real pilot

| item | monthly |
|---|---|
| SERP data — 15,000 queries × top-20 × 3 pulls/week @ $0.0006/unit | ~$234 |
| Crawling ~150k URLs on one small VPS | ~$50 |
| CrUX field data | free |
| Embeddings, run locally on CPU | free |
| LLM rubric scoring, monthly on a 2,000-page sample | ~$60 |
| **total** | **~$350–600** |

A rounding error against one month of one analyst's time, which is the actual
argument for doing it.

---

## Repo map

```
python/
  config.py       35-factor registry — the single source of truth
  simulate.py     the synthetic Google, whose weights we know
  model.py        two heads, IRLS solver, IPW, ranking metrics
  validate.py     bootstrap, noise floor, verdicts, power, separation test
  causal.py       fixed effects, matching, DiD, split tests, MDE curve
  cluster.py      SERP-shape fingerprint and per-cluster weights
  knowledge_graph.py  entity graph from the SERP: salience, coverage, briefs
  collect.py      the real data path — off by default, it costs money
  run_demo.py     one command, end to end
  sync_writeup.py keeps docs/index.html numbers tied to lab.json
dashboard/        Next.js app, reads the JSON run_demo.py writes
outputs/lab.json  every number on the dashboard
```

---

## What is real and what is not

Being explicit, because the distinction is the whole point:

- **Real**: every method, every estimator, every statistic. The IRLS solver,
  the bootstrap, the IPW correction, the errors-in-variables adjustment, the
  matching, the DiD, the power analysis. All of it runs and all of it is
  reproducible with `--seed`.
- **Synthetic**: the SERPs. That is deliberate — a harness with known ground
  truth is the only place these methods can be *audited* rather than merely
  applied. `collect.py` carries the live wiring (DataForSEO, CrUX, Search
  Console → BigQuery) and is disabled behind an explicit flag because it
  spends money and touches other people's servers.
- **Not measured at all**: the effort scores behind the ROI table. Those are a
  judgement call about how much work one standard deviation costs, labelled as
  such everywhere they appear. Yours would be better than mine — you have the
  delivery data.

Conclusions hold across seeds. `--seed 7`, `--seed 123` and `--seed 999` all
produce a naive pipeline that invents the decoy at 0.33–0.41 and loses the
real factor, and a corrected pipeline that recovers it.
