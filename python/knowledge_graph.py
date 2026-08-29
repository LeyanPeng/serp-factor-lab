"""
The knowledge graph -- modelling the topic the way a search engine models it.

Everything else in this lab attacks the problem from the model side: fit a
ranker, recover weights, put an interval on them. This file attacks it from
the other side, and it is the half that turns a weight into work somebody can
actually do on Monday.

The argument is short. A search engine does not rank strings, it ranks
entities and the relations between them. So a ranking model built only out of
term-overlap features is measuring the wrong object, and no amount of
correction will fix a feature that describes the wrong thing. "Content depth
= 0.84" tells an account manager a factor matters. It does not tell a writer
what to write.

What we can see
---------------
We cannot see Google's weights. We CAN see two things Google will hand over:

  1. Its ANSWER -- the ten pages it chose for a query. Those choices are a
     statement about what the topic is made of, written by Google rather than
     by us.
  2. Its READING of a page -- Cloud Natural Language `analyzeEntities`
     returns, for arbitrary text, the entities Google's own extractor finds
     and a `salience` score in [0, 1] saying how central each one is to the
     document. That is Google grading our page for us, per 1,000 characters,
     with a free monthly tier.

Pool (1), run it through (2), and you have the topic's entity graph as Google
sees it. That is what this module builds.

Why this is a knowledge graph and not a word graph
--------------------------------------------------
Three things, and the demo prints evidence of all three:

  linking   "FSCS" and "Financial Services Compensation Scheme" are one node,
            not two. Surface form is not identity. A co-occurrence graph over
            words cannot make that call; a graph over entities must.
  typing    nodes carry a type (ORG, SCHEME, REGULATION, SOFTWARE...), so
            "which regulator governs this topic" is a query you can run.
  relations edges are typed -- PROTECTED_BY, REGULATED_BY, INTEGRATES_WITH --
            rather than "these two words appeared near each other".

What comes out, and why an agency cares
---------------------------------------
  salience     PageRank over the entity graph. Which entities define the
               topic, ranked, derived from Google's own ten choices.
  coverage     share of the topic's salience mass a page carries. A scalar,
               so it drops into the ranking model beside every other factor
               as `kg_entity_coverage`.
  gap          WHICH salient entities the page is missing. Not a score -- a
               list. That is a content brief, generated rather than guessed.
  brand tie    is the client's brand an entity in this topic's graph at all,
               and how far from the core? If the answer is "not present",
               no amount of on-page work will fix it, and that is a different
               and much more expensive conversation to have early.

Production wiring
-----------------
The offline demo below uses a closed gazetteer so this file runs with no
network and no API key. In production the three steps map to:

  extract + link   Cloud Natural Language `analyzeEntities` (salience comes
                   free in the same response), reconciled against Wikidata
                   QIDs. Note that Google's own guidance is to use Wikidata
                   dumps when you need a graph of interconnected entities --
                   the Knowledge Graph Search API returns single entities and
                   is being migrated to Cloud Enterprise Knowledge Graph.
  relations        dependency patterns over the same text, plus schema.org
                   and `sameAs` triples already present in the page markup.
  brand tie        Knowledge Panel presence and Wikidata membership are the
                   observable end of whether Google has accepted the brand as
                   an entity.

Run it:  python python/knowledge_graph.py
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import numpy as np
from scipy import sparse

DAMPING = 0.85
TOL = 1e-12
MAX_ITER = 200


# ---------------------------------------------------------------------------
# Entity model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Entity:
    id: str                       # canonical id -- identity lives here
    name: str                     # display form
    type: str                     # ORG | GOV | SCHEME | REGULATION | ...
    aliases: tuple[str, ...] = ()
    wikidata: str = ""            # QID; filled by reconciliation in
                                  # production, deliberately blank offline
                                  # rather than guessed


# A closed gazetteer standing in for `analyzeEntities` + Wikidata
# reconciliation. Small on purpose: the point is the graph, not the NER.
GAZETTEER: list[Entity] = [
    Entity("fscs", "FSCS", "SCHEME",
           ("fscs", "financial services compensation scheme")),
    Entity("companies_house", "Companies House", "GOV", ("companies house",)),
    Entity("hmrc", "HMRC", "GOV", ("hmrc", "hm revenue and customs")),
    Entity("cass", "Current Account Switch Service", "SCHEME",
           ("current account switch service", "cass", "switch service",
            "switch guarantee")),
    Entity("banking_licence", "Banking licence", "REGULATION",
           ("banking licence", "banking license", "full banking licence",
            "authorised uk bank")),
    Entity("emoney", "E-money institution", "REGULATION",
           ("e-money", "emoney", "electronic money institution",
            "e-money institution", "safeguarding")),
    Entity("business_current_account", "Business current account", "PRODUCT",
           ("business current account", "business bank account",
            "business account")),
    Entity("overdraft", "Arranged overdraft", "PRODUCT",
           ("arranged overdraft", "overdraft")),
    Entity("monthly_fee", "Monthly account fee", "CONCEPT",
           ("monthly fee", "monthly account fee", "account fee")),
    Entity("cash_deposit", "Cash deposit charge", "CONCEPT",
           ("cash deposit", "cash deposits", "cash handling")),
    Entity("xero", "Xero", "SOFTWARE", ("xero",)),
    Entity("quickbooks", "QuickBooks", "SOFTWARE", ("quickbooks",)),
    Entity("psc", "Person with significant control", "REGULATION",
           ("significant control", "psc")),
    Entity("sole_trader", "Sole trader", "CONCEPT", ("sole trader", "sole traders")),
    Entity("limited_company", "Limited company", "CONCEPT",
           ("limited company", "limited companies")),
    Entity("invoice_finance", "Invoice finance", "PRODUCT", ("invoice finance",)),
    Entity("direct_debit", "Direct debit", "CONCEPT",
           ("direct debit", "direct debits")),
    Entity("challenger_bank", "Challenger bank", "ORG",
           ("challenger bank", "challenger banks", "digital challenger")),
    # The client. Present in the gazetteer, absent from the SERP -- which is
    # the finding.
    Entity("client_brand", "Northbank (client)", "ORG",
           ("northbank",)),
]

BY_ID = {e.id: e for e in GAZETTEER}

# Typed relation patterns. Order matters: first match wins.
RELATION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("PROTECTED_BY", re.compile(
        r"\b(protected|covered)\s+by\b|\bprotection\s+covers\b"
        r"|\bcover\s+only\s+applies\b|\bis\s+not\s+covered\b")),
    ("REGULATED_BY", re.compile(
        r"\b(regulated|authorised|authorized)\s+by\b|\bholds\s+a\s+full\b"
        r"|\bunder\s+safeguarding\s+rules\b")),
    ("INTEGRATES_WITH", re.compile(r"\bintegrat\w*\s+with\b")),
    ("OFFERED_BY", re.compile(r"\b(offered|provided|issued)\s+by\b")),
    ("REQUIRED_BY", re.compile(r"\b(required|expected|requested)\s+by\b")),
]


@dataclass
class KnowledgeGraph:
    ids: list[str]
    salience: np.ndarray                       # aligned with ids, sums to 1
    adjacency: sparse.csr_matrix
    edge_types: dict[tuple[str, str], set] = field(default_factory=dict)
    linked: dict[str, set] = field(default_factory=dict)   # id -> surface forms

    def name(self, eid: str) -> str:
        return BY_ID[eid].name

    def top(self, k: int = 12) -> list[tuple[str, float]]:
        order = np.argsort(-self.salience)[:k]
        return [(self.ids[i], float(self.salience[i])) for i in order]


# ---------------------------------------------------------------------------
# 1-2. extract + link: surface forms -> canonical entity ids
# ---------------------------------------------------------------------------

def link(text: str) -> dict[str, set]:
    """
    Map a document to the entities it mentions, recording which surface form
    produced each hit. Returns {entity_id: {surface forms seen}}.

    This is the step a word graph cannot perform: two different strings
    resolve to one identity, and one string could in principle resolve to
    different identities in different contexts.
    """
    low = " " + re.sub(r"[^a-z0-9 ]+", " ", text.lower()) + " "
    low = re.sub(r"\s+", " ", low)
    found: dict[str, set] = {}
    for ent in GAZETTEER:
        for alias in ent.aliases:
            if f" {alias} " in low:
                found.setdefault(ent.id, set()).add(alias)
    return found


def _sentences(text: str) -> list[str]:
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if s.strip()]


# ---------------------------------------------------------------------------
# 3-4. relate + rank
# ---------------------------------------------------------------------------

def build(docs: list[str], ranks: list[int] | None = None) -> KnowledgeGraph:
    """
    Build the topic's entity graph from the pages Google chose.

    Pages ranked higher get a larger vote in defining the topic, discounted
    by 1/log2(1+r) -- the same discount NDCG uses. Google put them in that
    order; declining to use that would be squeamish rather than rigorous.
    """
    if ranks is None:
        ranks = list(range(1, len(docs) + 1))
    weights = [1.0 / np.log2(1.0 + r) for r in ranks]

    linked_all: dict[str, set] = {}
    pair_w: dict[tuple[str, str], float] = {}
    edge_types: dict[tuple[str, str], set] = {}

    for doc, w in zip(docs, weights):
        for sent in _sentences(doc):
            hits = link(sent)
            if not hits:
                continue
            for eid, forms in hits.items():
                linked_all.setdefault(eid, set()).update(forms)

            rel = "CO_OCCURS"
            for name, pat in RELATION_PATTERNS:
                if pat.search(sent.lower()):
                    rel = name
                    break

            present = sorted(hits)
            for a in range(len(present)):
                for b in range(a + 1, len(present)):
                    key = (present[a], present[b])
                    pair_w[key] = pair_w.get(key, 0.0) + w
                    edge_types.setdefault(key, set()).add(rel)

    ids = sorted({e for pair in pair_w for e in pair} | set(linked_all))
    if not ids:
        raise ValueError("no entity linked in any document")
    index = {e: i for i, e in enumerate(ids)}
    n = len(ids)

    rows, cols, vals = [], [], []
    for (a, b), w in pair_w.items():
        i, j = index[a], index[b]
        rows += [i, j]
        cols += [j, i]
        vals += [w, w]
    A = sparse.coo_matrix((vals, (rows, cols)), shape=(n, n)).tocsr()
    A.sum_duplicates()

    # PageRank by power iteration -- the same routine host_pagerank runs over
    # the link graph, pointed at entities instead of hosts.
    deg = np.asarray(A.sum(axis=1)).ravel()
    dangling = deg == 0
    M = sparse.diags(1.0 / np.where(dangling, 1.0, deg)) @ A

    r = np.full(n, 1.0 / n)
    tele = np.full(n, 1.0 / n)
    for _ in range(MAX_ITER):
        nxt = DAMPING * (M.T @ r + r[dangling].sum() * tele) + (1 - DAMPING) * tele
        if np.abs(nxt - r).sum() < TOL:
            r = nxt
            break
        r = nxt

    return KnowledgeGraph(ids=ids, salience=r / r.sum(), adjacency=A,
                          edge_types=edge_types, linked=linked_all)


# ---------------------------------------------------------------------------
# 5. score
# ---------------------------------------------------------------------------

def coverage(page_text: str, kg: KnowledgeGraph, k: int = 14) -> float:
    """
    Share of the topic's salience mass this page carries. 1.0 means the page
    touches every entity the SERP agrees defines the topic. This is the
    scalar the ranking model consumes as `kg_entity_coverage`.
    """
    order = np.argsort(-kg.salience)[:k]
    present = set(link(page_text))
    mass = kg.salience[order]
    hit = np.array([kg.ids[i] in present for i in order], dtype=float)
    return float((mass * hit).sum() / mass.sum())


def gap(page_text: str, kg: KnowledgeGraph, k: int = 14,
        limit: int = 8) -> list[tuple[str, float, str]]:
    """The salient entities this page is missing, worst first. The brief."""
    order = np.argsort(-kg.salience)[:k]
    present = set(link(page_text))
    return [(kg.ids[i], float(kg.salience[i]), BY_ID[kg.ids[i]].type)
            for i in order if kg.ids[i] not in present][:limit]


def brand_tie(brand_id: str, kg: KnowledgeGraph) -> dict:
    """
    Is the client an entity in this topic's graph, and how central?

    'Not present' is the most useful answer this module produces. It means no
    amount of on-page editing moves the needle, because Google does not yet
    associate the brand with the topic at all. That is an entity-building
    problem -- schema.org, sameAs, Wikidata, citations -- on a different
    budget and a different timeline.
    """
    if brand_id not in kg.ids:
        return {"present": False, "salience": 0.0, "rank": None,
                "verdict": "absent from the topic graph"}
    i = kg.ids.index(brand_id)
    order = list(np.argsort(-kg.salience))
    return {"present": True, "salience": float(kg.salience[i]),
            "rank": order.index(i) + 1, "verdict": "present"}


# ---------------------------------------------------------------------------
# Offline demo -- one commercial SERP, in miniature. Page 9 is thin on purpose.
# ---------------------------------------------------------------------------

QUERY = "business bank account uk"

SERP_PAGES = [
    """Opening a business bank account in the UK takes about ten minutes online.
    You will need proof of identity, proof of address and your company
    registration number from Companies House. Most providers charge a monthly
    fee, though several offer an introductory free banking period. Deposits are
    protected by the Financial Services Compensation Scheme up to eighty-five
    thousand pounds per depositor.""",

    """A business current account keeps company money separate from personal
    money, which HMRC expects for a limited company. Look at the monthly fee,
    cash deposit charges, and whether an arranged overdraft is available.
    Integration with accounting software such as Xero or QuickBooks saves hours
    at year end. Eligible deposits are protected by the FSCS. Switching is
    handled by the Current Account Switch Service, which moves direct debits
    automatically.""",

    """Digital challenger banks changed business banking in the UK. They offer
    fast onboarding and no monthly fee on entry tiers. The trade-off is cash
    deposit handling and, for some providers, no arranged overdraft. Check
    whether the provider holds a full banking licence or operates as an e-money
    institution, because FSCS cover only applies to the former. Xero
    integration is usually strong.""",

    """To open a business current account you need your company registration
    number from Companies House, the details of every director and anyone with
    significant control, and expected turnover. Sole traders need proof of
    identity. Monthly fee, transaction charges and cash deposit limits are the
    three numbers worth comparing.""",

    """Business banking charges are rarely a single number. A typical structure
    is a monthly fee, a per-transaction charge, and a percentage on cash
    deposit. A free banking period reduces the first of these and none of the
    others. Model your own volume against the tariff.""",

    """An arranged overdraft on a business current account is credit and is
    priced accordingly. Providers assess turnover and the credit profile of the
    directors. Many challenger banks do not offer an overdraft at all. Compare
    the representative rate against invoice finance before committing.""",

    """The Current Account Switch Service moves your business current account
    between providers in seven working days, redirecting direct debits. Not
    every business account is covered by the switch guarantee, so confirm
    eligibility first.""",

    """FSCS protection covers eligible deposits held with an authorised UK bank
    up to eighty-five thousand pounds per depositor. Limited companies qualify.
    An e-money institution is not covered, and instead holds customer money
    under safeguarding rules. Check the banking licence before depositing large
    balances.""",

    # thin page, deliberately
    """We help you find the right business account. Our team compares the market
    so you do not have to. Fast, simple and free to use. Get started today and
    open your account in minutes with our trusted partners.""",
]


def demo() -> dict:
    kg = build(SERP_PAGES)
    strong, thin = SERP_PAGES[1], SERP_PAGES[-1]

    multi = sorted(
        ((eid, forms) for eid, forms in kg.linked.items() if len(forms) > 1),
        key=lambda kv: -len(kv[1]))

    # An edge can carry several relations across sentences. Drop the
    # CO_OCCURS fallback whenever a real relation was also observed --
    # otherwise it sorts first alphabetically and hides the typed one.
    typed = []
    for (a, b), t in kg.edge_types.items():
        real = sorted(t - {"CO_OCCURS"})
        if real:
            typed.append((a, b, real[0]))

    return {
        "query": QUERY,
        "n_pages": len(SERP_PAGES),
        "n_entities": len(kg.ids),
        "n_edges": int(kg.adjacency.nnz // 2),
        "salience": [{"entity": BY_ID[e].name, "type": BY_ID[e].type,
                      "salience": round(s, 4)} for e, s in kg.top(10)],
        "linking_evidence": [{"entity": BY_ID[e].name,
                              "surface_forms": sorted(f)} for e, f in multi[:3]],
        "typed_relations": [{"from": BY_ID[a].name, "rel": r,
                             "to": BY_ID[b].name} for a, b, r in typed[:6]],
        "strong_coverage": round(coverage(strong, kg), 3),
        "thin_coverage": round(coverage(thin, kg), 3),
        "thin_gap": [{"entity": BY_ID[e].name, "type": t,
                      "salience": round(s, 4)} for e, s, t in gap(thin, kg)],
        "brand": brand_tie("client_brand", kg),
    }


def print_demo(out: dict | None = None) -> dict:
    out = out or demo()
    print(f'  query: "{out["query"]}"   {out["n_pages"]} pages  ->  '
          f'{out["n_entities"]} entities, {out["n_edges"]} typed edges')

    print("\n  what the SERP says this topic is made of (PageRank over entities):")
    for row in out["salience"][:8]:
        print(f'    {row["salience"]:.3f}  {row["entity"]:<32} {row["type"]}')

    print("\n  entity linking -- one node, several surface forms:")
    for row in out["linking_evidence"]:
        print(f'    {row["entity"]:<32} <- {", ".join(row["surface_forms"])}')
    print("    a word graph would have scored those as unrelated strings.")

    if out["typed_relations"]:
        print("\n  typed edges, not just adjacency:")
        for row in out["typed_relations"]:
            print(f'    {row["from"]} --{row["rel"]}-> {row["to"]}')

    print(f'\n  entity coverage, page ranked #2 : {out["strong_coverage"]:.2f}')
    print(f'  entity coverage, thin page      : {out["thin_coverage"]:.2f}')

    print("\n  the brief this generates for the thin page -- entities the SERP")
    print("  carries and it does not:")
    for row in out["thin_gap"]:
        print(f'    {row["entity"]:<34} {row["type"]}')

    b = out["brand"]
    print(f'\n  client brand in this topic graph: {b["verdict"]}')
    print("  That is the expensive finding. If the brand is not an entity in")
    print("  the topic, on-page work cannot fix it -- that is a schema.org,")
    print("  sameAs, Wikidata and citations problem, on a different budget.")
    return out


if __name__ == "__main__":
    print_demo()
