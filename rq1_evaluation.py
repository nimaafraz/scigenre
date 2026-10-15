"""Stage 3 (RQ1): SciGenre against keyword rules, database document types, TF-IDF with a linear SVM, Smyth &
Cunningham's rule and the LLM baseline on the held-out test papers, checks of the venue labels, and the replication
of Smyth & Cunningham's published labels.

    python rq1_evaluation.py baselines [fetch|evaluate|all]   SciGenre against the baselines
    python rq1_evaluation.py label-checks                     checks of the venue labels
    python rq1_evaluation.py sc-replication fetch|evaluate    Smyth & Cunningham's published labels
"""
import json
import os
import re
import sys
import time
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.proportion import proportion_confint

from common import clean_text, model_text, openalex_get_all, PAPER_TITLE_KEYWORDS

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# Smyth & Cunningham's review rule
# =====================================================================================================================
#
# The review-paper rule of Smyth & Cunningham (2026), "Have we reached the beginning of the end for review papers?",
# Quantitative Science Studies 7, 1044-1061, doi:10.1162/QSS.a.488.
#
# Reference implementation: the authors' released code, review_paper_rule.py (it reproduces the published counts of
# their 17M data set), imported unchanged. It is not redistributed with this pipeline: obtain it from the authors and
# place it at inputs/smyth_cunningham/review_paper_rule.py (see README.md). `released_record` turns one of our cached
# Semantic Scholar records into the record their `classify` reads, and `is_review_released` applies it. Their code reads
# the venue and journal names only, uses separate phrase lists for titles and abstracts, and requires at least one
# reference.
#
# Everything below `released_record` is our earlier implementation from the published description (Section 4.1.2 and
# Supplementary Table S1, MODE = "published"), kept only as a sensitivity check.
#
# A paper is a review if
#   1. Semantic Scholar's publicationTypes contains Review or MetaAnalysis, and
#   2. its title or abstract contains one of the 48 strict review phrases of Table S1 ("exact matches with titles and
#      abstracts": the phrase appears verbatim, ignoring case), or it is in a review venue, a venue or journal whose
#      name contains "review", "survey" or "trends in" (e.g. Computing Surveys, Trends in Cell Biology).
# Every review gets one subtype:
#   meta        publicationTypes contains MetaAnalysis, or the title or abstract contains a strict phrase with "meta";
#   systematic  not meta, and the title or abstract contains a strict phrase with "systematic";
#   narrative   any other review.
# Other papers are "regular" (their dataset calls them "original").
#
# Their study population is not part of the rule: Semantic Scholar records typed JournalArticle, with an abstract of
# at least five words and at least one reference, published 2000-2024 (`in_study_population`).
#
# Not the default: MODE = "as_run" is a reconstruction from their published labels (figshare
# doi:10.6084/m9.figshare.32077821; rq1_evaluation.py sc-replication, hcr_study/sc_replication/). It matches abstracts against a
# narrower list than Table S1 (generic phrases such as "overview of" and the three-word "systematic literature ..."
# phrases count in titles only, "a review of" and "a tutorial" in abstracts) and agrees with their labels for 99.87%
# of 26,589 re-fetched Computer Science papers, against 98.19% for the published description. It is kept only for
# sensitivity checks.

_RPR = None


def sc_code():
    """The authors' released code, loaded on first use from inputs/smyth_cunningham/review_paper_rule.py."""
    global _RPR
    if _RPR is None:
        path = os.path.join(HERE, "inputs", "smyth_cunningham")
        if not os.path.exists(os.path.join(path, "review_paper_rule.py")):
            raise SystemExit(f"Smyth & Cunningham's code is missing: place review_paper_rule.py in {path} (README.md)")
        sys.path.insert(0, path)
        import review_paper_rule
        _RPR = review_paper_rule
    return _RPR


# Supplementary Table S1, in the published order
PHRASES = [
    "literature review", "literature survey", "literature study", "literature analysis",
    "comparative review", "comparative survey", "comparative study", "comparative analysis",
    "bibliographic review", "bibliographic survey", "bibliographic study", "bibliographic analysis",
    "bibliometric review", "bibliometric survey", "bibliometric study", "bibliometric analysis",
    "scientometric review", "scientometric survey", "scientometric study", "scientometric analysis",
    "systematic review", "systematic survey", "systematic study", "systematic analysis",
    "systematic literature review", "systematic literature survey",
    "systematic literature study", "systematic literature analysis",
    "meta analysis", "meta review", "meta study", "meta-analysis", "meta-review", "meta-study",
    "mapping study", "systematic mapping",
    "emerging trends", "comprehensive survey", "contemporary survey",
    "systematization of knowledge", "systematisation of knowledge",
    "tutorial", "review of", "research directions", "a review",
    "case studies", "overview of", "summary of",
]
assert len(PHRASES) == 48

# Not the default: the abstract list that reproduces their published labels (see the module docstring)
TITLE_ONLY = {"overview of", "review of", "summary of", "research directions", "case studies", "a review", "tutorial",
              "systematic literature review", "systematic literature survey", "systematic literature study",
              "systematic literature analysis"}
ABSTRACT_PHRASES_AS_RUN = [p for p in PHRASES if p not in TITLE_ONLY] + ["a review of", "a tutorial"]

REVIEW_VENUE_TERMS = ("review", "survey", "trends in")
REVIEW_TYPES = {"Review", "MetaAnalysis"}
MODE = "published"         # Table S1 on titles and abstracts, as described; "as_run" only for sensitivity checks


def _str(x) -> str:
    return "" if x is None or (isinstance(x, float) and x != x) else str(x)


def _types(types) -> List[str]:
    """publicationTypes as a list; accepts a list, None/NaN, or a string such as 'JournalArticle; Review'."""
    if types is None or (isinstance(types, float) and types != types):
        return []
    if isinstance(types, str):
        return [t.strip() for t in re.split(r"[;,|]", types) if t.strip()]
    return [str(t) for t in types]


class Rule:
    def __init__(self, mode: str = MODE, match: str = "substring"):
        """mode: "published" (default) or "as_run". match: "substring" (verbatim, ignoring case) or "word"."""
        self.mode, self.match = mode, match
        self.title_phrases = list(PHRASES)
        self.abstract_phrases = list(PHRASES) if mode == "published" else list(ABSTRACT_PHRASES_AS_RUN)
        if match == "word":
            self._word = {p: re.compile(rf"\b{re.escape(p)}\b") for p in set(self.title_phrases) | set(self.abstract_phrases)}

    def _found(self, phrases: Iterable[str], text: str) -> set:
        if self.match == "word":
            return {p for p in phrases if self._word[p].search(text)}
        return {p for p in phrases if p in text}

    def phrases(self, title, abstract) -> set:
        """Strict review phrases found in the title and abstract."""
        return self._found(self.title_phrases, _str(title).lower()) | self._found(self.abstract_phrases, _str(abstract).lower())

    @staticmethod
    def is_review_venue(venue) -> bool:
        v = _str(venue).lower()
        return any(t in v for t in REVIEW_VENUE_TERMS)

    def paper_type(self, types, title, abstract, venue) -> str:
        """'regular', 'narrative', 'systematic' or 'meta'."""
        ts = set(_types(types))
        if not ts & REVIEW_TYPES:
            return "regular"
        found = self.phrases(title, abstract)
        if not (found or self.is_review_venue(venue)):
            return "regular"
        if "MetaAnalysis" in ts or any("meta" in p for p in found):
            return "meta"
        if any("systematic" in p for p in found):
            return "systematic"
        return "narrative"

    def is_review(self, types, title, abstract, venue) -> bool:
        return self.paper_type(types, title, abstract, venue) != "regular"

    def classify(self, df: pd.DataFrame, types="publicationTypes", title="title", abstract="abstract",
                 venue="venue") -> pd.Series:
        """paper_type for every row of a frame."""
        return pd.Series([self.paper_type(t, ti, a, v) for t, ti, a, v in
                          zip(df[types], df[title], df[abstract], df[venue])], index=df.index)


RULE = Rule()
paper_type = RULE.paper_type
is_review = RULE.is_review
is_review_venue = Rule.is_review_venue


def in_study_population(types, abstract, reference_count: Optional[int], year: Optional[int]) -> bool:
    """Their 17M data set: JournalArticle, abstract of at least five words, at least one reference, 2000-2024."""
    return ("JournalArticle" in _types(types) and len(_str(abstract).split()) >= 5 and (reference_count or 0) > 0
            and year is not None and year == year and 2000 <= int(year) <= 2024)


def s2_venue(rec: Optional[dict]) -> str:
    """Semantic Scholar's venue, journal and publication-venue names (cached keys of analyses.py s2-types)."""
    rec = rec or {}
    return " | ".join(_str(rec.get(k)) for k in ("venue", "journal", "publication_venue"))


def from_s2_record(rec: Optional[dict], title=None, abstract=None):
    """(types, title, abstract, venue) for a cached Semantic Scholar record (analyses.py s2-types). Their data set only holds
    papers with an abstract, so our title and abstract fill in where Semantic Scholar's API returns none. A paper
    that Semantic Scholar does not return is untyped and therefore regular."""
    rec = rec or {}
    s2_abs = _str(rec.get("abstract"))
    return (rec.get("types") or [], _str(rec.get("title")) or _str(title),
            s2_abs if len(s2_abs.split()) >= 5 else _str(abstract), s2_venue(rec))


def released_record(rec: Optional[dict], title=None, abstract=None, venue_extra: str = "") -> dict:
    """A cached Semantic Scholar record (rq1_evaluation.py baselines, analyses.py s2-types) as the record the authors' `classify` reads.
    Their data set only holds papers with an abstract, so our title and abstract fill in where Semantic Scholar's API
    returns none. venue_extra is appended to the venue name (the full venue name, for the sensitivity check)."""
    rec = rec or {}
    s2_abs = _str(rec.get("abstract"))
    return {"title": _str(rec.get("title")) or _str(title),
            "abstract": s2_abs if len(s2_abs.split()) >= 5 else _str(abstract),
            "venue": " ".join(v for v in (_str(rec.get("venue")), _str(venue_extra)) if v),
            "journal": {"name": _str(rec.get("journal"))},
            "publicationtypes": rec.get("types") or [],
            "referencecount": rec.get("reference_count")}


def is_review_released(rec: Optional[dict], title=None, abstract=None, venue_extra: str = "") -> bool:
    """The authors' rule on a cached Semantic Scholar record. A paper that Semantic Scholar does not return is
    untyped and therefore not a review."""
    return rec is not None and bool(sc_code().classify(released_record(rec, title, abstract, venue_extra))["is_review"])


def is_review_array(types, titles, abstracts, venues, rule: Rule = RULE) -> np.ndarray:
    return np.array([rule.is_review(t, ti, a, v) for t, ti, a, v in zip(types, titles, abstracts, venues)], dtype=bool)


# =====================================================================================================================
# Error correction and database baselines
# =====================================================================================================================
#
# Error correction and the baselines that profiles.py classify records for every paper:
#   * corrected_shares      adjusted classify-and-count (Forman 2008) from the model's recall and false-positive rate
#   * indexer_is_review     the document type assigned by the indexer (OpenAlex) says "review"
#   * smyth_cunningham_rule Smyth & Cunningham's rule (rq1_evaluation.py), here on OpenAlex types and venues

# Smyth & Cunningham (2026), Quantitative Science Studies 7:1044-1061, Section 4.1.2 and Supplementary Table S1.
# One implementation, in rq1_evaluation.py, checked against their published labels (rq1_evaluation.py sc-replication).


def corrected_shares(flagged: float, cites_flagged: float, tpr, fpr):
    """
    Adjusted classify-and-count: the true survey share of papers, pi = (q - FPR) / (TPR - FPR), and of citations,
    PPV * (citation share of flagged papers) + FOR * (citation share of the others), with PPV = TPR pi / q and
    FOR = (1 - TPR) pi / (1 - q). Works on arrays of (TPR, FPR) samples as well.
    """
    q = flagged
    pi = np.clip((q - fpr) / np.maximum(tpr - fpr, 1e-9), 0, 1)
    ppv = np.clip(tpr * pi / q, 0, 1) if q > 0 else 0.0
    omit = np.clip((1 - tpr) * pi / (1 - q), 0, 1) if q < 1 else 0.0
    return pi, ppv * cites_flagged + omit * (1 - cites_flagged)


def indexer_is_review(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: the document type assigned by the indexer (OpenAlex / Semantic Scholar) says 'review'."""
    return frame["Type"].str.contains(r"\breview\b", case=False, regex=True).astype(int).values


def _s2_style_types(t: str) -> list:
    """'JournalArticle; Review' -> ['JournalArticle', 'Review']; OpenAlex 'review' -> ['Review'] (approximation)."""
    out = []
    for x in re.split(r"[;,|]", t or ""):
        x = x.strip()
        k = x.lower().replace("-", "").replace(" ", "")
        out.append("Review" if k == "review" else "MetaAnalysis" if k == "metaanalysis" else x)
    return out


def smyth_cunningham_rule(frame: pd.DataFrame) -> np.ndarray:
    """Baseline: Smyth & Cunningham's rule. Faithful when frame["Type"] holds Semantic Scholar publicationTypes and
    frame["Venue"] Semantic Scholar's venue; with OpenAlex types and venues it is an approximation."""
    return is_review_array(frame["Type"].map(_s2_style_types), frame["Title"], frame["Abstract"],
                           frame["Venue"]).astype(int)


# =====================================================================================================================
# SciGenre against the baselines on the held-out test papers
# =====================================================================================================================
#
# RQ1 on the one venue ground truth: the final SciGenre (train.py final) against keyword rules, database
# document types, the Smyth & Cunningham rule (the authors' released code, rq1_evaluation.py) and TF-IDF with a linear
# SVM, all scored on the held-out test split (10% of papers, stratified by venue, never used for training or selection).
#
# Semantic Scholar records (types, title, abstract, venue names, reference count) are fetched by DOI for the test papers;
# papers that Semantic Scholar does not return count as untyped, as in Smyth & Cunningham's rule. Semantic Scholar often
# stores a venue under an abbreviation (e.g. "ACM Comput. Surv."), which contains none of the rule's venue terms, so the
# rule is also run with each paper's full venue name (OpenAlex source name) added to Semantic Scholar's. The rule as
# implemented from the published description (one phrase list for titles and abstracts) is a further sensitivity check.
# Precision and F1 are also given at a survey rate of 8%, close to that of the HCR-listed researchers' profiles.
# Usage: rq1_evaluation.py baselines [fetch|evaluate|all]     Output: hcr_study/venue_all/rq0_results.{json,md}, rq0_by_venue.csv

load_dotenv(os.path.join(HERE, ".env"), override=True)
GT = os.path.join(HERE, "hcr_study", "venue_all"); S2_CACHE = os.path.join(GT, "s2_test.json")
PI = 0.08
ABSTRACT_WORDS = re.compile(r"\b(?:survey|surveys|surveyed|review|reviews|reviewed|reviewing|overview)\b", re.I)


def load_test_split():
    G = pd.read_csv(os.path.join(GT, "ground_truth.csv"), dtype={"OpenAlexId": str, "doi": str})
    G = G[G.usable].merge(pd.read_csv(os.path.join(GT, "split.csv"), dtype={"OpenAlexId": str}), on="OpenAlexId")
    G["doi"] = G.doi.fillna("").str.replace("https://doi.org/", "", regex=False).str.lower()
    G["y"] = (G.label == "survey").astype(int)
    return G


def fetch_s2_test():
    G = load_test_split(); T = G[(G.split == "test") & (G.doi != "")]
    cache = json.load(open(S2_CACHE)) if os.path.exists(S2_CACHE) else {}
    # papers cached before reference counts were needed get only their reference count; other fields stay as first fetched
    todo = [d for d in T.doi.unique() if d not in cache or (cache[d] is not None and "reference_count" not in cache[d])]
    key = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip(); last = 0.0
    print(f"Semantic Scholar: {len(todo)} DOIs", flush=True)
    for i in range(0, len(todo), 500):
        chunk = todo[i:i + 500]
        for attempt in range(10):
            wait = 1.1 - (time.time() - last)
            if wait > 0: time.sleep(wait)
            last = time.time()
            r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch",
                              params={"fields": "title,abstract,publicationTypes,venue,journal,publicationVenue,referenceCount"},
                              json={"ids": ["DOI:" + d for d in chunk]}, headers={"x-api-key": key} if key else {}, timeout=120)
            if r.status_code == 200: break
            time.sleep(min(60, 3 * 2 ** attempt))
        for d, x in zip(chunk, r.json()):
            if cache.get(d) is not None:
                cache[d]["reference_count"] = None if x is None else x.get("referenceCount")
                continue
            cache[d] = None if x is None else {"types": x.get("publicationTypes") or [], "venue": x.get("venue") or "",
                                               "journal": (x.get("journal") or {}).get("name") or "",
                                               "publication_venue": (x.get("publicationVenue") or {}).get("name") or "",
                                               "title": x.get("title") or "", "abstract": x.get("abstract") or "",
                                               "reference_count": x.get("referenceCount")}
        json.dump(cache, open(S2_CACHE, "w"))
    print("FETCH DONE", flush=True)


def source_names(ids):
    """OpenAlex display names of the ground-truth venues (cached in source_names.json)."""
    path = os.path.join(GT, "source_names.json")
    names = json.load(open(path)) if os.path.exists(path) else {}
    todo = sorted(set(ids) - set(names))
    if todo:
        for i in range(0, len(todo), 50):
            for x in openalex_get_all("sources", {"filter": "ids.openalex:" + "|".join(todo[i:i + 50]), "select": "id,display_name"}):
                names[x["id"].rsplit("/", 1)[-1]] = x["display_name"]
        json.dump(names, open(path, "w"), indent=1)
    return names


BASELINES = {"SciBERT, fine-tuned": "scibert", "BERT, fine-tuned": "bert",
             "gpt-oss-120b, zero-shot": "llm_gpt-oss-120b"}


def ap8(y, s):
    """Average precision with precision reweighted to an 8% survey rate."""
    order = np.argsort(-s); y = np.asarray(y)[order]
    tp, fp = np.cumsum(y), np.cumsum(1 - y); P, N = y.sum(), (1 - y).sum()
    rec, fpr = tp / P, fp / N
    prec = rec * PI / np.maximum(rec * PI + fpr * (1 - PI), 1e-12)
    return float(np.sum(np.diff(np.r_[0, rec]) * prec))


def scores(y, f):
    y, f = np.asarray(y, bool), np.asarray(f, bool)
    fp, n_r, tp, n_s = int((f & ~y).sum()), int((~y).sum()), int((f & y).sum()), int(y.sum())
    fpr, rec = fp / n_r, tp / n_s
    p8 = rec * PI / (rec * PI + fpr * (1 - PI)) if rec > 0 else 0.0
    return {"research flagged": fp, "research": n_r, "FPR %": 100 * fpr,
            "FPR CI": [100 * x for x in proportion_confint(fp, n_r, method="wilson")],
            "surveys found": tp, "surveys": n_s, "recall %": 100 * rec,
            "recall CI": [100 * x for x in proportion_confint(tp, n_s, method="wilson")],
            "precision %": 100 * tp / max(tp + fp, 1), "precision at 8% %": 100 * p8,
            "F1 at 8% %": 100 * 2 * p8 * rec / max(p8 + rec, 1e-12), "accuracy %": 100 * (f == y).mean()}


def evaluate_baselines():
    G = load_test_split(); tr, te = G[G.split == "train"], G[G.split == "test"].copy()
    P = pd.read_csv(os.path.join(HERE, "hcr_study", "models_final", "seed42", "preds.csv"), dtype={"OpenAlexId": str})
    te = te.merge(P[P.split == "test"][["OpenAlexId", "p"]], on="OpenAlexId", how="left")
    assert te.p.notna().all()
    cache = json.load(open(S2_CACHE)) if os.path.exists(S2_CACHE) else {}
    rec = [cache.get(d) if d else None for d in te.doi]
    te["s2_found"] = [r is not None for r in rec]
    full = te.source_id.map(source_names(te.source_id.unique())).fillna("").str.replace("®", "", regex=False)   # "Trends® in" -> "Trends in"
    title, abstract = te.title.fillna("").map(clean_text), te.abstract.fillna("").map(clean_text)
    # TF-IDF + linear SVM on the same training papers and the same input text as the final model
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=500000)
    Xtr = vec.fit_transform([model_text(t, a) for t, a in zip(tr.title, tr.abstract)])
    svm = LinearSVC(class_weight="balanced", C=1.0, random_state=42).fit(Xtr, tr.y)
    svm_score = svm.decision_function(vec.transform([model_text(t, a) for t, a in zip(te.title, te.abstract)]))
    # baselines trained or run on UCD Sonic (bash run.sh sonic train, llm): encoders fine-tuned with the final model's data, split and recipe
    # (train.py final --model), and zero-shot open LLMs (train.py llm, p = 1 for an answer of survey)
    base_p = {}
    for name, tag in BASELINES.items():
        path = os.path.join(HERE, "hcr_study", "models_baselines", tag, "preds.csv")
        if os.path.exists(path):
            B = pd.read_csv(path, dtype={"OpenAlexId": str}); B = B[B.split == "test"].set_index("OpenAlexId")
            base_p[name] = te.OpenAlexId.map(B.p).values
            assert not np.isnan(base_p[name]).any(), name
    methods = {
        "Title keyword rule": title.map(lambda t: bool(PAPER_TITLE_KEYWORDS.search(t))).values,
        "Title or abstract keyword rule": np.array([bool(PAPER_TITLE_KEYWORDS.search(t) or ABSTRACT_WORDS.search(a)) for t, a in zip(title, abstract)]),
        "OpenAlex review type": (te["type"].fillna("").str.lower() == "review").values,
        "Semantic Scholar review type": np.array([bool(r) and bool(set(r["types"]) & REVIEW_TYPES) for r in rec]),
        "Smyth & Cunningham rule": np.array([is_review_released(r, t, a) for r, t, a in zip(rec, te.title, te.abstract)]),
        "Smyth & Cunningham rule, full venue names": np.array([is_review_released(r, t, a, v) for r, t, a, v in zip(rec, te.title, te.abstract, full)]),
        "TF-IDF + linear SVM": svm_score > 0,
        **{name: p >= 0.5 for name, p in base_p.items()},
        "SciGenre (ours)": (te.p >= 0.5).values,
    }
    out = {"test papers": len(te), "test surveys": int(te.y.sum()), "test venues": int(te.venue.nunique()),
           "semantic scholar coverage %": 100 * te.s2_found.mean(), "methods": {}}
    ours_ok = methods["SciGenre (ours)"] == te.y.astype(bool).values
    for name, f in methods.items():
        r = scores(te.y, f)
        if name != "SciGenre (ours)":
            ok = f == te.y.astype(bool).values
            t = [[int((ours_ok & ok).sum()), int((ours_ok & ~ok).sum())], [int((~ours_ok & ok).sum()), int((~ours_ok & ~ok).sum())]]
            r["McNemar vs ours"] = {"ours right, other wrong": t[0][1], "ours wrong, other right": t[1][0], "p": float(mcnemar(t, exact=True).pvalue)}
        out["methods"][name] = r
    # within their study population (Semantic Scholar journal articles 2000-2024 with an abstract and a reference)
    pop = np.array([r is not None and in_study_population(r.get("types"), released_record(r, t, a)["abstract"], r.get("reference_count"), yr)
                    for r, t, a, yr in zip(rec, te.title, te.abstract, te.year)])
    out["within Smyth & Cunningham's study population"] = {
        "surveys": int((te.y.values.astype(bool) & pop).sum()), "research": int((~te.y.values.astype(bool) & pop).sum()),
        **{name: scores(te.y[pop], methods[name][pop]) for name in ("Smyth & Cunningham rule", "Smyth & Cunningham rule, full venue names", "SciGenre (ours)")}}
    # why the rule misses surveys
    def why(r, t, a, v):
        if r is None: return "not in Semantic Scholar"
        c = sc_code().classify(released_record(r, t, a, v))
        if c["is_review"]: return "found"
        if not (c["has_review_type"] or c["has_meta_analysis_type"]): return "no Review or MetaAnalysis type"
        if not c["has_references"]: return "no references in Semantic Scholar"
        return "no review phrase, venue not matched"
    ys = te.y.values.astype(bool)
    out["surveys by outcome of the rule"] = {
        lab: pd.Series([why(r, t, a, v) for r, t, a, v, s_ in zip(rec, te.title, te.abstract, vv, ys) if s_]).value_counts().to_dict()
        for lab, vv in (("Semantic Scholar venue names", [""] * len(te)), ("full venue names", full))}
    # leave-venue-out (train.py holdout): SciGenre, and TF-IDF + SVM trained the same way, without the paper's venue
    VH = os.path.join(HERE, "hcr_study", "models_venue_holdout")
    if os.path.exists(os.path.join(VH, "summary.json")):
        H = pd.concat([pd.read_csv(os.path.join(VH, f"fold{k}", "preds.csv"), dtype={"OpenAlexId": str}) for k in (1, 2, 3)])
        p_unseen = te.OpenAlexId.map(H[~H.venue_seen].set_index("OpenAlexId").p).values
        assert not np.isnan(p_unseen).any()
        fold_of = pd.read_csv(os.path.join(VH, "folds.csv")).set_index("venue").fold
        tfidf_unseen = np.zeros(len(te))
        for k in (1, 2, 3):
            trk, tek = tr[tr.venue.map(fold_of) != k], (te.venue.map(fold_of) == k).values
            vk = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, max_features=500000)
            sk = LinearSVC(class_weight="balanced", C=1.0, random_state=42).fit(vk.fit_transform([model_text(t, a) for t, a in zip(trk.title, trk.abstract)]), trk.y)
            tfidf_unseen[tek] = sk.decision_function(vk.transform([model_text(t, a) for t, a in zip(te.title[tek], te.abstract[tek])]))
        te["p_venue_held_out"], te["tfidf_venue_held_out"] = p_unseen, tfidf_unseen
        unseen = {"SciGenre, venue held out": p_unseen >= 0.5, "TF-IDF + linear SVM, venue held out": tfidf_unseen > 0}
        yb = te.y.values.astype(bool); ours_u = unseen["SciGenre, venue held out"] == yb
        block = {}
        for name, f in unseen.items():
            block[name] = scores(te.y, f)
        for name in ("Smyth & Cunningham rule", "Smyth & Cunningham rule, full venue names", "Semantic Scholar review type",
                     "TF-IDF + linear SVM, venue held out"):
            other = (unseen.get(name, methods.get(name))) == yb
            t = [[int((ours_u & other).sum()), int((ours_u & ~other).sum())], [int((~ours_u & other).sum()), int((~ours_u & ~other).sum())]]
            block[f"McNemar, venue held out, vs {name}"] = {"ours right, other wrong": t[0][1], "ours wrong, other right": t[1][0],
                                                            "p": float(mcnemar(t, exact=True).pvalue)}
        out["venue held out (test papers)"] = block
        te[["OpenAlexId", "p_venue_held_out", "tfidf_venue_held_out"]].to_csv(os.path.join(GT, "rq0_venue_held_out.csv"), index=False)
    published = np.array([is_review(*from_s2_record(r, t, a)) for r, t, a in zip(rec, te.title, te.abstract)])
    out["sensitivity: Smyth & Cunningham rule as published (one phrase list)"] = scores(te.y, published)
    out["semantic scholar reference count missing or zero, among returned papers"] = int(sum(1 for r in rec if r is not None and not r.get("reference_count")))
    out["methods"]["SciGenre (ours)"]["AP8 %"] = 100 * ap8(te.y, te.p.values)
    out["methods"]["TF-IDF + linear SVM"]["AP8 %"] = 100 * ap8(te.y, svm_score)
    for name, p in base_p.items():
        if "zero-shot" not in name: out["methods"][name]["AP8 %"] = 100 * ap8(te.y, p)
    # seed variation: each encoder trained with seeds 42-46 (seed 42 is the paper's SciGenre and the McNemar tests)
    SEED_DIRS = {"SciGenre (ours)": lambda sd: os.path.join(HERE, "hcr_study", "models_final", f"seed{sd}"),
                 "SciBERT, fine-tuned": lambda sd: os.path.join(HERE, "hcr_study", "models_baselines", "scibert" if sd == 42 else f"scibert_seed{sd}"),
                 "BERT, fine-tuned": lambda sd: os.path.join(HERE, "hcr_study", "models_baselines", "bert" if sd == 42 else f"bert_seed{sd}")}
    out["seed variation (test)"] = {}
    for name, d in SEED_DIRS.items():
        seeds = {sd: json.load(open(m))["test"] for sd in (42, 43, 44, 45, 46) if os.path.exists(m := os.path.join(d(sd), "metrics.json"))}
        if len(seeds) > 1:
            out["seed variation (test)"][name] = {"seeds": sorted(seeds), **{k: {"mean": float(np.mean(v)), "sd": float(np.std(v, ddof=1)), "min": float(min(v)), "max": float(max(v))}
                                                  for k in ("recall", "FPR", "F1_8") for v in [[seeds[sd][k] for sd in sorted(seeds)]]}}
    # by abstract source and by venue (final model)
    f = methods["SciGenre (ours)"]
    out["by abstract source"] = {s: scores(d.y, f[d.index]) for s, d in te.reset_index(drop=True).groupby("abstract_source")}
    rows = []
    for v, d in te.reset_index(drop=True).groupby("venue"):
        y, ff = d.y.values.astype(bool), f[d.index]
        rows.append({"venue": v, "label": d.label.iloc[0], "test papers": len(d),
                     "labeled survey": int(ff.sum()), "% labeled survey": round(100 * ff.mean(), 2)})
    V = pd.DataFrame(rows).sort_values(["label", "test papers"], ascending=[False, False])
    V.to_csv(os.path.join(GT, "rq0_by_venue.csv"), index=False)
    json.dump(out, open(os.path.join(GT, "rq0_results.json"), "w"), indent=1, default=float)
    md = [f"Held-out test split: {len(te)} papers ({int(te.y.sum())} surveys) from {te.venue.nunique()} venues; "
          f"Semantic Scholar returned {100 * te.s2_found.mean():.1f}% of them.", "",
          "| Method | FPR (n) | Recall | Precision | At 8%: precision | At 8%: F1 | AP8 | McNemar p vs ours |", "|---|---|---|---|---|---|---|---|"]
    for name, r in out["methods"].items():
        mc = r.get("McNemar vs ours", {}).get("p")
        md.append(f"| {name} | {r['FPR %']:.2f}% ({r['research flagged']}) | {r['recall %']:.1f}% | {r['precision %']:.1f}% | {r['precision at 8% %']:.1f}% | "
                  f"{r['F1 at 8% %']:.1f}% | {r.get('AP8 %', float('nan')):.1f} | {'' if mc is None else f'{mc:.2g}'} |")
    md += ["", "Final model by abstract source:", ""] + [f"- {s}: {r['research']} research, FPR {r['FPR %']:.2f}%; {r['surveys']} surveys, recall {r['recall %']:.1f}%"
                                                         for s, r in out["by abstract source"].items()]
    sp = out["within Smyth & Cunningham's study population"]
    md += ["", f"Within Smyth & Cunningham's study population ({sp['surveys']} surveys, {sp['research']} research papers):"] + [
        f"- {k}: recall {v['recall %']:.1f}%, FPR {v['FPR %']:.2f}%" for k, v in sp.items() if isinstance(v, dict)]
    md += ["", "Surveys by outcome of the Smyth & Cunningham rule:"] + [f"- {k}: {v}" for k, v in out["surveys by outcome of the rule"].items()]
    if "venue held out (test papers)" in out:
        md += ["", "Venue held out in training (train.py holdout), test papers:"]
        for k, v in out["venue held out (test papers)"].items():
            md.append(f"- {k}: " + (f"recall {v['recall %']:.1f}%, FPR {v['FPR %']:.2f}%, F1 at 8% {v['F1 at 8% %']:.1f}%" if "recall %" in v
                                    else f"{v['ours right, other wrong']} vs {v['ours wrong, other right']}, p = {v['p']:.2g}"))
    if out.get("seed variation (test)"):
        md += ["", "Seed variation (test split; mean, SD and range over the seeds):"]
        for name, sv in out["seed variation (test)"].items():
            md.append(f"- {name}, seeds {sv['seeds']}: " + ", ".join(
                f"{k} {v['mean']:.2f} (sd {v['sd']:.2f}, {v['min']:.2f}-{v['max']:.2f})" for k, v in sv.items() if k != "seeds"))
    ps = out["sensitivity: Smyth & Cunningham rule as published (one phrase list)"]
    md += ["", f"Sensitivity, rule as published (one phrase list): recall {ps['recall %']:.1f}%, FPR {ps['FPR %']:.2f}%, F1 at 8% {ps['F1 at 8% %']:.1f}%"]
    md += ["", V.to_markdown(index=False)]
    open(os.path.join(GT, "rq0_results.md"), "w").write("\n".join(md)); print("\n".join(md[:14]))
    print("\n".join(md[14:20]))


def baselines():
    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("fetch", "all"): fetch_s2_test()
    if step in ("evaluate", "all"): evaluate_baselines()


# =====================================================================================================================
# Checks of the venue labels
# =====================================================================================================================

def label_checks():
    """Two checks of the venue ground truth on the held-out test papers (secondary).

    1. Title-test threshold: research venues are admitted when at most 0.3% of their titles contain a survey term. The test
       metrics are recomputed with stricter thresholds, dropping the research venues above each one.
    2. Shared titles: test papers whose normalized title also appears among the training papers, and the metrics without
       them.

    Input: as rq1_evaluation.py baselines (ground truth, split, model predictions)
    Output: hcr_study/venue_all/rq0_label_checks.md"""
    METHODS = {"SciGenre (ours)": os.path.join(HERE, "hcr_study", "models_final", "seed42", "preds.csv"),
               **{name: os.path.join(HERE, "hcr_study", "models_baselines", tag, "preds.csv") for name, tag in BASELINES.items()}}


    # survey terms of the title test that admitted research venues (Supplementary Section S1)
    TITLE_TERMS = re.compile(r"\b(?:survey|review|tutorial|overview|state[- ]of[- ]the[- ]art|primer|an introduction to|introduction to|"
                             r"taxonomy|retrospective|perspectives? on|lecture notes)\b", re.I)


    def title_test():
        """Per research venue: papers after non-papers are removed, and how many of their titles contain a survey term."""
        A = pd.read_csv(os.path.join(GT, "ground_truth.csv"), dtype={"OpenAlexId": str}, low_memory=False)
        A = A[(A.label == "research") & ~A.non_paper.astype(str).str.lower().eq("true")]
        return pd.DataFrame({"venue": A.venue, "label": A.label, "papers (non-papers removed)": 1,
                             "survey-term titles": A.title.fillna("").str.contains(TITLE_TERMS).astype(int)})


    def norm(t):
        return re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()


    def line(te, f):
        r = scores(te.y, f)
        return (f"recall {r['recall %']:.1f}% ({r['surveys found']} of {r['surveys']}), FPR {r['FPR %']:.2f}% "
                f"({r['research flagged']} of {r['research']}; 95% CI {r['FPR CI'][0]:.2f} to {r['FPR CI'][1]:.2f}), "
                f"F1 at 8% {r['F1 at 8% %']:.1f}%")


    def main():
        G = load_test_split()
        tr, te = G[G.split == "train"], G[G.split == "test"].copy()
        for name, path in METHODS.items():
            P = pd.read_csv(path, dtype={"OpenAlexId": str})
            te[name] = te.OpenAlexId.map(P[P.split == "test"].set_index("OpenAlexId").p) >= 0.5
        # survey-term title rate of each research venue, as in Table S2 (title test of 7 Oct 2026)
        T = title_test()
        rate = T[T.label == "research"].groupby("venue").agg(sum=("survey-term titles", "sum"),
                                                              size=("papers (non-papers removed)", "sum"))
        rate["mean"] = rate["sum"] / rate["size"]
        assert set(rate.index) == set(te.loc[te.label == "research", "venue"]), "venue names differ"
        out = ["# Checks of the venue ground truth on the test papers", "", "## 1. Title-test threshold",
               "Survey-term title rate of the research venues: " + "; ".join(
                   f"{v} {100 * r['mean']:.2f}% ({r['sum']:.0f} of {r['size']:.0f})" for v, r in rate.sort_values("mean").iterrows()), ""]
        for thr in (0.30, 0.20, 0.10):
            keep = set(rate.index[(100 * rate["mean"]).round(2) <= thr])
            dropped = sorted(set(rate.index) - keep)
            d = te[(te.label == "survey") | te.venue.isin(keep)]
            out.append(f"### Threshold {thr:.2f}%{' (as in the paper)' if thr == 0.30 else ''}: {len(keep)} research venues"
                       + (f", dropping {', '.join(dropped)}" if dropped else ""))
            out += [f"- {name}: {line(d, d[name].values)}" for name in METHODS] + [""]
        seen = set(tr.title.map(norm))
        te["shared"] = te.title.map(norm).isin(seen)
        s = te[te.shared]
        out += ["## 2. Test papers whose title also appears among the training papers",
                f"- {len(s)} of {len(te)} ({100 * len(s) / len(te):.1f}%): {int(s.y.sum())} surveys, {int((1 - s.y).sum())} "
                f"research papers",
                "- most frequent shared titles: " + "; ".join(f"\"{t}\" ({n})" for t, n in s.title.value_counts().head(6).items())]
        d = te[~te.shared]
        out += [f"- without them, {name}: {line(d, d[name].values)}" for name in METHODS]
        with open(os.path.join(GT, "rq0_label_checks.md"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        print("\n".join(out))


    main()


# =====================================================================================================================
# Replication of Smyth & Cunningham's published labels
# =====================================================================================================================

def sc_replication():
    """Reproduce Smyth & Cunningham's published review labels with their released code (review_paper_rule.py, via
    rq1_evaluation.py), and with our earlier implementations from the published description.

    Their figshare dataset (doi:10.6084/m9.figshare.32077821; its Computer Science rows are the input
    inputs/smyth_cunningham/sc_labels_cs.csv) gives, for every paper of their 17M data set, the
    Semantic Scholar corpus id and their label (paper_type: original / narrative / systematic / meta). It has no
    titles, abstracts or types, so a sample of their Computer Science papers is re-fetched from the Semantic Scholar
    API and relabelled by our rule. Variants are compared: the published reading (Table S1 on title and abstract, as
    substrings or whole words) and the as-run reconstruction (a narrower abstract list), each with each Semantic
    Scholar venue field. The variant that best reproduces their labels is the module's default.

    Semantic Scholar records change over time (they downloaded in the first half of 2025), so agreement is measured
    on records that still meet their inclusion criteria today (JournalArticle, abstract of at least five words, at
    least one reference); records that no longer do are reported separately.

    Usage: rq1_evaluation.py sc-replication fetch | evaluate      Output: hcr_study/sc_replication/"""
    load_dotenv(os.path.join(HERE, ".env"), override=True)
    KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    OUT = os.path.join(HERE, "hcr_study", "sc_replication"); os.makedirs(OUT, exist_ok=True); CACHE = os.path.join(OUT, "s2_records.json")
    LABELS = os.path.join(HERE, "inputs", "smyth_cunningham", "sc_labels_cs.csv")
    FIELDS = "corpusId,title,abstract,publicationTypes,venue,journal,publicationVenue,referenceCount,year"
    N_REGULAR, N_NARRATIVE = 40000, 6000          # all systematic and meta reviews are taken


    def sample():
        L = pd.read_csv(LABELS, dtype={"corpusid": str})
        rng = np.random.default_rng(2026); parts = []
        for t, d in L.groupby("paper_type"):
            n = {"original": N_REGULAR, "narrative": N_NARRATIVE}.get(t, len(d))
            parts.append(d.iloc[rng.choice(len(d), min(n, len(d)), replace=False)])
        S = pd.concat(parts); S.to_csv(os.path.join(OUT, "sample.csv"), index=False)
        print(L.paper_type.value_counts().to_string(), "\nsample:", S.paper_type.value_counts().to_dict(), flush=True)
        return S


    def fetch():
        S = sample()
        cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
        todo = [c for c in S.corpusid if c not in cache]; last = 0.0
        for i in range(0, len(todo), 500):
            chunk = todo[i:i + 500]
            for attempt in range(10):
                wait = 1.1 - (time.time() - last)
                if wait > 0: time.sleep(wait)
                last = time.time()
                r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch", params={"fields": FIELDS},
                                  json={"ids": [f"CorpusId:{c}" for c in chunk]}, headers={"x-api-key": KEY} if KEY else {}, timeout=120)
                if r.status_code == 200: break
                time.sleep(min(60, 3 * 2 ** attempt))
            else:
                raise SystemExit(f"failed at {i}: {r.status_code}")
            for c, x in zip(chunk, r.json()):
                cache[c] = x
            if (i // 500) % 10 == 0:
                json.dump(cache, open(CACHE, "w")); print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
        json.dump(cache, open(CACHE, "w")); print("FETCH DONE", flush=True)


    def evaluate():
        S = pd.read_csv(os.path.join(OUT, "sample.csv"), dtype={"corpusid": str})
        cache = json.load(open(CACHE))
        rec = [cache.get(c) or {} for c in S.corpusid]
        S["found"] = [bool(x) for x in rec]
        S["title"] = [x.get("title") for x in rec]; S["abstract"] = [x.get("abstract") for x in rec]
        S["types"] = [x.get("publicationTypes") or [] for x in rec]
        S["v_venue"] = [x.get("venue") or "" for x in rec]
        S["v_journal"] = [(x.get("journal") or {}).get("name") or "" for x in rec]
        S["v_pubvenue"] = [(x.get("publicationVenue") or {}).get("name") or "" for x in rec]
        S["v_any"] = S.v_venue + " | " + S.v_journal + " | " + S.v_pubvenue      # = s2_venue
        S["eligible"] = [in_study_population(t, a, x.get("referenceCount"), x.get("year")) for t, a, x in zip(S.types, S.abstract, rec)]
        S["theirs"] = S.paper_type.replace({"original": "regular"})
        S["their_review"] = S.theirs != "regular"
        E = S[S.found & S.eligible].copy()
        print(f"sample {len(S)}, found {S.found.sum()}, still eligible {len(E)}  "
              f"(their reviews {E.their_review.sum()}, regular {(~E.their_review).sum()})")
        print(f"  their reviews typed Review/MetaAnalysis today: {E[E.their_review].types.map(lambda t: bool(set(t) & REVIEW_TYPES)).mean():.1%}; "
              f"their regular papers typed so: {E[~E.their_review].types.map(lambda t: bool(set(t) & REVIEW_TYPES)).mean():.1%}")
        rows = []
        for mode, match in (("as_run", "substring"), ("published", "substring"), ("published", "word")):
            rule = Rule(mode, match)
            for vcol in ("v_venue", "v_journal", "v_pubvenue", "v_any"):
                ours = rule.classify(E, types="types", venue=vcol)
                rv = ours != "regular"
                tp = int((rv & E.their_review).sum()); fp = int((rv & ~E.their_review).sum()); fn = int((~rv & E.their_review).sum())
                rows.append({"mode": mode, "match": match, "venue field": vcol[2:], "agreement %": round(100 * (ours == E.theirs).mean(), 3),
                             "review agreement %": round(100 * (rv == E.their_review).mean(), 3),
                             "ours not theirs": fp, "theirs not ours": fn, "both review": tp,
                             "subtype agreement on shared reviews %": round(100 * (ours[rv & E.their_review] == E.theirs[rv & E.their_review]).mean(), 2)})
                E[f"ours_{mode}_{match}_{vcol[2:]}"] = ours
        R = pd.DataFrame(rows).sort_values("agreement %", ascending=False)
        print(R.to_markdown(index=False))
        best = R.iloc[0]; col = f"ours_{best['mode']}_{best['match']}_{best['venue field']}"
        print(f"\nbest variant: {best['mode']}, {best['match']} matching, venue field {best['venue field']}")
        print(pd.crosstab(E.theirs, E[col], rownames=["theirs"], colnames=["ours"], margins=True).to_markdown())
        dis = E[E[col] != E.theirs]
        dis[["corpusid", "theirs", col, "types", "title", "v_venue", "v_journal", "abstract"]].to_csv(os.path.join(OUT, "disagreements.csv"), index=False)
        for _, r in dis.head(25).iterrows():
            print(f"  theirs {r.theirs:10s} ours {r[col]:10s} {r.types} | {str(r.title)[:80]} | {r.v_venue[:40]}")
        R.to_csv(os.path.join(OUT, "variants.csv"), index=False)
        # the authors' released code (sc_code()) on the same records: does it reproduce their labels?
        code = pd.Series([sc_code().classify({"title": x.get("title"), "abstract": x.get("abstract"), "venue": x.get("venue"),
                                           "journal": x.get("journal"), "publicationtypes": x.get("publicationTypes"),
                                           "referencecount": x.get("referenceCount")})["paper_type"]
                          for x, keep in zip(rec, S.found & S.eligible) if keep], index=E.index).replace({"original": "regular"})
        released = {"agreement %": round(100 * (code == E.theirs).mean(), 3),
                    "review agreement %": round(100 * ((code != "regular") == E.their_review).mean(), 3)}
        print(f"\nauthors' released code: agreement {released['agreement %']}%, review vs regular {released['review agreement %']}%")
        json.dump({"n_sample": int(len(S)), "n_found": int(S.found.sum()), "n_eligible": int(len(E)), "best": best.to_dict(),
                   "released code": released, "variants": rows}, open(os.path.join(OUT, "replication.json"), "w"), indent=1, default=str)


    {"fetch": fetch, "evaluate": evaluate}[sys.argv[1]]()


STEPS = {"baselines": baselines, "label-checks": label_checks, "sc-replication": sc_replication}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv.pop(1)]()
