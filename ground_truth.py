"""Stage 1: the ground truth (every paper of 18 survey venues and 14 research venues; the venue is the label) and
the abstracts that OpenAlex lacks, for the ground truth and for every downloaded profile.

    python ground_truth.py venues [openalex|s2|scopus|build|all]          papers of the venues and their abstracts
    python ground_truth.py s2-abstracts                                   missing abstracts of the profiles: Semantic Scholar
    python ground_truth.py publisher-abstracts [doi|springer|title|all]   then Scopus and Springer Nature
"""
import glob
import json
import os
import re
import sys
import time

import pandas as pd
import requests
from dotenv import load_dotenv

import common
from common import clean_text, is_non_paper, openalex_get, prepare_frame

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# Ground truth from survey venues and research venues
# =====================================================================================================================

def venues():
    """The ground truth: every paper of venues whose officially stated scope is surveys or original research; the venue
    is the label, with no hand checking.
    Steps: 1. OpenAlex, all works per source (cursor paging, no e-mail sent);
           2. missing abstracts from Semantic Scholar by DOI (batch API, 500 per request, SEMANTIC_SCHOLAR_API_KEY), then
              from Scopus by DOI for what is still missing (Scopus Search, view COMPLETE, ELSEVIER_API_KEY; keys from
              .env, never printed);
           3. one table with the venue label, document type, abstract source and whether the paper is usable
              (not a non-paper, abstract from some source).
    Usage: ground_truth.py venues [openalex|s2|scopus|build|all]   Output: hcr_study/venue_all/ (ground_truth.csv, venues.csv)"""
    load_dotenv(os.path.join(HERE, ".env"), override=True)
    common.OPENALEX_MAILTO = ""
    OUT = os.path.join(HERE, "hcr_study", "venue_all"); os.makedirs(OUT, exist_ok=True)
    # Venue rule (OpenAlex source IDs; the official statements are quoted in the paper's Table S1):
    #   survey venues: the official policy states that the venue publishes only surveys, reviews or tutorials;
    #   research venues: the official scope asks only for original research, lists no survey, review or overview article
    #   type, and at most 0.3% of titles carry a survey term.
    # Excluded because their official policies admit surveys or other non-research papers: IEEE TSP, TIP and TIFS
    # (Overview Articles), PACMMOD (experimental surveys), IEEE TMC, NeurIPS and ICML (position papers), PACMPL (ICFP
    # pearls and experience reports), IEEE TAC. Excluded on the title test or editorial policy: JMLR, TACL, IEEE Trans.
    # Robotics, Automatica, IEEE Trans. Information Theory, ICSE and WIREs DMKD. ACM Computing Surveys counts from 2000:
    # before then it also printed position statements and readers' letters (YEAR_FROM).
    SURVEY = {"S23688054": "IEEE Communications Surveys & Tutorials", "S157921468": "ACM Computing Surveys", "S73121659": "Computer Science Review",
              "S4210191328": "Annual Review of Control, Robotics, and Autonomous Systems",
              # Foundations and Trends series
              "S4210188176": "F&T Machine Learning", "S23590115": "F&T Networking", "S197106261": "F&T Information Retrieval",
              "S123473935": "F&T Signal Processing", "S70831867": "F&T Communications and Information Theory",
              "S139685423": "F&T Databases", "S110848248": "F&T Theoretical Computer Science",
              "S189699570": "F&T Human-Computer Interaction", "S4210223006": "F&T Privacy and Security",
              "S83210289": "F&T Computer Graphics and Vision", "S43319301": "F&T Robotics",
              "S4210233022": "F&T Programming Languages", "S4210187074": "F&T Optimization",
              "S138962964": "F&T Electronic Design Automation"}
    RESEARCH = {"S147316732": "IEEE Communications Letters", "S2500830676": "IEEE Wireless Communications Letters",
                "S120629676": "IEEE Signal Processing Letters", "S147953040": "Information Processing Letters",
                "S63459445": "IEEE Transactions on Wireless Communications", "S196647941": "IEEE Transactions on Communications",
                "S62238642": "IEEE/ACM Transactions on Networking", "S153560523": "SIAM Journal on Computing",
                "S193109227": "ACM Transactions on Computer Systems",
                "S4210193547": "Proceedings of the ACM on Measurement and Analysis of Computing Systems", "S185367456": "ACM Transactions on Graphics",
                "S118992489": "Journal of the ACM", "S133795288": "IEEE Transactions on Dependable and Secure Computing",
                "S4210174050": "ACM Transactions on Privacy and Security", "S2642811": "ACM Transactions on Privacy and Security"}   # S2642811: TISSEC, its predecessor
    VENUES = {**{k: (v, "survey") for k, v in SURVEY.items()}, **{k: (v, "research") for k, v in RESEARCH.items()}}
    YEAR_FROM = {"S157921468": 2000}     # ACM Computing Surveys from 2000
    FILES = lambda: [os.path.join(OUT, f"openalex_{sid}.csv") for sid in VENUES if os.path.exists(os.path.join(OUT, f"openalex_{sid}.csv"))]
    SELECT = "id,doi,title,publication_year,abstract_inverted_index,type,referenced_works_count,cited_by_count"
    inv = lambda ii: " ".join(w for _, w in sorted((p, w) for w, ps in ii.items() for p in ps)) if ii else ""
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()

    def fetch_openalex():
        for sid, (name, cls) in VENUES.items():
            path = os.path.join(OUT, f"openalex_{sid}.csv")
            if os.path.exists(path): continue
            rows, cursor = [], "*"
            while cursor:
                d = openalex_get("works", {"filter": f"primary_location.source.id:{sid}", "select": SELECT, "per-page": 200, "cursor": cursor})
                res = d.get("results", [])
                for w in res:
                    rows.append({"OpenAlexId": w["id"].rsplit("/", 1)[-1], "doi": (w.get("doi") or "").replace("https://doi.org/", "").lower(),
                                 "title": clean_text(w.get("title") or ""), "abstract": clean_text(inv(w.get("abstract_inverted_index"))),
                                 "year": w.get("publication_year"), "type": w.get("type"), "ReferenceCount": w.get("referenced_works_count") or 0,
                                 "citationCount": w.get("cited_by_count") or 0, "venue": name, "class": cls, "source_id": sid})
                cursor = d.get("meta", {}).get("next_cursor") if res else None
                time.sleep(0.1)
            pd.DataFrame(rows).to_csv(path, index=False)
            print(f"{cls:8s} {name[:60]:60s} {len(rows):6d} works", flush=True)

    S2K = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip(); S2CACHE = os.path.join(OUT, "s2_abstracts.json")
    def fetch_s2():
        A = pd.concat([pd.read_csv(f, dtype=str) for f in FILES()], ignore_index=True)
        has = A.abstract.fillna("").map(lambda a: len(a.split()) >= 10)
        cache = json.load(open(S2CACHE)) if os.path.exists(S2CACHE) else {}
        todo = sorted(set(A.doi[~has & A.doi.notna() & (A.doi != "")]) - set(cache))
        print(f"Semantic Scholar: {len(todo)} DOIs without an OpenAlex abstract", flush=True)
        last = 0.0
        for i in range(0, len(todo), 500):
            chunk = todo[i:i + 500]
            for attempt in range(10):
                wait = 1.1 - (time.time() - last)
                if wait > 0: time.sleep(wait)
                last = time.time()
                try:
                    r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch", params={"fields": "abstract"},
                                      json={"ids": ["DOI:" + d for d in chunk]}, headers={"x-api-key": S2K} if S2K else {}, timeout=90)
                except requests.RequestException:
                    time.sleep(5 * (attempt + 1)); continue
                if r.status_code == 200: break
                time.sleep(min(60, 3 * 2 ** attempt))
            else:
                print(f"  failed at {i}", flush=True); continue
            for d, x in zip(chunk, r.json()):
                cache[d] = (x.get("abstract") or "") if isinstance(x, dict) else ""
            if (i // 500) % 10 == 0: json.dump(cache, open(S2CACHE, "w")); print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
        json.dump(cache, open(S2CACHE, "w"))

    EK = os.getenv("ELSEVIER_API_KEY", "").strip(); SCACHE = os.path.join(OUT, "scopus_abstracts.json")
    def fetch_scopus():
        A = pd.concat([pd.read_csv(f, dtype=str) for f in FILES()], ignore_index=True)
        s2 = json.load(open(S2CACHE)) if os.path.exists(S2CACHE) else {}
        has = A.abstract.fillna("").map(lambda a: len(a.split()) >= 10) | A.doi.fillna("").map(lambda d: len(clean_text(s2.get(d, "")).split()) >= 10)
        cache = json.load(open(SCACHE)) if os.path.exists(SCACHE) else {}
        todo = sorted(set(A.doi[~has & A.doi.notna() & (A.doi != "")]) - set(cache))
        print(f"Scopus: {len(todo)} DOIs without an OpenAlex abstract", flush=True)
        last = 0.0
        for i in range(0, len(todo), 25):
            chunk = todo[i:i + 25]
            for attempt in range(8):
                wait = 0.5 - (time.time() - last)
                if wait > 0: time.sleep(wait)
                last = time.time()
                try:
                    r = requests.get("https://api.elsevier.com/content/search/scopus", params={"query": " OR ".join(f'DOI("{d}")' for d in chunk),
                                     "view": "COMPLETE", "count": 25}, headers={"X-ELS-APIKey": EK, "Accept": "application/json"}, timeout=90)
                except requests.RequestException:
                    time.sleep(5 * (attempt + 1)); continue
                if r.status_code == 200: break
                if r.status_code in (429, 500, 502, 503, 504): time.sleep(min(120, 5 * 2 ** attempt)); continue
                print(f"  status {r.status_code}: {r.text[:120]}", flush=True); r = None; break
            if r is None or r.status_code != 200: continue
            left = r.headers.get("X-RateLimit-Remaining")
            if left is not None and int(left) < 300: print(f"Scopus quota nearly used ({left} left); stopping", flush=True); break
            found = {str(e.get("prism:doi", "")).lower(): e.get("dc:description") or "" for e in r.json().get("search-results", {}).get("entry", []) if "error" not in e}
            for d in chunk: cache[d] = found.get(d, "")
            if (i // 25) % 40 == 0: json.dump(cache, open(SCACHE, "w")); print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
        json.dump(cache, open(SCACHE, "w"))

    def build():
        """One ground truth: every paper of every venue, labeled by its venue (survey or research venue); no hand
    checking. Non-papers are removed by the pipeline's document-type rule (OpenAlex type or title: editorials,
    errata, front matter, ...). Papers without an abstract from any source are listed but not used.
    Outputs: ground_truth.csv (papers) and venues.csv (appendix table: papers and abstract source shares)."""
        A = pd.concat([pd.read_csv(f, dtype={"OpenAlexId": str, "doi": str}) for f in FILES()], ignore_index=True)
        A = A.drop(columns=[c for c in ("set",) if c in A]).drop_duplicates("OpenAlexId")
        for sid, y0 in YEAR_FROM.items():
            A = A[~((A.source_id == sid) & (pd.to_numeric(A.year, errors="coerce") < y0))]
        A = A.rename(columns={"class": "label"})
        sc = json.load(open(SCACHE)) if os.path.exists(SCACHE) else {}
        s2 = json.load(open(S2CACHE)) if os.path.exists(S2CACHE) else {}
        A["abstract_source"] = "OpenAlex"
        for name, cache in (("Semantic Scholar", s2), ("Scopus", sc)):
            fill = A.abstract.fillna("").map(lambda a: len(a.split()) < 10) & A.doi.fillna("").map(lambda d: len(clean_text(cache.get(d, "")).split()) >= 10)
            A.loc[fill, "abstract"] = A.loc[fill, "doi"].map(lambda d: clean_text(cache[d])); A.loc[fill, "abstract_source"] = name
        A.loc[A.abstract.fillna("").map(lambda a: len(a.split()) < 10), "abstract_source"] = "none"
        A["non_paper"] = is_non_paper(prepare_frame(A))
        A["usable"] = ~A.non_paper & (A.abstract_source != "none")
        A.to_csv(os.path.join(OUT, "ground_truth.csv"), index=False)
        U = A[A.usable]
        v = A.groupby(["label", "venue"]).agg(works=("OpenAlexId", "size"), non_papers=("non_paper", "sum"),
                                              no_abstract=("abstract_source", lambda x: (x == "none").sum()), first_year=("year", "min"),
                                              last_year=("year", "max")).reset_index()
        u = U.groupby("venue").agg(papers=("OpenAlexId", "size"), openalex=("abstract_source", lambda x: 100 * (x == "OpenAlex").mean()),
                                   semantic_scholar=("abstract_source", lambda x: 100 * (x == "Semantic Scholar").mean()),
                                   scopus=("abstract_source", lambda x: 100 * (x == "Scopus").mean())).reset_index()
        v = v.merge(u, on="venue", how="left").round(1).sort_values(["label", "papers"], ascending=[False, False])
        v.to_csv(os.path.join(OUT, "venues.csv"), index=False); print(v.to_markdown(index=False))
        print(v.groupby("label")[["works", "non_papers", "no_abstract", "papers"]].sum().to_markdown())

    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    if step in ("openalex", "all"): fetch_openalex()
    if step in ("s2", "all"): fetch_s2()
    if step in ("scopus", "all"): fetch_scopus()
    if step in ("build", "all"): build()


# =====================================================================================================================
# Missing abstracts: Semantic Scholar
# =====================================================================================================================

def s2_abstracts():
    """Missing abstracts for the HCR study papers (title + abstract is the only model input, so papers without an
    abstract cannot be classified). Source: Semantic Scholar batch API by DOI, key SEMANTIC_SCHOLAR_API_KEY from
    .env (header x-api-key), at most one request per second. Covers the downloaded profiles of the listed,
    matched, reference and boundary researchers (other folders under hcr_study/ with PROFILE_DIRS, comma-separated).
    Incremental: run it again after every profile download, before classification.
    Output: hcr_study/abstracts/s2_abstracts.json {doi: abstract or null}; prints coverage before and after."""
    load_dotenv(os.path.join(HERE, ".env"), override=True)
    KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip(); HEADERS = {"x-api-key": KEY} if KEY else {}
    URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
    OUT = os.path.join(HERE, "hcr_study", "abstracts"); os.makedirs(OUT, exist_ok=True); CACHE = os.path.join(OUT, "s2_abstracts.json")
    DIRS = ["profiles", "controls/profiles", "reference/profiles", "boundary/profiles", "boundary_artificialin/profiles"]
    if os.getenv("PROFILE_DIRS"): DIRS = os.environ["PROFILE_DIRS"].split(",")      # e.g. fellows2026/profiles
    has_ab = lambda a: len(clean_text(a).split()) >= 10
    frames = []
    for d in DIRS:
        for f in glob.glob(os.path.join(HERE, "hcr_study", d, "*.csv")):
            x = pd.read_csv(f, usecols=["id", "doi", "abstract"]); x["group"] = d.split("/")[0]; frames.append(x)
    P = pd.concat(frames, ignore_index=True)
    P["doi"] = P.doi.fillna("").str.replace("https://doi.org/", "", regex=False).str.lower()
    P["has"] = P.abstract.fillna("").map(has_ab)
    miss = P[~P.has]
    print(f"papers {len(P)}; without abstract {len(miss)} ({100*len(miss)/len(P):.1f}%); of these with DOI {int((miss.doi != '').sum())}")
    print(miss.groupby("group").size().to_dict())
    if "--count" in sys.argv: raise SystemExit
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    todo = sorted(set(miss.doi[miss.doi != ""]) - set(cache))
    last = 0.0
    for i in range(0, len(todo), 500):
        chunk = todo[i:i + 500]
        for attempt in range(12):
            wait = 1.1 - (time.time() - last)
            if wait > 0: time.sleep(wait)
            last = time.time()
            r = requests.post(URL, params={"fields": "abstract"}, json={"ids": ["DOI:" + d for d in chunk]}, headers=HEADERS, timeout=60)
            if r.status_code == 200: break
            time.sleep(min(60, 3 * 2 ** attempt))
        else: raise SystemExit(f"failed at {i}: {r.status_code} {r.text[:200]}")
        for d, res in zip(chunk, r.json()):
            cache[d] = (res or {}).get("abstract")
        json.dump(cache, open(CACHE + ".tmp", "w")); os.replace(CACHE + ".tmp", CACHE)      # atomic: readers never see a partial file
        if (i // 500) % 20 == 0: print(f"  {min(i + 500, len(todo))}/{len(todo)}", flush=True)
    got = miss.doi.map(lambda d: has_ab(cache.get(d) or "") if d else False)
    print(f"filled from Semantic Scholar: {int(got.sum())} of {len(miss)}; still without abstract: {len(miss) - int(got.sum())} "
          f"({100*(len(miss) - got.sum())/len(P):.1f}% of all papers)")
    print("still missing by group:", miss[~got.values].groupby("group").size().to_dict())


# =====================================================================================================================
# Missing abstracts: Scopus and Springer Nature
# =====================================================================================================================

def publisher_abstracts():
    """Abstracts for HCR-study papers that neither OpenAlex nor Semantic Scholar provides (title + abstract is the
    only model input). Keys from .env (never printed):
      1. Scopus Search API, view COMPLETE (ELSEVIER_API_KEY): 25 DOIs per request, at most 2 requests per second.
      2. Springer Nature Meta API (SPRINGER_META_API_KEY): Springer DOIs still missing, 25 per request, 1 per second.
      3. Scopus Search by exact title for papers without a DOI (25 titles per request; a match needs the same
         normalized title and publication year within one year).
    Only article-like papers are queried (document-type non-papers are not classified anyway).
    Output: hcr_study/abstracts/publisher_abstracts.json {key: {"abstract": str, "source": str}}, key = DOI or
    "title:<normalized title>|<year>". Usage: ground_truth.py publisher-abstracts [doi|springer|title|all]"""
    load_dotenv(os.path.join(HERE, ".env"), override=True)
    EK = os.getenv("ELSEVIER_API_KEY", "").strip(); SK = os.getenv("SPRINGER_META_API_KEY", "").strip()
    OUT = os.path.join(HERE, "hcr_study", "abstracts"); CACHE = os.path.join(OUT, "publisher_abstracts.json")
    S2 = json.load(open(os.path.join(OUT, "s2_abstracts.json")))
    DIRS = ["profiles", "controls/profiles", "reference/profiles", "boundary/profiles", "boundary_artificialin/profiles"]
    if os.getenv("PROFILE_DIRS"): DIRS = os.environ["PROFILE_DIRS"].split(",")      # e.g. fellows2026/profiles
    has_ab = lambda a: len(clean_text(a).split()) >= 10
    norm = lambda t: re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()

    def missing_papers():
        fr = [pd.read_csv(f, usecols=lambda c: c in ("title", "doi", "abstract", "type", "venue", "year")) for d in DIRS for f in glob.glob(os.path.join(HERE, "hcr_study", d, "*.csv"))]
        P = pd.concat(fr, ignore_index=True)
        P["doi"] = P.doi.fillna("").str.replace("https://doi.org/", "", regex=False).str.lower()
        P = P[~P.abstract.fillna("").map(has_ab) & ~P.doi.map(lambda d: has_ab(S2.get(d) or "") if d else False)]
        P = P[~is_non_paper(prepare_frame(P))]
        return P.drop_duplicates(subset=["doi", "title"])

    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    def save(): json.dump(cache, open(CACHE + ".tmp", "w")); os.replace(CACHE + ".tmp", CACHE)      # atomic
    _last = {"scopus": 0.0, "springer": 0.0}
    def get(kind, url, params, headers, min_gap):
        for attempt in range(8):
            wait = min_gap - (time.time() - _last[kind])
            if wait > 0: time.sleep(wait)
            _last[kind] = time.time()
            try:
                r = requests.get(url, params=params, headers=headers, timeout=90)
            except requests.RequestException:
                time.sleep(5 * (attempt + 1)); continue
            if r.status_code == 200: return r
            if r.status_code in (429, 500, 502, 503, 504): time.sleep(min(120, 5 * 2 ** attempt)); continue
            print(f"  {kind} status {r.status_code}: {r.text[:150]}", flush=True); return None
        return None

    def scopus(query):
        r = get("scopus", "https://api.elsevier.com/content/search/scopus", {"query": query, "view": "COMPLETE", "count": 25},
                {"X-ELS-APIKey": EK, "Accept": "application/json"}, 0.5)
        if r is None: return None
        left = r.headers.get("X-RateLimit-Remaining")
        if left is not None and int(left) < 200: raise SystemExit(f"Scopus weekly quota nearly used ({left} left); stopping")
        return [e for e in r.json().get("search-results", {}).get("entry", []) if "error" not in e]

    def by_doi(P):
        todo = sorted(set(P.doi[P.doi != ""]) - set(cache))
        print(f"Scopus by DOI: {len(todo)} DOIs", flush=True)
        for i in range(0, len(todo), 25):
            chunk = todo[i:i + 25]
            ents = scopus(" OR ".join(f'DOI("{d}")' for d in chunk))
            if ents is None: continue
            found = {str(e.get("prism:doi", "")).lower(): e.get("dc:description") or "" for e in ents}
            for d in chunk:
                cache[d] = {"abstract": found.get(d, ""), "source": "scopus" if has_ab(found.get(d, "")) else "none"}
            if (i // 25) % 40 == 0: save(); print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
        save()

    def springer(P):
        pref = ("10.1007/", "10.1023/", "10.1186/", "10.1038/", "10.1140/", "10.1057/", "10.1365/")
        todo = sorted(d for d in set(P.doi) if d.startswith(pref) and not has_ab((cache.get(d) or {}).get("abstract", "")) and (cache.get(d) or {}).get("springer") is None)
        print(f"Springer Meta API: {len(todo)} DOIs", flush=True)
        for i in range(0, len(todo), 25):
            chunk = todo[i:i + 25]
            r = get("springer", "https://api.springernature.com/meta/v2/json",
                    {"q": "(" + " OR ".join(f'doi:"{d}"' for d in chunk) + ")", "p": 25, "api_key": SK}, {}, 1.0)
            if r is None: continue
            found = {}
            for x in r.json().get("records", []):
                a = x.get("abstract") or ""
                if isinstance(a, dict): a = " ".join(str(v) for v in a.values())
                found[str(x.get("doi", "")).lower()] = a
            for d in chunk:
                a = found.get(d, "")
                prev = cache.get(d) or {"abstract": "", "source": "none"}
                cache[d] = {"abstract": a, "source": "springer"} if has_ab(a) else {**prev, "springer": "none"}
            if (i // 25) % 40 == 0: save(); print(f"  {i + len(chunk)}/{len(todo)}", flush=True)
        save()

    def by_title(P):
        T = P[(P.doi == "") & P.title.notna()].copy()
        T["key"] = ["title:" + norm(t) + "|" + str(int(y)) if y == y else "title:" + norm(t) + "|" for t, y in zip(T.title, T.year)]
        T = T[~T.key.isin(set(cache)) & (T.title.str.len() > 20)].drop_duplicates("key")
        print(f"Scopus by title: {len(T)} papers", flush=True)
        rows = list(T.itertuples())
        size = 25 if len(rows) > 3000 else 8      # retries (failed batches only) use short, cleaned queries
        for i in range(0, len(rows), size):
            chunk = rows[i:i + size]
            clean = (lambda t: re.sub(r"\s+", " ", re.sub(r"[^A-Za-z0-9 ]+", " ", str(t))).strip()) if size == 8 else (lambda t: re.sub(r'["\\{}()\[\]]', " ", str(t)).strip())
            q = " OR ".join('TITLE("' + clean(r.title) + '")' for r in chunk)
            ents = scopus(q)
            if ents is None: continue
            found = {}
            for e in ents:
                y = str(e.get("prism:coverDate", ""))[:4]
                found.setdefault(norm(e.get("dc:title", "")), []).append((int(y) if y.isdigit() else None, e.get("dc:description") or ""))
            for r in chunk:
                a = ""
                for y, ab in found.get(norm(r.title), []):
                    if r.year == r.year and y is not None and abs(y - int(r.year)) <= 1 and has_ab(ab): a = ab; break
                cache[r.key] = {"abstract": a, "source": "scopus-title" if has_ab(a) else "none"}
            if (i // size) % 40 == 0: save(); print(f"  {i + len(chunk)}/{len(rows)}", flush=True)
        save()

    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    P = missing_papers()
    print(f"article-like papers without an abstract: {len(P)} ({int((P.doi != '').sum())} with DOI)", flush=True)
    if step in ("doi", "all"): by_doi(P)
    if step in ("springer", "all"): springer(P)
    if step in ("title", "all"): by_title(P)
    got = sum(1 for v in cache.values() if has_ab(v.get("abstract", "")))
    src = pd.Series([v.get("source") for v in cache.values() if has_ab(v.get("abstract", ""))]).value_counts().to_dict()
    print(f"abstracts found: {got} of {len(cache)} queried; by source {src}")


STEPS = {"venues": venues, "s2-abstracts": s2_abstracts, "publisher-abstracts": publisher_abstracts}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv.pop(1)]()
