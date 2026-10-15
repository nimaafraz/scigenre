"""Stage 4: the 2025 HCR list in Computer Science, resolved to OpenAlex authors and Stanford-Elsevier records, the
publication profiles, the HCR lists of 2014-2025, and the classification of every profile by SciGenre.

    python profiles.py fetch                                            OpenAlex authors and profiles of the 2025 list
    python profiles.py els-screen                                       derived columns of the Stanford-Elsevier table
    python profiles.py els-match                                        Stanford-Elsevier record of each HCR-listed researcher
    python profiles.py history build|resolve|merge|els|fetch|classify   HCR lists 2014-2025
    python profiles.py classify                                         classify the HCR-listed researchers' profiles
"""
import glob
import hashlib
import json
import os
import re
import sys
import unicodedata

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

import rq1_evaluation
from common import (
    calculate_indices, clean_text, is_non_paper, merge_versions, model_text, openalex_get, openalex_get_all,
    prepare_frame, short_id, WORK_FIELDS, work_to_row,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# SciGenre applied to publication profiles
# =====================================================================================================================
#
# The final SciGenre classifier (train.py final, installed by train.py install) applied to author profiles, for RQ2-RQ4.
# Input: title and abstract only (common.model_text: publisher boilerplate removed; never the venue).
# Abstracts: OpenAlex, else Semantic Scholar by DOI (ground_truth.py s2-abstracts), else Scopus or Springer Nature by DOI or
# exact title (ground_truth.py publisher-abstracts). Papers still without an abstract
# (fewer than 10 words) are not classified and stay in both versions of every indicator.
# Categories: non-paper (OpenAlex type or title: books, editorials, errata, ...), no abstract, survey
# (P(survey) >= threshold), research.
# Scoring on another machine with a GPU: SE_COLLECT=1 records every text the analyses
# classify in hcr_study/scores/texts.jsonl instead of running the model (the outputs of such a run are placeholders and
# are deleted); train.py score scores them with the installed model; proba() then reads
# hcr_study/scores/scores.csv when its model fingerprint matches the installed model, and scores any text missing
# from it here (reported).

MODEL_DIR = os.path.join(HERE, "model")
CFG = json.load(open(os.path.join(MODEL_DIR, "scigenre.json")))
THRESHOLD, MIN_WORDS, MAX_LEN = CFG["threshold"], CFG["min_abstract_words"], CFG["max_length"]
TEST_RECALL, TEST_FPR = CFG["test_recall"] / 100, CFG["test_fpr"] / 100     # held-out test split, for the error correction
_S2 = os.path.join(HERE, "hcr_study", "abstracts", "s2_abstracts.json")
S2_ABSTRACTS = json.load(open(_S2)) if os.path.exists(_S2) else {}
_PUB = os.path.join(HERE, "hcr_study", "abstracts", "publisher_abstracts.json")
PUB_ABSTRACTS = json.load(open(_PUB)) if os.path.exists(_PUB) else {}
_norm = lambda t: re.sub(r"[^a-z0-9]+", " ", str(t).lower()).strip()
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
_M = {}
def _model():
    """Tokenizer and model, loaded on first use (not at all when every score comes from the score cache)."""
    if not _M:
        _M["tok"] = AutoTokenizer.from_pretrained(MODEL_DIR)
        _M["model"] = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, dtype=torch.float32).to(DEVICE).eval()
    return _M["tok"], _M["model"]

SCORES = os.environ.get("SE_SCORES_DIR") or os.path.join(HERE, "hcr_study", "scores")   # another cache for side analyses
COLLECT = os.environ.get("SE_COLLECT") == "1"
text_hash = lambda t: hashlib.sha1(t.encode("utf-8")).hexdigest()

def model_fingerprint(model_dir=MODEL_DIR) -> str:
    """SHA-256 of the model weights: ties a score cache to one trained model."""
    h = hashlib.sha256()
    with open(os.path.join(model_dir, "model.safetensors"), "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""): h.update(block)
    return h.hexdigest()

_CACHE = {}
def _cache():
    if "scores" not in _CACHE:
        _CACHE["scores"], _CACHE["missed"] = {}, 0
        meta, path = os.path.join(SCORES, "scores.json"), os.path.join(SCORES, "scores.csv")
        if not COLLECT and os.path.exists(meta) and os.path.exists(path):
            if json.load(open(meta))["model_sha256"] == model_fingerprint():
                c = pd.read_csv(path, dtype={"h": str})
                _CACHE["scores"] = dict(zip(c.h, c.p))
            else:
                print("profiles.py: hcr_study/scores/scores.csv belongs to another model; scoring here", flush=True)
    return _CACHE["scores"]

_gt = pd.read_csv(os.path.join(HERE, "hcr_study", "venue_all", "ground_truth.csv"), usecols=["OpenAlexId", "title", "usable"], dtype={"OpenAlexId": str})
_split = pd.read_csv(os.path.join(HERE, "hcr_study", "venue_all", "split.csv"), dtype={"OpenAlexId": str})
_train = _gt[_gt.usable].merge(_split[_split.split == "train"], on="OpenAlexId").rename(columns={"title": "Title"})
title_key = lambda t: " ".join(str(t).lower().split())
TRAIN_TITLES = set(_train.Title.map(title_key))      # profile papers that were training papers


def has_abstract(a) -> bool:
    return len(clean_text(a).split()) >= MIN_WORDS


@torch.no_grad()
def score_texts(texts, bs=64) -> np.ndarray:
    """P(survey) of model texts with the installed model on this computer."""
    tok, model = _model()
    order = np.argsort([len(t) for t in texts])          # length-sorted batches: fewer padded tokens
    out = np.empty(len(texts))
    for i in range(0, len(texts), bs):
        idx = order[i:i + bs]
        enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, pad_to_multiple_of=64,
                  return_tensors="pt").to(DEVICE)
        out[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, CFG["survey_label_index"]].cpu().numpy()
    return out


def proba(titles, abstracts, bs=64) -> np.ndarray:
    texts = [model_text(t, a) for t, a in zip(titles, abstracts)]
    if COLLECT:                                          # record the texts for train.py score; placeholder scores
        os.makedirs(SCORES, exist_ok=True)
        with open(os.path.join(SCORES, "texts.jsonl"), "a") as f:
            for t in texts: f.write(json.dumps({"h": text_hash(t), "t": t}) + "\n")
        return np.zeros(len(texts))
    cache = _cache()
    keys = [text_hash(t) for t in texts]
    out = np.array([cache.get(k, np.nan) for k in keys], dtype=float)
    miss = np.isnan(out)
    if miss.any():
        if cache:
            _CACHE["missed"] += int(miss.sum())
            print(f"profiles.py: {int(miss.sum())} texts not in the score cache, scored here (total {_CACHE['missed']})", flush=True)
        out[miss] = score_texts([t for t, m in zip(texts, miss) if m], bs)
    return out


def score_profile(df: pd.DataFrame) -> pd.DataFrame:
    """Per-paper result for one profile (columns title, abstract, type, venue, doi)."""
    doi = df.get("doi", pd.Series("", index=df.index)).fillna("").str.replace("https://doi.org/", "", regex=False).str.lower()
    own = df.get("abstract", pd.Series("", index=df.index)).fillna("")
    s2 = doi.map(lambda d: S2_ABSTRACTS.get(d) or "")
    years = df.get("year", pd.Series(np.nan, index=df.index))
    pub_rec = [PUB_ABSTRACTS.get(d) if d else PUB_ABSTRACTS.get("title:" + _norm(t) + "|" + (str(int(y)) if y == y else ""))
               for d, t, y in zip(doi, df["title"], years)]
    pub = pd.Series([(r or {}).get("abstract") or "" for r in pub_rec], index=df.index)
    pub_src = pd.Series([(r or {}).get("source") or "none" for r in pub_rec], index=df.index)
    use_own, use_s2, use_pub = own.map(has_abstract), s2.map(has_abstract), pub.map(has_abstract)
    abstract = own.where(use_own, s2.where(use_s2, pub))
    source = np.where(use_own, "openalex", np.where(use_s2, "semantic scholar", np.where(use_pub, pub_src, "none")))
    frame = prepare_frame(df.assign(abstract=abstract))
    ok = source != "none"
    score = np.full(len(df), np.nan)
    if ok.any():
        score[ok] = proba(frame.Title[ok].tolist(), frame.Abstract[ok].tolist())
    non_paper = is_non_paper(frame)
    category = np.where(non_paper, "non-paper", np.where(~ok, "no abstract", np.where(score >= THRESHOLD, "survey", "research")))
    return pd.DataFrame({"Category": category, "SurveyScore": np.round(score, 4), "Abstract source": source,
                         "In training data": df["title"].map(title_key).isin(TRAIN_TITLES).values}, index=df.index)


# =====================================================================================================================
# The 2025 HCR list in Computer Science: OpenAlex authors and profiles
# =====================================================================================================================
#
# HCR study, step 1: resolve the researchers on the Clarivate HCR 2025 Computer Science
# list (inputs/hcr_lists/2025_HCR.xlsx) to OpenAlex and download their profiles.
#
# Uses the OpenAlex client (common.py) and profile format (common.py),
# plus each work's OpenAlex citation-normalized percentile so that top-1% papers
# (an approximation of Clarivate's Highly Cited Papers) can be counted.
#
#     python profiles.py fetch
#
# Output:
#     hcr_study/resolution.csv           one row per HCR-listed researcher, chosen OpenAlex ID,
#                                        match evidence and status
#     hcr_study/profiles/<ID>.csv        publication profile per researcher

# ============================================================
# SETTINGS
# ============================================================

HCR_LISTS = os.path.join(HERE, "inputs", "hcr_lists")          # Clarivate's lists, <year>_HCR.xlsx
OUT_DIR = os.path.join(HERE, "hcr_study")
PROFILE_DIR = os.path.join(OUT_DIR, "profiles")

# The 2025 Computer Science rows in the order in which the paper's run read them (Clarivate's web listing). That
# order is the spreadsheet's alphabetical order except that the spreadsheet's rows 91-100 come after its rows
# 101-119. The order matters: comparison_groups.py controls matches the HCR-listed researchers in a seeded random order of the rows.
WEB_ORDER = list(range(90)) + list(range(100, 119)) + list(range(90, 100))

# Match exceptions set by hand: rule "openalex_id" (name -> OpenAlex author ID, used instead of the name search) and
# rule "els_reject" (name -> els.csv institution that is a namesake, profiles.py els-match). The file names people, so it is
# not released; without it every researcher is matched by the rules alone.
EXCEPTIONS = os.path.join(HERE, "inputs", "match_exceptions.csv")


def match_exceptions(rule):
    """{HCR name: value} for one rule of inputs/match_exceptions.csv (columns rule, name, value); {} without the file."""
    if not os.path.exists(EXCEPTIONS):
        return {}
    e = pd.read_csv(EXCEPTIONS, dtype=str).fillna("")
    e = e[e["rule"].str.strip() == rule]
    return dict(zip(e["name"].str.strip(), e["value"].str.strip()))


KNOWN_IDS = match_exceptions("openalex_id")


def hcr_2025_cs():
    """The 2025 Computer Science list: Full Name ('Last, First'), Category, Primary and Secondary Affiliations."""
    x = pd.read_excel(os.path.join(HCR_LISTS, "2025_HCR.xlsx"), dtype=str)
    cs = x[x["Category"] == "Computer Science"].fillna("").reset_index(drop=True)
    assert len(cs) == len(WEB_ORDER), f"expected {len(WEB_ORDER)} Computer Science rows, found {len(cs)}"
    cs = cs.iloc[WEB_ORDER].reset_index(drop=True)
    return pd.DataFrame({"Full Name": cs["Last Name"].str.strip() + ", " + cs["First Name"].str.strip(),
                         "Category": cs["Category"], "Primary Affiliation": cs["Primary Affiliation"],
                         "Secondary Affiliations": cs["Secondary Affiliations"]})

CS_FIELDS = {"Computer Science", "Engineering", "Mathematics", "Decision Sciences", "Physics and Astronomy"}

STOPWORDS = {
    "university", "of", "the", "and", "technology", "science", "institute", "school",
    "college", "china", "mainland", "united", "states", "kingdom", "department",
    "engineering", "sciences", "national", "research", "de", "for", "hong", "kong",
    "sar", "south", "korea", "japan", "canada", "australia", "singapore", "germany",
}

AUTHOR_FIELDS = "id,display_name,display_name_alternatives,works_count,cited_by_count,last_known_institutions,affiliations,topics"
EXTRA_WORK_FIELDS = ",citation_normalized_percentile"


# ============================================================
# MATCHING
# ============================================================

def ascii_lower(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def tokens(s):
    return {w for w in re.findall(r"[a-z]+", ascii_lower(s)) if w not in STOPWORDS and len(w) > 2}


def inst_key(s):
    """Full institution name without the trailing country, letters only ('universityofhongkong')."""
    s = ascii_lower(s)
    s = re.sub(r",[^,]*$", "", s) if "," in s else s
    s = re.sub(r"^the ", "", s)
    return re.sub(r"[^a-z]", "", s)


def institution_overlap(row, author):
    """Distinctive shared words, plus 2 for a full-name match (catches names made only of
    generic words, e.g. 'University of Hong Kong')."""
    affil = tokens(row["Primary Affiliation"]) | tokens(row.get("Secondary Affiliations", ""))
    names = institution_names(author)
    overlap = len(affil & set().union(*[tokens(n) for n in names] or [set()]))
    hcr_keys = [inst_key(a) for a in (row["Primary Affiliation"], row.get("Secondary Affiliations", "")) if a]
    for k in hcr_keys:
        for n in names:
            nk = inst_key(n)
            if k and nk and (k == nk or (len(nk) > 12 and nk in k) or (len(k) > 12 and k in nk)):
                return overlap + 2
    return overlap


def hcr_query(name):
    last, first = name.split(",", 1)
    first = re.sub(r"\(.*?\)", "", first).strip()
    return f"{first} {last.strip()}", re.sub(r"[^a-z]", "", ascii_lower(last))


def name_ok(hcr_last, hcr_given, author):
    """
    Some recorded name variant must contain the surname as a word (or as the joined
    words of a compound surname) and, apart from it, exactly the HCR given names,
    allowing one trailing middle initial ("Jane Q. Doe" for "Doe, Jane").
    """
    given = re.sub(r"[^a-z]", "", ascii_lower(hcr_given))
    names = [author["display_name"]] + (author.get("display_name_alternatives") or [])
    for n in names:
        parts = re.findall(r"[a-z]+", ascii_lower(n))
        for i in range(len(parts)):
            for j in range(i + 1, min(i + 3, len(parts)) + 1):
                if "".join(parts[i:j]) != hcr_last:
                    continue
                rest = "".join(parts[:i] + parts[j:])
                if rest == given or (rest.startswith(given) and len(rest) - len(given) <= 1):
                    return True
    return False


def institution_names(author):
    names = [i.get("display_name", "") for i in (author.get("last_known_institutions") or [])]
    names += [(a.get("institution") or {}).get("display_name", "") for a in (author.get("affiliations") or [])]
    return names


def cs_share(author):
    topics = author.get("topics") or []
    total = sum(t.get("count", 0) for t in topics)
    cs = sum(t.get("count", 0) for t in topics if (t.get("field") or {}).get("display_name") in CS_FIELDS)
    return cs / total if total else 0.0


def resolve_author(row):
    name = row["Full Name"]
    if name in KNOWN_IDS:
        a = openalex_get(f"authors/{KNOWN_IDS[name]}", {"select": AUTHOR_FIELDS})
        return a, "known ID", 0, None

    query, last = hcr_query(name)
    first = re.sub(r"\(.*?\)", "", name.split(",", 1)[1])
    res = openalex_get("authors", {"search": query, "per-page": 50, "select": AUTHOR_FIELDS}).get("results", [])

    scored = []
    for a in res:
        if not name_ok(last, first, a):
            continue
        inst_overlap = institution_overlap(row, a)
        score = 3 * min(inst_overlap, 3) + 4 * cs_share(a) + np.log10(max(a.get("works_count", 1), 1))
        scored.append((score, inst_overlap, a))
    if not scored:
        return None, "not found", 0, None

    scored.sort(key=lambda s: s[0], reverse=True)
    best_score, best_inst, best = scored[0]
    runner = scored[1] if len(scored) > 1 else None

    if best_inst == 0:
        status = "weak (no institution match)"
    elif cs_share(best) < 0.5:
        status = "weak (low CS share)"
    elif runner and runner[1] > 0 and runner[0] > best_score - 1.5:
        status = "possible split profile"
    else:
        status = "ok"
    return best, status, best_inst, runner[2] if runner else None


# ============================================================
# PROFILES
# ============================================================

def work_row(w):
    r = work_to_row(w)
    pct = w.get("citation_normalized_percentile") or {}
    r["cnp_value"] = pct.get("value")
    r["top1"] = bool(pct.get("is_in_top_1_percent"))
    r["top10"] = bool(pct.get("is_in_top_10_percent"))
    return r


def fetch_profile(author_id):
    path = os.path.join(PROFILE_DIR, f"{author_id}.csv")
    if os.path.exists(path):
        return pd.read_csv(path)
    works = openalex_get_all("works", {"filter": f"author.id:{author_id}",
                                       "select": WORK_FIELDS + EXTRA_WORK_FIELDS})
    df = merge_versions(pd.DataFrame([work_row(w) for w in works]))
    df.to_csv(path, index=False)
    return df


# ============================================================
# RUN
# ============================================================

def fetch_listed():
    os.makedirs(PROFILE_DIR, exist_ok=True)
    hcr = hcr_2025_cs()
    hcr["Full Name"] = hcr["Full Name"].str.strip()
    hcr = hcr.drop_duplicates("Full Name").reset_index(drop=True)
    print(f"HCR-listed researchers: {len(hcr)}")

    rows = []
    for _, r in hcr.iterrows():
        a, status, inst, runner = resolve_author(r)
        row = {"HCR name": r["Full Name"], "HCR affiliation": r["Primary Affiliation"], "Status": status}
        if a is not None:
            row.update({
                "OpenAlex ID": short_id(a["id"]),
                "OpenAlex name": a["display_name"],
                "Works": a.get("works_count"),
                "Cited by": a.get("cited_by_count"),
                "Institution overlap": inst,
                "CS share": round(cs_share(a), 2),
                "Last known institution": "; ".join(i.get("display_name", "") for i in (a.get("last_known_institutions") or [])),
            })
        if runner is not None:
            row.update({"Runner-up ID": short_id(runner["id"]), "Runner-up name": runner["display_name"],
                        "Runner-up works": runner.get("works_count")})
        rows.append(row)
        print(f"  {r['Full Name'][:30]:30s} {status:28s} {row.get('OpenAlex ID', '')} {row.get('Works', '')}")

    res = pd.DataFrame(rows)
    res.to_csv(os.path.join(OUT_DIR, "resolution.csv"), index=False)

    print("\nDownloading profiles...")
    for _, r in res.dropna(subset=["OpenAlex ID"]).iterrows():
        df = fetch_profile(r["OpenAlex ID"])
        print(f"  {r['HCR name'][:30]:30s} {len(df):5d} works, top-1%: {int(df['top1'].sum())}")

    with open(os.path.join(OUT_DIR, "fetch_meta.json"), "w") as f:
        json.dump({"retrieved": pd.Timestamp.today().strftime("%Y-%m-%d"), "source": "OpenAlex",
                   "hcr_list": "Clarivate HCR 2025, Computer Science (inputs/hcr_lists/2025_HCR.xlsx)"}, f, indent=1)
    print(res["Status"].value_counts().to_string())


# =====================================================================================================================
# Derived columns of the Stanford-Elsevier career table
# =====================================================================================================================

def els_screen():
    """Derived columns of the Stanford-Elsevier career table (els.csv) used by profiles.py els-match
    ===================================================================================

    Reads only the Stanford-Elsevier files (inputs/stanford_elsevier/els.csv and, optionally, the
    2024 single-year table) and adds career and citation-profile columns to every record:

        Career years, Active years, Citation years, Career cohort (10-year bands)
        Citation velocity (ns/yr), h/hm (ns), Papers/year, 2024 rank jump (log10)
        peer percentiles (same Science-Metrix subfield and career cohort) and the
        survey proxy = mean(peer pct of LOW cprat (ns), peer pct of h/hm)

    els.csv has no document types, so the proxy is only a proxy: a survey is typically cited once by
    many unrelated papers, so citations per distinct citing paper (cprat) stays close to 1.
    The analyses use profiles.py els-match's match of the HCR-listed researchers to els.csv; the proxy itself
    is only compared with the measured survey dependence (profiles.py classify, summary.md).

    Self-citation is not used.

    Output: hcr_study/BTAS_Survey_Rise_All.csv"""
    # ============================================================
    # SETTINGS
    # ============================================================

    SE = os.path.join(HERE, "inputs", "stanford_elsevier")
    INPUT_FILE = os.path.join(SE, "els.csv")
    SINGLEYR_FILE = os.path.join(SE, "Table_1_Authors_singleyr_2024_pubs_since_1788_wopp_extracted_202508.xlsx")
    SINGLEYR_CACHE = os.path.join(STUDY, "singleyr_2024_growth_cache.csv")

    OUTPUT_CSV = os.path.join(STUDY, "BTAS_Survey_Rise_All.csv")

    END_YEAR = 2024
    CITATION_WINDOW_START = 1996
    MIN_GROUP_SIZE = 30

    # Profile thresholds (peer percentiles within subfield + career cohort).
    SURVEY_PROXY_MIN = 0.80
    VELOCITY_MIN = 0.90
    MAX_CAREER_YEARS = 20       # "rapid" rise: first paper within the last 20 years
    PROLIFIC_MIN = 0.90


    # ============================================================
    # HELPERS
    # ============================================================

    def peer_percentile(df, values):
        """Subfield + cohort -> cohort -> all, below MIN_GROUP_SIZE."""
        levels = [
            [df["sm-subfield-1"], df["Career cohort"]],
            [df["Career cohort"]],
            None,
        ]
        pct = pd.Series(np.nan, index=df.index)
        for keys in levels:
            if keys is None:
                p = values.rank(pct=True)
                n = pd.Series(values.notna().sum(), index=df.index)
            else:
                g = values.groupby(keys)
                p = g.rank(pct=True)
                n = g.transform("count")
            fill = pct.isna() & values.notna() & (n >= MIN_GROUP_SIZE)
            pct[fill] = p[fill]
        return pct


    def load_singleyr():
        """2024 single-year table (rank and citations in 2024 alone); optional, cached after the first read."""
        if os.path.exists(SINGLEYR_CACHE):
            return pd.read_csv(SINGLEYR_CACHE)
        if not os.path.exists(SINGLEYR_FILE):
            return None
        print("Reading single-year 2024 table (first run only, slow)...")
        sy = pd.read_excel(
            SINGLEYR_FILE,
            sheet_name="Data",
            usecols=["authfull", "firstyr", "np6024", "nc2424 (ns)", "h24 (ns)", "rank (ns)"],
        ).rename(columns={"h24 (ns)": "h 2024-only (ns)", "rank (ns)": "rank 2024 (ns)"})
        sy.to_csv(SINGLEYR_CACHE, index=False)
        return sy


    # ============================================================
    # READ DATA
    # ============================================================

    os.makedirs(STUDY, exist_ok=True)
    print("Reading els.csv...")

    df = pd.read_csv(INPUT_FILE, thousands=",", low_memory=False)
    df.columns = df.columns.str.replace("﻿", "", regex=False)

    for col in ["authfull", "inst_name", "cntry", "sm-subfield-1", "sm-field"]:
        df[col] = (
            df[col].fillna("").astype(str)
            .str.replace(r"\s+", " ", regex=True).str.strip()
        )
    df["sm-subfield-1"] = df["sm-subfield-1"].replace("", "Unknown")

    key = ["authfull", "firstyr", "np6024"]

    sy = load_singleyr()
    if sy is not None:
        sy = sy.dropna(subset=["authfull"]).drop_duplicates(key, keep=False)
        df = df.merge(sy[key + ["nc2424 (ns)", "rank 2024 (ns)"]], on=key, how="left")
    else:
        print(f"{os.path.basename(SINGLEYR_FILE)} not found; 2024 rank jump skipped.")
        df["nc2424 (ns)"] = np.nan
        df["rank 2024 (ns)"] = np.nan


    # ============================================================
    # CAREER
    # ============================================================

    first_year = df["firstyr"].where(df["firstyr"] > 0, df["lastyr"]).clip(upper=END_YEAR)
    last_year = df["lastyr"].clip(upper=END_YEAR)

    df["Career years"] = (END_YEAR - first_year + 1).clip(lower=1)
    df["Active years"] = (last_year - first_year + 1).clip(lower=1)
    df["Citation years"] = (
        END_YEAR - np.maximum(first_year, CITATION_WINDOW_START) + 1
    ).clip(lower=1)

    df["Career cohort"] = pd.cut(
        df["Career years"],
        bins=[0, 10, 20, 30, 40, 50, 1000],
        labels=["1-10", "11-20", "21-30", "31-40", "41-50", "51+"],
    ).astype(str)


    # ============================================================
    # PROXIES
    # ============================================================

    df["Citation velocity (ns/yr)"] = df["nc9624 (ns)"] / df["Citation years"]
    df["h/hm (ns)"] = df["h24 (ns)"] / df["hm24 (ns)"].replace(0, np.nan)
    df["Papers/year"] = df["np6024"] / df["Active years"]
    df["2024 rank jump (log10)"] = np.log10(df["rank (ns)"] / df["rank 2024 (ns)"])

    df["P low cprat"] = 1 - peer_percentile(df, df["cprat (ns)"])
    df["P h/hm"] = peer_percentile(df, df["h/hm (ns)"])
    df["P citation velocity"] = peer_percentile(df, df["Citation velocity (ns/yr)"])
    df["P papers/year"] = peer_percentile(df, df["Papers/year"])
    df["P 2024 rank jump"] = peer_percentile(df, df["2024 rank jump (log10)"])

    # Survey-dependence proxy.
    df["Survey proxy"] = (df["P low cprat"] + df["P h/hm"]) / 2


    # ============================================================
    # PROFILE MATCH (sets the row order of the output)
    # ============================================================

    df["C survey-style citations"] = df["Survey proxy"] >= SURVEY_PROXY_MIN
    df["C rapid citation growth"] = df["P citation velocity"] >= VELOCITY_MIN
    df["C recent career"] = df["Career years"] <= MAX_CAREER_YEARS
    df["C prolific / recurring co-authors"] = df["P papers/year"] >= PROLIFIC_MIN

    df["Profile match"] = (
        df["C survey-style citations"]
        & df["C rapid citation growth"]
        & df["C recent career"]
    )

    df["Profile score"] = df[[
        "Survey proxy", "P citation velocity", "P papers/year",
    ]].mean(axis=1) + 0.25 * df["P 2024 rank jump"].fillna(0)

    crit = [c for c in df.columns if c.startswith("C ")]
    df["Criteria met"] = df[crit].sum(axis=1)
    df["Criteria"] = df[crit].apply(lambda r: "; ".join(c[2:] for c in crit if r[c]), axis=1)

    df = df.sort_values(
        ["Profile match", "Profile score"], ascending=False
    ).reset_index(drop=True)
    df["Profile rank"] = np.arange(1, len(df) + 1)
    df["Subfield profile rank"] = (
        df.groupby("sm-subfield-1")["Profile rank"].rank(method="min").astype(int)
    )


    # ============================================================
    # OUTPUT
    # ============================================================

    df.to_csv(OUTPUT_CSV, index=False)
    print(f"{len(df):,} records; CSV: {OUTPUT_CSV}")


# =====================================================================================================================
# Stanford-Elsevier record of each HCR-listed researcher
# =====================================================================================================================

def els_match():
    """Clarivate Highly Cited Researchers (Computer Science) vs els.csv
    ================================================================

    Takes the Clarivate HCR 2025 Computer Science list (inputs/hcr_lists/2025_HCR.xlsx,
    read by profiles.hcr_2025_cs) and matches each person to els.csv (Stanford-Elsevier
    career table, with the columns added by profiles.py els-screen) by name, institution
    words and computing subfield. The matched els.csv rows (Scopus citations, papers,
    first year, Science-Metrix subfield) are used by profiles.py classify (Scopus consistency,
    field groups), comparison_groups.py controls (matching), analyses.py outlier-model and analyses.py composite.

    Within-HCR percentiles of the els.csv proxies are added for comparison:
        Survey proxy          mean(pct of LOW cprat (ns), pct of h/hm)
        Rapid rise            citation velocity, short career, 2024 rank jump

    Self-citation is not used.

    Output: hcr_study/BTAS_HCR_Computer_Science.csv"""
    # ============================================================
    # SETTINGS
    # ============================================================

    ELS_FILE = os.path.join(STUDY, "BTAS_Survey_Rise_All.csv")   # from profiles.py els-screen (els.csv + derived columns)

    OUTPUT_CSV = os.path.join(STUDY, "BTAS_HCR_Computer_Science.csv")

    END_YEAR = 2024

    # Namesakes rejected by hand: HCR name -> els.csv institution to reject (rule "els_reject" of
    # inputs/match_exceptions.csv, which is not released because it names people; see profiles.py fetch).
    MANUAL_REJECT = match_exceptions("els_reject")
    # manually verified career-table rows for names the rule misses or matches to several records (value "authfull|inst_name")
    MANUAL_MATCH = {k: tuple(v.split("|", 1)) for k, v in match_exceptions("els_match").items()}

    COMPUTING_SUBFIELDS = {
        "Networking & Telecommunications",
        "Artificial Intelligence & Image Processing",
        "Distributed Computing",
        "Information Systems",
        "Software Engineering",
        "Computer Hardware & Architecture",
        "Computation Theory & Mathematics",
        "Industrial Engineering & Automation",
        "Electrical & Electronic Engineering",
        "Bioinformatics",
    }

    STOPWORDS = {
        "university", "of", "the", "and", "technology", "science", "institute",
        "school", "college", "china", "mainland", "united", "states", "kingdom",
        "department", "engineering", "sciences", "national", "research", "de",
        "for", "hong", "kong", "sar", "south", "-",
    }


    # ============================================================
    # NAME MATCHING
    # ============================================================

    def norm(s):
        s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()
        return re.sub(r"[^a-z]", "", s)


    def split_name(name):
        if "," not in name:
            return norm(name), []
        last, first = name.split(",", 1)
        first = re.sub(r"\(.*?\)", "", first)
        return norm(last), [norm(t) for t in re.split(r"[\s\-\.]+", first) if norm(t)]


    def tokens(s):
        words = re.findall(r"[a-z]+", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 2}


    def match_score(hcr, cand):
        """Higher is better; None if the name is not compatible."""
        h_last, h_first = split_name(hcr["Full Name"])
        c_last, c_first = split_name(cand["authfull"])
        if h_last != c_last or not h_first or not c_first:
            return None

        # Conflicting middle initials (e.g. "Jane Q." vs "Jane R.") -> different person.
        if len(h_first) > 1 and len(c_first) > 1 and h_first[0] == c_first[0] \
                and h_first[1][0] != c_first[1][0]:
            return None

        h_given = "".join(h_first)
        c_given = "".join(c_first)
        if c_given == h_given or (c_first[0] == h_first[0] and (
            len(h_first) == 1 or len(c_first) == 1 or c_given.startswith(h_given)
        )):
            name_pts = 3
        elif c_given.startswith(h_first[0]) and len(h_first[0]) > 1:
            name_pts = 1  # e.g. "Jo" vs "Joanna": weak
        else:
            return None

        affil = tokens(hcr["Primary Affiliation"]) | tokens(hcr.get("Secondary Affiliations", ""))
        inst_pts = len(affil & tokens(cand["inst_name"]))
        field_pts = 2 if cand["sm-subfield-1"] in COMPUTING_SUBFIELDS else 0

        # A weak name match needs institutional support.
        if name_pts == 1 and inst_pts == 0:
            return None
        # Outside computing, require institutional support too.
        if field_pts == 0 and inst_pts == 0:
            return None
        return name_pts + 2 * inst_pts + field_pts


    # ============================================================
    # READ
    # ============================================================

    hcr = hcr_2025_cs()
    hcr["Full Name"] = hcr["Full Name"].str.strip()
    hcr = hcr.drop_duplicates("Full Name").reset_index(drop=True)

    els = pd.read_csv(ELS_FILE, low_memory=False)
    els["_last"] = els["authfull"].map(lambda n: split_name(n)[0])

    print(f"HCR names: {len(hcr)}")

    rows = []
    for _, h in hcr.iterrows():
        if h["Full Name"] in MANUAL_MATCH:
            a, inst = MANUAL_MATCH[h["Full Name"]]
            hit = els.index[(els["authfull"] == a) & (els["inst_name"] == inst)]
            assert len(hit) == 1, (h["Full Name"], len(hit))
            rows.append({"HCR name": h["Full Name"], "HCR affiliation": h["Primary Affiliation"], "Match status": "ok",
                         "Match score": "manual", "Candidates": 1, "els index": hit[0]})
            continue
        last = split_name(h["Full Name"])[0]
        pool = els[els["_last"] == last]
        scored = []
        for idx, c in pool.iterrows():
            if MANUAL_REJECT.get(h["Full Name"]) == c["inst_name"]:
                continue
            s = match_score(h, c)
            if s is not None:
                scored.append((s, idx))
        scored.sort(reverse=True)

        if not scored:
            rows.append({"HCR name": h["Full Name"], "HCR affiliation": h["Primary Affiliation"],
                         "Match status": "not found", "els index": None})
            continue

        best_s, best_idx = scored[0]
        runner = scored[1][0] if len(scored) > 1 else None
        if best_s < 3:
            status = "weak"
        elif runner is not None and runner >= best_s - 1:
            status = "ambiguous"
        else:
            status = "ok"
        rows.append({"HCR name": h["Full Name"], "HCR affiliation": h["Primary Affiliation"],
                     "Match status": status, "Match score": best_s,
                     "Candidates": len(scored), "els index": best_idx})

    matches = pd.DataFrame(rows)

    # Matches with status ok or ambiguous are kept; the analyses use status ok only
    # (ambiguous name matches are often namesakes).
    use = matches[matches["Match status"].isin(["ok", "ambiguous"])].copy()
    print(matches["Match status"].value_counts().to_string())

    df = els.loc[use["els index"].astype(int)].copy()
    df.insert(0, "HCR name", use["HCR name"].values)
    df.insert(1, "HCR affiliation", use["HCR affiliation"].values)
    df.insert(2, "Match status", use["Match status"].values)
    df = df.reset_index(drop=True)


    # ============================================================
    # WITHIN-HCR PERCENTILES
    # ============================================================

    def pct(s, higher=True):
        return (s if higher else -s).rank(pct=True)


    df["HCR pct low cprat"] = pct(df["cprat (ns)"], higher=False)
    df["HCR pct h/hm"] = pct(df["h/hm (ns)"])
    df["HCR pct citation velocity"] = pct(df["Citation velocity (ns/yr)"])
    df["HCR pct short career"] = pct(df["Career years"], higher=False)
    df["HCR pct papers/year"] = pct(df["Papers/year"])
    df["HCR pct 2024 rank jump"] = pct(df["2024 rank jump (log10)"])

    df["HCR survey proxy"] = (df["HCR pct low cprat"] + df["HCR pct h/hm"]) / 2
    df["HCR rapid rise"] = df[[
        "HCR pct citation velocity", "HCR pct short career", "HCR pct 2024 rank jump",
    ]].mean(axis=1)

    # Combined score (orders the output rows).
    df["HCR pattern score"] = (df["HCR survey proxy"] + df["HCR rapid rise"]) / 2

    df = df.sort_values("HCR pattern score", ascending=False).reset_index(drop=True)
    df.insert(0, "HCR pattern rank", np.arange(1, len(df) + 1))


    # ============================================================
    # OUTPUT
    # ============================================================

    cols = [
        "HCR pattern rank", "HCR name", "HCR affiliation", "authfull", "inst_name",
        "sm-subfield-1", "firstyr", "Match status",
        "HCR pattern score", "HCR survey proxy", "HCR rapid rise",
        "HCR pct low cprat", "HCR pct h/hm", "HCR pct citation velocity",
        "HCR pct short career", "HCR pct 2024 rank jump", "HCR pct papers/year",
        "cprat (ns)", "h/hm (ns)", "Citation velocity (ns/yr)", "Papers/year",
        "2024 rank jump (log10)", "np6024", "nc9624 (ns)", "h24 (ns)", "hm24 (ns)",
        "Survey proxy", "Profile match",
    ]
    df[cols].to_csv(OUTPUT_CSV, index=False)


    # ============================================================
    # CONSOLE
    # ============================================================

    print(f"\nMatched: {len(df)} (ok {int((df['Match status'] == 'ok').sum())}, ambiguous "
          f"{int((df['Match status'] == 'ambiguous').sum())}); written to {OUTPUT_CSV}")


# =====================================================================================================================
# HCR Computer Science lists 2014-2025
# =====================================================================================================================

def history():
    """Researchers listed in the Clarivate HCR Computer Science category, 2014-2025.

    Builds one row per person from Clarivate's historical lists (inputs/hcr_lists/<year>_HCR.xlsx),
    resolves each person in OpenAlex with the same rules as the 2025 study (profiles.resolve_author; the
    2025 resolutions are reused), matches each to els.csv (Scopus) with the rules of profiles.py els-match,
    retrieves and classifies the profiles with the final model, and applies the same plausibility
    checks (OpenAlex/Scopus citation ratio 0.6-3.0, paper ratio 0.4-3.0).

        python profiles.py history build
        python profiles.py history resolve     (resumable; WORKER=k NWORKERS=n for parallel parts)
        python profiles.py history merge       (after parallel parts)
        python profiles.py history els
        python profiles.py history fetch       (profiles of every usable resolution, so that their abstracts can be filled
                                           by ground_truth.py s2-abstracts and ground_truth.py publisher-abstracts before classification)
        python profiles.py history classify    (resumable)

    Output: hcr_study/history/ (names.csv, resolution.csv, els_match.csv, per_researcher.csv)"""
    HCR_DIR = os.path.join(HERE, "inputs", "hcr_lists")
    OUT = os.path.join(HERE, "hcr_study", "history")
    COMPUTING_SUBFIELDS = {
        "Networking & Telecommunications", "Artificial Intelligence & Image Processing", "Distributed Computing",
        "Information Systems", "Software Engineering", "Computer Hardware & Architecture",
        "Computation Theory & Mathematics", "Industrial Engineering & Automation",
        "Electrical & Electronic Engineering", "Bioinformatics",
    }
    STOPWORDS = {
        "university", "of", "the", "and", "technology", "science", "institute", "school", "college", "china",
        "mainland", "united", "states", "kingdom", "department", "engineering", "sciences", "national",
        "research", "de", "for", "hong", "kong", "sar", "south", "-", "univ", "inst", "usa", "peoples", "republic",
    }
    RANK = {"ok": 0, "known ID": 0, "possible split profile": 1, "weak (no institution match)": 2,
            "weak (low CS share)": 2, "not found": 3}


    def norm(s):
        return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


    def person_key(first, last):
        first = re.sub(r"[\u2010-\u2015]", "-", str(first)).strip()
        return norm(last) + "|" + norm(re.split(r"[\s.\-]+", first)[0])


    def read_lists():
        """All historical lists, Computer Science category only, with harmonised columns."""
        frames = []
        for f in sorted(glob.glob(os.path.join(HCR_DIR, "*_HCR.xlsx"))):
            df = pd.read_excel(f, sheet_name=0, engine="calamine")
            cols = {}
            for c in df.columns:
                k = str(c).lower().replace(" ", "").replace("_", "")
                if k.startswith("first"):
                    cols[c] = "first"
                elif k in ("lastname", "familyname"):
                    cols[c] = "last"
                elif k == "category":
                    cols[c] = "category"
                elif "primary" in k:
                    cols[c] = "primary"
                elif "secondary" in k:
                    cols[c] = "secondary"
            df = df.rename(columns=cols).reindex(columns=["first", "last", "category", "primary", "secondary"])
            df["list year"] = int(os.path.basename(f)[:4])
            frames.append(df)
        h = pd.concat(frames, ignore_index=True)
        h = h[h["category"].astype(str).str.strip() == "Computer Science"].copy()
        for c in ("first", "last", "primary", "secondary"):
            h[c] = h[c].fillna("").astype(str).str.strip()
        h["key"] = [person_key(f, l) for f, l in zip(h["first"], h["last"])]
        return h


    def build():
        os.makedirs(OUT, exist_ok=True)
        h = read_lists()
        h.to_csv(os.path.join(OUT, "listings.csv"), index=False)
        rows = []
        for key, e in h.sort_values("list year").groupby("key"):
            last = e.iloc[-1]
            # the most complete given-name variant
            first = max(e["first"], key=len)
            affs = list(dict.fromkeys(a for a in e["primary"][::-1].tolist() + e["secondary"][::-1].tolist() if a))
            rows.append({"key": key, "Full Name": f"{last['last']}, {first}", "Primary Affiliation": last["primary"],
                         "Secondary Affiliations": last["secondary"], "All affiliations": " | ".join(affs),
                         "Years listed": ";".join(str(y) for y in sorted(e["list year"])),
                         "Times listed": len(e), "First listed": e["list year"].min(), "Last listed": e["list year"].max()})
        names = pd.DataFrame(rows)
        names.to_csv(os.path.join(OUT, "names.csv"), index=False)
        print(f"{len(h)} listings, {len(names)} people; per year: {h.groupby('list year').size().to_dict()}")


    def resolve():

        names = pd.read_csv(os.path.join(OUT, "names.csv")).fillna("")
        prev = pd.read_csv(os.path.join(HERE, "hcr_study", "resolution.csv"))
        prev["key"] = [person_key(n.split(",", 1)[1], n.split(",", 1)[0]) for n in prev["HCR name"]]
        prev = prev.set_index("key")
        # Optional parallel workers: WORKER=k NWORKERS=n, each writes its own part file.
        k, n = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
        path = os.path.join(OUT, "resolution.csv" if n == 1 else f"resolution_part{k}.csv")
        parts = [os.path.join(OUT, f) for f in os.listdir(OUT) if f.startswith("resolution")]
        done = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True) if parts else pd.DataFrame(columns=["key"])
        done = done[~done["Status"].astype(str).str.startswith("error")].drop_duplicates("key") if len(done) else done
        mine = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["key", "Status"])
        rows = mine[mine["key"].isin(set(done["key"]))].to_dict("records")
        todo = names[~names["key"].isin(set(done["key"]))].iloc[k::n]
        print(f"{len(names)} people, {len(done)} resolved earlier, {len(todo)} to do")

        for i, (_, r) in enumerate(todo.iterrows(), 1):
            row = {"key": r["key"], "Full Name": r["Full Name"]}
            if r["key"] in prev.index and pd.notna(prev.loc[r["key"], "OpenAlex ID"]):
                p = prev.loc[r["key"]]
                row.update({"Status": p["Status"], "OpenAlex ID": p["OpenAlex ID"], "OpenAlex name": p["OpenAlex name"],
                            "Source": "2025 study"})
            else:
                best = None
                # Latest affiliation first, then the earlier ones.
                affs = [a for a in r["All affiliations"].split(" | ") if a] or [""]
                try:
                    for aff in affs[:4]:
                        a, status, inst, _ = resolve_author(pd.Series({"Full Name": r["Full Name"], "Primary Affiliation": aff,
                                                                  "Secondary Affiliations": ""}))
                        if best is None or RANK.get(status, 9) < RANK.get(best[1], 9):
                            best = (a, status, inst, aff)
                        if RANK.get(status, 9) == 0:
                            break
                except Exception as e:
                    rows.append({**row, "Status": f"error: {e}"[:80]})
                    continue
                a, status, inst, aff = best
                row.update({"Status": status, "Matched affiliation": aff, "Source": "resolved"})
                if a is not None:
                    row.update({"OpenAlex ID": short_id(a["id"]), "OpenAlex name": a["display_name"],
                                "Works": a.get("works_count"), "Cited by": a.get("cited_by_count"),
                                "Institution overlap": inst, "CS share": round(cs_share(a), 2)})
            if row.get("OpenAlex ID") and RANK.get(row["Status"], 9) <= 1:
                try:
                    fetch_profile(row["OpenAlex ID"])
                except Exception as e:
                    row["Status"] = f"error: {e}"[:80]
            rows.append(row)
            if i % 20 == 0:
                pd.DataFrame(rows).to_csv(path, index=False)
                print(f"  {i}/{len(todo)}", flush=True)
        res = pd.DataFrame(rows)
        res.to_csv(path, index=False)
        print(res["Status"].value_counts().to_string())


    def merge_parts():
        parts = [os.path.join(OUT, f) for f in sorted(os.listdir(OUT)) if f.startswith("resolution")]
        res = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True)
        res["_err"] = res["Status"].astype(str).str.startswith("error")
        res = res.sort_values("_err").drop_duplicates("key").drop(columns="_err")
        res.to_csv(os.path.join(OUT, "resolution.csv"), index=False)
        for f in parts:
            if os.path.basename(f) != "resolution.csv":
                os.remove(f)
        print(f"{len(res)} people: " + ", ".join(f"{k} {v}" for k, v in res["Status"].value_counts().items()))


    def tokens(s):
        words = re.findall(r"[a-z]+", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())
        return {w for w in words if w not in STOPWORDS and len(w) > 2}


    def split_name(name):
        last, first = name.split(",", 1)
        first = re.sub(r"\(.*?\)", "", first)
        return norm(last), [norm(t) for t in re.split(r"[\s\-.]+", first) if norm(t)]


    def match_score(full_name, affs, cand):
        """match_score of profiles.py els-match, with all the person's listed affiliations."""
        h_last, h_first = split_name(full_name)
        c_last, c_first = split_name(cand["authfull"]) if "," in str(cand["authfull"]) else (None, [])
        if h_last != c_last or not h_first or not c_first:
            return None
        if len(h_first) > 1 and len(c_first) > 1 and h_first[0] == c_first[0] and h_first[1][0] != c_first[1][0]:
            return None
        h_given, c_given = "".join(h_first), "".join(c_first)
        if c_given == h_given or (c_first[0] == h_first[0] and (len(h_first) == 1 or len(c_first) == 1
                                                                or c_given.startswith(h_given))):
            name_pts = 3
        elif c_given.startswith(h_first[0]) and len(h_first[0]) > 1:
            name_pts = 1
        else:
            return None
        inst_pts = len(affs & tokens(cand["inst_name"]))
        field_pts = 2 if cand["sm-subfield-1"] in COMPUTING_SUBFIELDS else 0
        if (name_pts == 1 and inst_pts == 0) or (field_pts == 0 and inst_pts == 0):
            return None
        return name_pts + 2 * inst_pts + field_pts


    def els_match():
        # manually verified career-table rows (inputs/match_exceptions.csv, rule els_match, value "authfull|inst_name")
        manual = {k: tuple(v.split("|", 1)) for k, v in match_exceptions("els_match").items()}
        names = pd.read_csv(os.path.join(OUT, "names.csv")).fillna("")
        els = pd.read_csv(os.path.join(HERE, "inputs", "stanford_elsevier", "els.csv"), thousands=",", low_memory=False,
                          usecols=["authfull", "inst_name", "cntry", "np6024", "firstyr", "lastyr", "nc9624 (ns)",
                                   "sm-subfield-1", "rank (ns)"])
        els["_last"] = els["authfull"].map(lambda n: split_name(n)[0] if "," in str(n) else norm(n))
        by_last = dict(tuple(els.groupby("_last")))
        rows = []
        for _, r in names.iterrows():
            affs = set().union(*[tokens(a) for a in r["All affiliations"].split(" | ")])
            pool = by_last.get(split_name(r["Full Name"])[0], els.iloc[:0])
            scored = sorted(((s, i) for i, c in pool.iterrows() if (s := match_score(r["Full Name"], affs, c)) is not None),
                            reverse=True)
            row = {"key": r["key"]}
            if r["Full Name"] in manual:
                a, inst = manual[r["Full Name"]]
                hit = els.index[(els["authfull"] == a) & (els["inst_name"] == inst)]
                assert len(hit) == 1, r["Full Name"]
                scored = [(99, hit[0])]
            if not scored:
                rows.append({**row, "els status": "not found"})
                continue
            (best_s, idx), runner = scored[0], (scored[1][0] if len(scored) > 1 else None)
            status = "weak" if best_s < 3 else ("ambiguous" if runner is not None and runner >= best_s - 1 else "ok")
            c = els.loc[idx]
            rows.append({**row, "els status": status, "authfull": c["authfull"], "inst_name": c["inst_name"],
                         "firstyr": c["firstyr"], "np6024": c["np6024"], "nc9624 (ns)": c["nc9624 (ns)"],
                         "sm-subfield-1": c["sm-subfield-1"]})
        m = pd.DataFrame(rows)
        m.to_csv(os.path.join(OUT, "els_match.csv"), index=False)
        print(m["els status"].value_counts().to_string())


    def usable(res):
        """Usable resolutions. 'weak (low CS share)' is accepted when the OpenAlex name has the listed
    surname and first given name: early lists put many bioinformaticians and statisticians in the
    Computer Science category, whose profiles are mostly outside computing."""
        ok = res["Status"].isin(["ok", "known ID", "possible split profile"])
        weak = (res["Status"] == "weak (low CS share)") & np.array([
            isinstance(n, str) and person_key(" ".join(n.split()[:-1]), n.split()[-1]) == k
            for n, k in zip(res["OpenAlex name"], res["key"])])
        return res[(ok | weak) & res["OpenAlex ID"].notna()].copy()


    def fetch():
        res = usable(pd.read_csv(os.path.join(OUT, "resolution.csv")))
        for i, aid in enumerate(res["OpenAlex ID"], 1):
            fetch_profile(aid)
            if i % 50 == 0:
                print(f"  profiles {i}/{len(res)}", flush=True)


    def classify():

        res = usable(pd.read_csv(os.path.join(OUT, "resolution.csv")))
        prev = pd.read_csv(os.path.join(HERE, "hcr_study", "resolution.csv"))
        cs_prev = dict(zip(prev["OpenAlex ID"], prev["CS share"]))
        res["CS share"] = res["CS share"].fillna(res["OpenAlex ID"].map(cs_prev))
        em = pd.read_csv(os.path.join(OUT, "els_match.csv")).set_index("key")
        names = pd.read_csv(os.path.join(OUT, "names.csv")).set_index("key")
        # Optional parallel workers (WORKER=k NWORKERS=n) only fill the per-paper cache.
        k, n = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
        rows = []
        for i, (_, r) in enumerate(res.iloc[k::n].iterrows(), 1):
            aid = r["OpenAlex ID"]
            df = fetch_profile(aid)
            df["citationCount"] = df["citationCount"].fillna(0).astype(int)
            cpath = os.path.join(CLASSIFIED_DIR, f"{aid}.csv")
            if os.path.exists(cpath):
                papers = pd.read_csv(cpath)
            else:
                papers, _ = classify_papers(df)
                papers.to_csv(cpath, index=False)
            if i % 25 == 0:
                print(f"  classified {i}/{len(res) // n}", flush=True)
            if n > 1:
                continue
            m, _, _ = author_metrics(df, papers["Category"].values)
            e = em.loc[r["key"]] if r["key"] in em.index else None
            ok_els = e is not None and e["els status"] == "ok"
            cr = m["Citations"] / e["nc9624 (ns)"] if ok_els else np.nan
            pr = m["Papers"] / e["np6024"] if ok_els else np.nan
            nm = names.loc[r["key"]]
            sub = e["sm-subfield-1"] if ok_els else None
            rows.append({"key": r["key"], "Full Name": r["Full Name"], "OpenAlex ID": aid, "Match status": r["Status"],
                         "CS share": r["CS share"],
                         "Core computing": bool((r["CS share"] >= 0.5) and (sub is None or sub in COMPUTING_SUBFIELDS)),
                         "Years listed": nm["Years listed"], "Times listed": nm["Times listed"],
                         "First listed": nm["First listed"], "Last listed": nm["Last listed"],
                         "els status": e["els status"] if e is not None else "not found",
                         "Subfield (els.csv)": sub, "firstyr": e["firstyr"] if ok_els else None,
                         "OpenAlex/Scopus citations": cr, "OpenAlex/Scopus papers": pr,
                         "Consistent with Scopus": bool(ok_els and 0.6 <= cr <= 3.0 and 0.4 <= pr <= 3.0),
                         **m})
        if n > 1:
            return
        per = pd.DataFrame(rows)
        per.to_csv(os.path.join(OUT, "per_researcher.csv"), index=False)
        print(f"{len(per)} classified; core computing {int(per['Core computing'].sum())}; Scopus-consistent "
              f"{int(per['Consistent with Scopus'].sum())}")


    {"build": build, "resolve": resolve, "merge": merge_parts, "els": els_match, "fetch": fetch,
     "classify": classify}[sys.argv[1]]()


# =====================================================================================================================
# Classification of the HCR-listed researchers' profiles
# =====================================================================================================================
#
# HCR study, step 2: apply the SciGenre to the researchers on the Clarivate HCR (Computer
# Science) list and measure how much of each record comes from surveys.
#
# Classification: the final model of RQ1 (train.py final, installed by train.py install, applied by
# profiles.py): SPECTER2 fine-tuned on venue labels, title and abstract only (never the venue), threshold 0.5.
# Surveys are removed; non-papers and papers without an abstract are kept in both versions.
#
# Per HCR-listed researcher:
#     papers, surveys, share of papers, share of citations, h / h*, i10 / i10*
#     top-1% papers in the HCR window (OpenAlex citation-normalized percentile;
#     an approximation of Clarivate's Highly Cited Papers), how many are
#     surveys, and the count left without them
#
# Group level:
#     flagged and error-corrected survey shares (adjusted classify-and-count,
#     Forman 2008), using the final model's recall and false-positive rate on the
#     held-out test split of the venue ground truth (model/scigenre.json)
#     split by research field (Science-Metrix subfield from els.csv)
#
# Run after profiles.py fetch, profiles.py els-match and train.py install:
#     python profiles.py classify

# ============================================================
# SETTINGS
# ============================================================

MODEL = MODEL_DIR
CLASSIFIED_DIR = os.path.join(STUDY, "classified")    # per-paper results, one CSV per researcher
os.makedirs(CLASSIFIED_DIR, exist_ok=True)
RESOLUTION = os.path.join(STUDY, "resolution.csv")
FIELD_FILE = os.path.join(STUDY, "BTAS_HCR_Computer_Science.csv")   # els.csv row per HCR-listed researcher (profiles.py els-match)

# Clarivate HCR 2025 counts Highly Cited Papers published 2014-2024.
HCP_START, HCP_END = 2014, 2024

USABLE_STATUS = {"ok", "known ID", "possible split profile"}


# ============================================================
# ERROR RATES FOR THE CORRECTION
# ============================================================

def error_rates():
    """Recall and false-positive rate of the final model on the held-out test split of the venue ground truth."""
    return TEST_RECALL, TEST_FPR, "final model on the held-out test split of the venue ground truth"


def corrected_shares(flagged, cites, tpr, fpr):
    """Per-profile adjusted classify-and-count (rq1_evaluation.corrected_shares)."""
    q = flagged.mean()
    c = cites[flagged].sum() / max(cites.sum(), 1)
    pi, cite_share = rq1_evaluation.corrected_shares(q, c, tpr, fpr)
    return float(pi), float(cite_share)


# ============================================================
# CLASSIFY ONE PROFILE
# ============================================================


def classify_papers(df):
    """
    Per-paper result for one profile: the final model (profiles.py) plus, for comparison, the paper's
    baselines (indexer type, Smyth & Cunningham rule). Returns (classified frame, prepared frame).
    """
    res = score_profile(df)
    frame = prepare_frame(df)
    out = df.copy()
    for c in res.columns:
        out[c] = res[c].values
    out["Survey (indexer type)"] = rq1_evaluation.indexer_is_review(frame) == 1
    out["Survey (Smyth & Cunningham)"] = rq1_evaluation.smyth_cunningham_rule(frame) == 1
    return out, frame


def author_metrics(df, category):
    survey = category == "survey"
    cites = df["citationCount"].fillna(0).astype(int).values
    h0, i0 = calculate_indices(df)
    h1, i1 = calculate_indices(df[~survey])

    window = df["year"].between(HCP_START, HCP_END).values
    top1 = df["top1"].fillna(False).astype(bool).values & window
    return {
        "Papers": len(df),
        "Surveys": int(survey.sum()),
        "Non-papers": int((category == "non-paper").sum()),
        "No abstract": int((category == "no abstract").sum()),
        "No abstract % of citations": 100 * cites[category == "no abstract"].sum() / max(cites.sum(), 1),
        "Citations": int(cites.sum()),
        "Citations*": int(cites[~survey].sum()),
        "Survey % of papers": 100 * survey.mean(),
        "Survey % of citations": 100 * cites[survey].sum() / max(cites.sum(), 1),
        # Dual reporting: the shares above are lower bounds (unclassified papers count as non-surveys); these use
        # only the papers the model could classify (survey or research).
        "Survey % of classified papers": 100 * survey.sum() / max(np.isin(category, ["survey", "research"]).sum(), 1),
        "Survey % of citations (classified)": 100 * cites[survey].sum() / max(cites[np.isin(category, ["survey", "research"])].sum(), 1),
        "h-index": h0, "h-index*": h1, "h drop": h0 - h1, "h change %": 100 * (h1 - h0) / max(h0, 1),
        "i10": i0, "i10*": i1,
        f"Top-1% papers {HCP_START}-{HCP_END}": int(top1.sum()),
        "Top-1% surveys": int((top1 & survey).sum()),
        "Top-1% papers*": int((top1 & ~survey).sum()),
        "Survey % of top-1%": 100 * (top1 & survey).sum() / max(top1.sum(), 1),
    }, survey, cites


# ============================================================
# RUN
# ============================================================

def classify_listed():
    tpr, fpr, rate_source = error_rates()
    print(f"Error rates for correction: TPR {tpr:.3f}, FPR {fpr:.4f} ({rate_source})")
    reuse = os.environ.get("REUSE_CLASSIFIED") == "1"   # reuse classifications made by the final model

    res = pd.read_csv(RESOLUTION)
    use = res[res["Status"].isin(USABLE_STATUS) & res["OpenAlex ID"].notna()]
    print(f"HCR-listed researchers usable: {len(use)} of {len(res)} (others are not resolved unambiguously)")

    # els.csv (Scopus) values per HCR-listed researcher, from profiles.py els-match's name matching.
    fields, scopus_cites, scopus_papers = {}, {}, {}
    if os.path.exists(FIELD_FILE):
        f = pd.read_csv(FIELD_FILE)
        f = f[f["Match status"] == "ok"]   # ambiguous name matches are often namesakes
        fields = dict(zip(f["HCR name"], f["sm-subfield-1"]))
        scopus_papers = dict(zip(f["HCR name"], f["np6024"]))
        scopus_cites = dict(zip(f["HCR name"], f["nc9624 (ns)"]))

    rows, flagged_papers = [], []
    for i, r in enumerate(use.itertuples(index=False), 1):
        path = os.path.join(PROFILE_DIR, f"{r[res.columns.get_loc('OpenAlex ID')]}.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        df["citationCount"] = df["citationCount"].fillna(0).astype(int)
        cpath = os.path.join(CLASSIFIED_DIR, os.path.basename(path))
        if reuse and os.path.exists(cpath) and "Abstract source" in pd.read_csv(cpath, nrows=1).columns:
            papers = pd.read_csv(cpath)
        else:
            papers, frame = classify_papers(df)
            papers.to_csv(cpath, index=False)
        category, score = papers["Category"].values, papers["SurveyScore"].values
        m, survey, cites = author_metrics(df, category)
        pi, cite_corr = corrected_shares(survey, cites, tpr, fpr)

        name = r[res.columns.get_loc("HCR name")]
        aid = r[res.columns.get_loc("OpenAlex ID")]
        subfield = fields.get(name, "Unmatched in els.csv")
        group = subfield if subfield in ("Networking & Telecommunications",
                                         "Artificial Intelligence & Image Processing") else "Other / unmatched"
        # Profile consistency with Scopus (els.csv): OpenAlex counts all citations and
        # typically runs higher than Scopus non-self citations, so only gross mismatches
        # (a namesake or a split record) are flagged.
        sc, sp = scopus_cites.get(name), scopus_papers.get(name)
        cite_ratio = m["Citations"] / sc if sc else np.nan
        paper_ratio = m["Papers"] / sp if sp else np.nan
        consistent = bool(sc) and 0.6 <= cite_ratio <= 3.0 and 0.4 <= paper_ratio <= 3.0

        rows.append({
            "HCR name": name, "OpenAlex ID": aid,
            "Match status": r[res.columns.get_loc("Status")],
            "OpenAlex/Scopus citations": cite_ratio,
            "OpenAlex/Scopus papers": paper_ratio,
            "Consistent with Scopus": consistent,
            "Subfield (els.csv)": subfield, "Field group": group,
            "Model": "final SPECTER2 (title + abstract)",
            **m,
            "Corrected survey % of papers": 100 * pi,
            "Corrected survey % of citations": 100 * cite_corr,
        })

        top = df.assign(Category=category, SurveyScore=np.round(score, 3))
        top = top[top["Category"] == "survey"].nlargest(5, "citationCount")
        for t in top.itertuples():
            flagged_papers.append({"HCR name": name, "title": t.title, "year": t.year,
                                   "venue": t.venue, "citations": t.citationCount,
                                   "top1": t.top1, "SurveyScore": t.SurveyScore})
        print(f"  [{i:3d}/{len(use)}] {name[:28]:28s} papers {m['Papers']:5d}  surveys {m['Surveys']:4d}  "
              f"cites {m['Survey % of citations']:5.1f}%  dh {m['h drop']:3d}  "
              f"top1 {m[f'Top-1% papers {HCP_START}-{HCP_END}']:3d}->{m['Top-1% papers*']:3d}")

    per = pd.DataFrame(rows)
    per.to_csv(os.path.join(STUDY, "per_member.csv"), index=False)
    pd.DataFrame(flagged_papers).to_csv(os.path.join(STUDY, "top_flagged_surveys.csv"), index=False)

    # ---------------- Group tables ----------------
    top1_col = f"Top-1% papers {HCP_START}-{HCP_END}"
    cols = ["Papers", "Surveys", "Survey % of papers", "Corrected survey % of papers",
            "Survey % of citations", "Corrected survey % of citations",
            "h-index", "h drop", "h change %", top1_col, "Top-1% surveys", "Survey % of top-1%"]

    def summarize(d, label):
        s = {"Group": label, "Researchers": len(d)}
        for c in cols:
            s[f"{c} (mean)"] = d[c].mean()
        s["Survey % of citations (median)"] = d["Survey % of citations"].median()
        s["Survey % of citations (IQR)"] = f"{d['Survey % of citations'].quantile(.25):.1f}-{d['Survey % of citations'].quantile(.75):.1f}"
        s["Researchers with >=25% of top-1% papers surveys"] = int((d["Survey % of top-1%"] >= 25).sum())
        return s

    verified = per[per["Consistent with Scopus"]]
    print(f"\nProfiles consistent with Scopus: {len(verified)} of {len(per)}")
    groups = [(per, "All classified HCR-listed researchers"), (verified, "Scopus-consistent profiles")]
    groups += [(verified[verified["Field group"] == g], f"Scopus-consistent: {g}")
               for g in ["Networking & Telecommunications", "Artificial Intelligence & Image Processing", "Other / unmatched"]]
    summary = pd.DataFrame([summarize(d, l) for d, l in groups if len(d)])

    # HCP qualification proxy: every listed researcher qualified, so the smallest top-1% count among
    # them is a lower bound on what qualification required (in OpenAlex terms).
    floor = int(per[top1_col].quantile(0.05))
    below = per[(per[top1_col] >= floor) & (per["Top-1% papers*"] < floor)]

    # Does the els.csv survey proxy track measured survey dependence? (validation at scale)
    proxy_note = ""
    if os.path.exists(FIELD_FILE):
        f = pd.read_csv(FIELD_FILE)[["HCR name", "cprat (ns)", "HCR survey proxy"]]
        j = per.merge(f, on="HCR name")
        if len(j) > 10:
            rho_c = j["Survey % of citations"].corr(j["cprat (ns)"], method="spearman")
            rho_p = j["Survey % of citations"].corr(j["HCR survey proxy"], method="spearman")
            proxy_note = (f"els.csv proxies vs measured survey % of citations (n = {len(j)}): "
                          f"Spearman with cprat (ns) {rho_c:.2f}, with survey proxy {rho_p:.2f}.")

    summary.to_csv(os.path.join(STUDY, "cohort_summary.csv"), index=False)

    with open(os.path.join(STUDY, "summary.md"), "w") as fh:
        fh.write("# SciGenre on Clarivate HCR (Computer Science)\n\n")
        fh.write(f"Error correction: TPR {tpr:.3f}, FPR {fpr:.4f} ({rate_source}).\n\n")
        fh.write(f"HCR-listed researchers classified: {len(per)}; consistent with Scopus: {len(verified)}.\n\n")
        fh.write(summary.round(1).to_markdown(index=False) + "\n\n")
        fh.write(f"Top-1% (OpenAlex) qualification proxy: 5th percentile of the listed researchers' counts = {floor}. "
                 f"HCR-listed researchers at or above it who fall below it without surveys: {len(below)}.\n\n")
        if proxy_note:
            fh.write(proxy_note + "\n")

    pd.set_option("display.width", 250)
    print("\n" + summary.round(1).T.to_string())
    print(f"\nQualification proxy (5th pct of top-1% counts) = {floor}; researchers falling below it without surveys: {len(below)}")
    if proxy_note:
        print(proxy_note)
    print(f"\nWritten to {STUDY}/")


STEPS = {"fetch": fetch_listed, "els-screen": els_screen, "els-match": els_match, "history": history, "classify": classify_listed}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv.pop(1)]()
