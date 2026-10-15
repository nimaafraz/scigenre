"""Code shared by every stage: the model input (title and abstract, cleaned), the OpenAlex client and profile
format, paper-level helpers (non-papers, h-index), and the figure style.
"""
import html
import os
import re
import time
import unicodedata
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# Model input: title and abstract, cleaned
# =====================================================================================================================
#
# Shared text preprocessing: the model input (title and abstract, publisher boilerplate removed), the non-paper
# patterns, and the title keyword rule used as a baseline.
# Used identically by training, classification and evaluation.

MAX_LENGTH = 384  # tokens; covers title + abstract for nearly all papers

# Title keyword rule (a baseline in rq1_evaluation.py baselines and analyses.py keyword-rules)
PAPER_TITLE_KEYWORDS = re.compile(
    r"\b(?:survey|surveys|review|reviews|overview|taxonomy|comparative study)\b",
    re.IGNORECASE,
)

# Publication types (OpenAlex type / Semantic Scholar publicationTypes) that are not papers
NON_PAPER_TYPES = re.compile(
    r"\b(?:book|book-chapter|booksection|editorial|erratum|paratext|peer-review|retraction|"
    r"dissertation|supplementary-materials|news|letterandcomments|lettersandcomments|"
    r"dataset|reference-entry|conference-abstract)\b",
    re.IGNORECASE,
)

# Front matter that is neither a survey nor research
NON_PAPER_TITLE = re.compile(
    r"\b(?:guest editorial|editorial|erratum|corrigendum|masthead|table of contents|"
    r"front cover|in memoriam|call for papers|"
    r"introduction to (?:the |this |our )?(?:special|invited|focus|featured|selected) (?:issue|section|top papers|papers|collection)|"
    r"guest editors?'? introduction)\b",
    re.IGNORECASE,
)


def clean_text(text) -> str:
    """Remove markup, LaTeX and formatting artifacts; collapse whitespace."""
    if text is None or (isinstance(text, float) and text != text):  # None / NaN
        return ""
    text = str(text)
    text = re.sub(r"<tex-math.*?</tex-math>", " ", text, flags=re.DOTALL)  # JATS math
    text = re.sub(r"<[^>]+>", " ", text)                                # HTML / JATS tags
    text = re.sub(r"\$[^$]*\$", " ", text)                             # inline LaTeX math
    text = re.sub(r"\\[a-zA-Z]+\*?(?:\{[^}]*\})?", " ", text)          # LaTeX commands
    text = re.sub(r"[{}\[\]]", " ", text)                              # brackets
    text = re.sub(r"^\s*abstract[:.\s-]*", "", text, flags=re.IGNORECASE)
    return re.sub(r"\s+", " ", text).strip()


# Publisher and indexing boilerplate that some abstracts carry (copyright lines, licences, journal citation
# lines, WIREs category lists, Annual Reviews notices). It names publishers and venues, so the model never sees it.
BOILERPLATE = re.compile(
    r"(?:©|\(c\)\s*(?:19|20)\d\d|\bcopyright\s*(?:©|\(c\)|(?:19|20)\d\d)|\ball rights reserved\b|"
    r"\bthis article is categorized under\b|\bexpected final online publication date\b|\bother information published in\b|"
    r"\blicen[cs]e:\s*https?://|\bpublished online in wiley\b|\bcreative commons attribution\b)",
    re.IGNORECASE,
)


# Headings of structured abstracts ("Background:", "Methods:", ...) at the start of a sentence.
SECTION_HEADER = re.compile(
    r"(?:^|(?<=[.!?] ))(?:background|objectives?|purpose|aims?|methods?|methodology|design(?:/methodology/approach)?|"
    r"approach|results?|findings|conclusions?|originality(?:/value)?|value|significance|context|introduction|"
    r"motivation|limitations|implications)\s*:\s*",
    re.IGNORECASE,
)


def strip_boilerplate(text: str) -> str:
    """Decode HTML entities, drop structured-abstract headings, then cut trailing boilerplate (marker in the last
    60% of the text) or drop the sentence with the marker."""
    text = SECTION_HEADER.sub("", html.unescape(text))
    text = re.sub(r"\bdoi:\s*10\.\S+", " ", text, flags=re.IGNORECASE)
    m = BOILERPLATE.search(text)
    while m:
        if m.start() >= 0.4 * len(text):
            text = text[:m.start()]
        else:
            s = text.rfind(". ", 0, m.start()) + 1
            e = text.find(". ", m.end()); e = len(text) if e < 0 else e + 1
            text = text[:s] + text[e:]
        m = BOILERPLATE.search(text)
    return re.sub(r"\s+", " ", text).strip()


def model_text(title, abstract) -> str:
    """Input of the final model: title and abstract only, boilerplate removed (never the venue)."""
    return f"{clean_text(title)}. {strip_boilerplate(clean_text(abstract))}"


# =====================================================================================================================
# OpenAlex API client
# =====================================================================================================================
#
# Small OpenAlex API client used by every step of the pipeline that queries OpenAlex.
# Set OPENALEX_API_KEY (free at https://openalex.org) in .env to avoid the small
# keyless daily budget; OPENALEX_MAILTO is optional.

load_dotenv(os.path.join(HERE, ".env"))

OPENALEX_URL = "https://api.openalex.org"
OPENALEX_MAILTO = os.getenv("OPENALEX_MAILTO", "").strip()
OPENALEX_API_KEY = os.getenv("OPENALEX_API_KEY", "").strip()


class OpenAlexBudgetError(RuntimeError):
    pass


def openalex_get(endpoint: str, params: dict, api_key: Optional[str] = None, retries: int = 6) -> dict:
    params = dict(params)
    key = (api_key or OPENALEX_API_KEY or "").strip()
    if key:
        params["api_key"] = key
    if OPENALEX_MAILTO:
        params["mailto"] = OPENALEX_MAILTO

    for attempt in range(retries):
        try:
            r = requests.get(f"{OPENALEX_URL}/{endpoint.lstrip('/')}", params=params, timeout=60)
            if r.status_code == 200:
                return r.json()
            if r.status_code == 429 and "budget" in r.text.lower():
                reset = r.headers.get("X-RateLimit-Reset")
                hours = f" (resets in about {int(reset) // 3600} h)" if reset and reset.isdigit() else ""
                raise OpenAlexBudgetError(
                    "OpenAlex daily budget exhausted" + hours + ". Get a free API key at "
                    "https://openalex.org and set OPENALEX_API_KEY in .env."
                )
            if r.status_code not in (429, 500, 502, 503, 504):
                raise RuntimeError(f"OpenAlex error {r.status_code}: {r.text[:300]}")
        except requests.RequestException as e:
            print(f"   ⚠️  Network error: {e}")
        wait = 2 ** attempt
        print(f"   ⏳ Retrying in {wait}s...")
        time.sleep(wait)
    raise RuntimeError("OpenAlex request failed after retries")


def openalex_get_all(endpoint: str, params: dict, api_key: Optional[str] = None) -> List[dict]:
    """Cursor-page through every result."""
    results, cursor = [], "*"
    while cursor:
        data = openalex_get(endpoint, {**params, "per-page": 200, "cursor": cursor}, api_key=api_key)
        page = data.get("results", [])
        results.extend(page)
        cursor = data.get("meta", {}).get("next_cursor")
        if not page:
            break
        time.sleep(0.1)
    return results


def rebuild_abstract(inverted: Optional[Dict[str, List[int]]]) -> str:
    if not inverted:
        return ""
    positions = [(pos, word) for word, idxs in inverted.items() for pos in idxs]
    return " ".join(word for _, word in sorted(positions))


def short_id(openalex_id: str) -> str:
    return re.sub(r"^https?://openalex.org/", "", openalex_id or "")


# =====================================================================================================================
# Publication profiles from OpenAlex
# =====================================================================================================================
#
# Publication profile format for OpenAlex authors (used by profiles.py fetch and every step that retrieves profiles).
#
# Columns: id, title, abstract, authors, citationCount, ReferenceCount, venue, year, type, doi. Versions of the
# same work (e.g. arXiv preprint and journal article) are merged by title, keeping the published version.

WORK_FIELDS = ("id,doi,title,publication_year,type,cited_by_count,referenced_works_count,"
               "abstract_inverted_index,primary_location,authorships")


def norm_title(title: str) -> str:
    return re.sub(r"[^a-z0-9]", "", str(title).lower())


def work_to_row(w: dict) -> dict:
    source = ((w.get("primary_location") or {}).get("source") or {})
    return {
        "id": short_id(w.get("id")),
        "title": w.get("title") or "",
        "abstract": rebuild_abstract(w.get("abstract_inverted_index")),
        "authors": "; ".join(a.get("author", {}).get("display_name", "") for a in w.get("authorships") or []),
        "citationCount": w.get("cited_by_count") or 0,
        "ReferenceCount": w.get("referenced_works_count") or 0,
        "venue": source.get("display_name") or "",
        "year": w.get("publication_year"),
        "type": w.get("type") or "",
        "doi": w.get("doi") or "",
    }


def merge_versions(df: pd.DataFrame) -> pd.DataFrame:
    """One row per title: prefer the non-preprint version, then the most cited."""
    df = df[df["title"].str.strip() != ""].copy()
    df["_key"] = df["title"].map(norm_title)
    df["_preprint"] = (df["type"] == "preprint").astype(int)
    df = df.sort_values(["_key", "_preprint", "citationCount"], ascending=[True, True, False])
    return df.drop_duplicates("_key").drop(columns=["_key", "_preprint"]).reset_index(drop=True)


# =====================================================================================================================
# Paper-level helpers: input columns, non-papers, h-index
# =====================================================================================================================
#
# Paper-level helpers shared by the pipeline: input normalization, the non-paper rule (books, editorials, errata, ...),
# and the h-index / i10-index calculation. SciGenre itself is applied in profiles.py (score_profile).

def prepare_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Map Semantic Scholar / dataset column names to Title, Abstract, ReferenceCount, Venue, Type."""
    out = pd.DataFrame(index=df.index)
    out["Title"] = df.get("Title", df.get("title", pd.Series("", index=df.index))).fillna("").astype(str)
    out["Abstract"] = df.get("Abstract", df.get("abstract", pd.Series("", index=df.index))).fillna("").astype(str)

    if "ReferenceCount" in df.columns:
        out["ReferenceCount"] = pd.to_numeric(df["ReferenceCount"], errors="coerce").fillna(0)
    elif "references" in df.columns:
        out["ReferenceCount"] = df["references"].fillna("").map(
            lambda s: len([r for r in str(s).split(";") if r.strip()]))
    else:
        out["ReferenceCount"] = 0

    out["Venue"] = df.get("Venue", df.get("venue", pd.Series("", index=df.index))).fillna("").astype(str)
    out["Type"] = df.get("Type", df.get("type", pd.Series("", index=df.index))).fillna("").astype(str)
    return out


def is_non_paper(frame: pd.DataFrame) -> np.ndarray:
    """Books, editorials, errata, ...: by publication type when known, otherwise by title."""
    by_title = frame["Title"].map(lambda t: bool(NON_PAPER_TITLE.search(clean_text(t))))
    by_type = frame["Type"].map(lambda t: bool(NON_PAPER_TYPES.search(t)))
    return (by_title | by_type).values


def calculate_indices(df: pd.DataFrame) -> Tuple[int, int]:
    """Calculate h-index and i10-index from citation counts."""
    citations = df["citationCount"].fillna(0).astype(int).sort_values(ascending=False).values
    h_index = int(sum(c >= (i + 1) for i, c in enumerate(citations)))
    i10_index = int(sum(c >= 10 for c in citations))
    return h_index, i10_index


# =====================================================================================================================
# Figure style
# =====================================================================================================================
#
# Shared figure style for the paper: palette, fonts, panel labels, and the marker that flags
# the ten most affected 2025 records (R1-R10) in every plot where they appear.
#
# R1-R10 are drawn as HCR-listed points (blue) with a diamond shape and dark outline; no
# individual labels are shown (re-identification, see the paper's ethics section).

FIG = os.path.join(STUDY, "figures")

# Reference palette (dataviz skill): colour follows the group in every figure. Checked with the dataviz
# validator for the two sets that appear together (HCR/matched/reference, HCR/ACM/IEEE/reference): colour-vision
# deficiency separation >= 9, normal vision >= 16. The reference gray is a deliberate neutral for the baseline group.
LISTED = "#2a78d6"      # HCR-listed researchers
MATCHED = "#eb6834"     # matched non-listed researchers
REFERENCE = "#7a7974"   # random non-listed reference sample
ACM, IEEE = "#1baf7a", "#eda100"
INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
FIELD = INK             # other studies' data: always a dashed ink line
SERIES, SERIES_2, BEFORE = LISTED, MATCHED, REFERENCE
NAMES = {"listed": "HCR-listed", "matched": "Matched peers", "reference": "Random sample",
         "acm": "ACM Fellows", "ieee": "IEEE Fellows"}
COLORS = {"listed": LISTED, "matched": MATCHED, "reference": REFERENCE, "acm": ACM, "ieee": IEEE}
# One encoding for "with surveys" vs "without surveys" of the same group: dashed or hollow = with surveys,
# solid or filled = without surveys, both in the group's colour.
LW, MS = 1.5, 3.5       # data lines and their markers
POINT = dict(s=7, alpha=0.55, linewidths=0)   # one dot per researcher

# FIG_PROFILE=springer: single-column Springer page (text width 5.15 in) instead of the IEEE two-column page
SPRINGER = os.environ.get("FIG_PROFILE", "") == "springer"
WIDE = 5.15 if SPRINGER else 7.16

R_LABEL = "R1–R10 (ten most affected records)"
R_MARK = dict(marker="D", s=15, facecolor=LISTED, edgecolor=INK, linewidths=0.6, zorder=6)

plt.rcParams.update({
    "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"], "mathtext.fontset": "stix",
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 6.5,
    "xtick.labelsize": 7, "ytick.labelsize": 7,
    "axes.edgecolor": INK_2, "axes.labelcolor": INK, "xtick.color": INK_2, "ytick.color": INK_2,
    "axes.linewidth": 0.6, "legend.frameon": False, "legend.handletextpad": 0.4, "legend.handlelength": 1.8,
    "lines.linewidth": LW, "lines.markersize": MS, "pdf.fonttype": 42,
})


def style(ax, grid="y"):
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    if grid == "both":
        ax.grid(True, color=GRID, linewidth=0.6)
    elif grid:
        getattr(ax, f"{grid}axis").grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def panel(ax, letter, title=None):
    """Bold panel letter with an optional short descriptor, above the top-left corner."""
    ax.annotate(f"({letter})", xy=(0, 1), xycoords="axes fraction", xytext=(0, 5), textcoords="offset points",
                fontsize=8, fontweight="bold", color=INK, ha="left", va="bottom")
    if title:
        ax.annotate(title, xy=(0, 1), xycoords="axes fraction", xytext=(15, 5), textcoords="offset points",
                    fontsize=7.5, color=INK_2, ha="left", va="bottom")


def save(fig, name):
    os.makedirs(FIG, exist_ok=True)
    fig.savefig(os.path.join(FIG, f"{name}.pdf"))
    fig.savefig(os.path.join(FIG, f"{name}.png"), dpi=300)
    plt.close(fig)


def flag(ax, x, y, label=True, **kw):
    """Draw R1-R10 points with the shared diamond marker."""
    style_ = {**R_MARK, **kw}
    return ax.scatter(x, y, label=R_LABEL if label else None, **style_)


def r_handle():
    return plt.Line2D([], [], linestyle="none", marker="D", markersize=4, markerfacecolor=LISTED,
                      markeredgecolor=INK, markeredgewidth=0.6, label=R_LABEL)


def _norm(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


def person_key(first, last):
    first = re.sub(r"[‐-―]", "-", str(first)).strip()
    return _norm(last) + "|" + _norm(re.split(r"[\s.\-]+", first)[0])


def r_sets():
    """R1-R10 as HCR names, OpenAlex IDs, and person keys (internal use only)."""
    top = pd.read_csv(os.path.join(STUDY, "v2_top10_key_INTERNAL.csv"))
    names = set(top["key"])
    sd = pd.read_csv(os.path.join(STUDY, "outliers", "survey_dependence_INTERNAL.csv"))
    ids = set(sd.loc[(sd.group == "listed") & sd["key"].isin(names), "id"])
    keys = {person_key(n.split(",", 1)[1], n.split(",", 1)[0]) for n in names}
    return {"names": names, "ids": ids, "keys": keys}


def boxes(ax, data, positions, colors, flags=None, width=0.5, fmt="{:.1f}%", labels=True, rng=None):
    """Neutral boxes (interquartile range, median, whiskers to the Tukey fences) over one dot per researcher in the
    group's colour. flags (boolean arrays) mark R1-R10 with the shared diamond. The median is written beside the box."""
    rng = rng if rng is not None else np.random.default_rng(42)
    flags = flags if flags is not None else [None] * len(data)
    for x0, v, col, f in zip(positions, data, colors, flags):
        v = np.asarray(v, dtype=float); f = np.zeros(len(v), bool) if f is None else np.asarray(f, bool)
        ok = ~np.isnan(v); v, f = v[ok], f[ok]
        ax.boxplot([v], positions=[x0], widths=width, showfliers=False, whis=1.5, patch_artist=True, zorder=2,
                   boxprops=dict(facecolor="none", edgecolor=INK_2, linewidth=0.7), whiskerprops=dict(color=INK_2, linewidth=0.7),
                   capprops=dict(color=INK_2, linewidth=0.7), medianprops=dict(color=INK, linewidth=1.3))
        xj = x0 + rng.uniform(-width * 0.34, width * 0.34, len(v))
        ax.scatter(xj[~f], v[~f], color=col, zorder=3, **POINT)
        if f.any():
            flag(ax, xj[f], v[f], label=False)
        if labels and len(v):
            ax.text(x0 + width * 0.56, np.median(v), fmt.format(np.median(v)), va="center", ha="left", fontsize=6.5,
                    color=INK, zorder=7, bbox=dict(boxstyle="square,pad=0.1", facecolor="white", edgecolor="none"))


def end_label(ax, x, y, text, dy=0.0, dx=4):
    """Direct label at the end of a line, in ink (the line's colour carries the identity)."""
    ax.annotate(text, (x, y), xytext=(dx, dy), textcoords="offset points", fontsize=6.5, color=INK, va="center", ha="left")


def jitter(n, width, rng):
    return rng.uniform(-width, width, n)
