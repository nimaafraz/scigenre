"""Stage 5: the comparison groups (matched peers, the random sample of the Stanford-Elsevier database, the
one-to-one peers of a sensitivity check, the boundary samples, and the IEEE and ACM Fellows), each resolved,
downloaded and classified with the rules used for the HCR-listed researchers.

    python comparison_groups.py controls select|classify [--missing]|exclude              one-to-one peers (sensitivity check)
    python comparison_groups.py reference select|classify|topics|exclude                  random sample of the Stanford-Elsevier database
    python comparison_groups.py any-list                                                  comparison researchers on any HCR list
    python comparison_groups.py peers select|classify|analyze                             matched peers
    python comparison_groups.py boundary select|merge|classify|analyze                    boundary samples (SUBFIELD_NAME)
    python comparison_groups.py fellows [load|match|academic|fetch|classify|report|all]   IEEE and ACM Fellows
"""
import glob
import json
import os
import re
import subprocess
import sys
import time
import unicodedata

import numpy as np
import pandas as pd
import requests

import common
import profiles
from common import merge_versions, openalex_get, openalex_get_all, short_id, WORK_FIELDS

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# One-to-one citation-matched researchers (sensitivity check)
# =====================================================================================================================

def controls():
    """Matched comparison group for the HCR study: highly cited researchers who are
    NOT on the Clarivate HCR (Computer Science) list.

    For each Scopus-consistent HCR-listed researcher with an unambiguous
    els.csv match, pick one non-listed researcher from els.csv (top-cited scientists,
    Aug 2025) with
        - the same Science-Metrix subfield (sm-subfield-1),
        - first publication year within +/- MAX_FIRSTYR_GAP,
        - non-self citations 1996-2024 within a factor CITATION_CALIPER,
        - recent activity (last publication year >= MIN_LASTYR),
        - not on any 2025 Highly Cited Researchers list (all categories, by name),
    nearest on log citations, then on first year, without replacement. HCR-listed researchers are
    matched in random order. Up to MAX_TRIES candidates are tried in that order until one resolves
    to a plausible OpenAlex profile: same resolver and Scopus-consistency check as the HCR-listed
    researchers, and at least half of the profile's OpenAlex topics in computer science or engineering.

    Question: at the same field, career stage and total citations, do HCR-listed researchers
    differ from non-listed researchers in survey dependence, and does the listed researchers'
    advantage in top-1% papers survive excluding surveys?

    Steps:
        python comparison_groups.py controls select              # choose + resolve + download
        python comparison_groups.py controls classify            # classify with the final model
        python comparison_groups.py controls exclude             # drop matches on a 2025 list (comparison_groups.py any-list), then select again
        python comparison_groups.py controls classify --missing  # classify the new matches only

    Output: hcr_study/controls/"""
    OUT = os.path.join(STUDY, "controls")
    PROFILE_DIR = os.path.join(OUT, "profiles")
    CLASSIFIED_DIR = os.path.join(OUT, "classified")

    ELS_FILE = os.path.join(HERE, "inputs", "stanford_elsevier", "els.csv")
    MEMBER_ELS = os.path.join(STUDY, "BTAS_HCR_Computer_Science.csv")   # HCR-listed researchers' els.csv rows (profiles.py els-match)
    MEMBERS = os.path.join(STUDY, "per_member.csv")
    HCR_2025_ALL = os.path.join(HERE, "inputs", "hcr_lists", "2025_HCR.xlsx")   # all 2025 categories

    MAX_FIRSTYR_GAP = 3
    CITATION_CALIPER = 1.25     # control citations within [1/1.25, 1.25] x the HCR-listed researcher's
    MIN_LASTYR = 2022
    MAX_TRIES = 6               # candidates tried per HCR-listed researcher before giving up
    SEED = 42

    # Same plausibility rule as the HCR-listed researchers (profiles.py classify), checked on the author record.
    CITE_RATIO = (0.6, 3.0)
    WORK_RATIO = (0.4, 3.0)


    def ascii_lower(s):
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


    def name_key(name):
        """'Surname, Given Names' -> 'surname|givennames' (letters only)."""
        if "," not in str(name):
            return re.sub(r"[^a-z]", "", ascii_lower(name))
        last, first = str(name).split(",", 1)
        first = re.sub(r"\(.*?\)", "", first)
        return re.sub(r"[^a-z]", "", ascii_lower(last)) + "|" + re.sub(r"[^a-z]", "", ascii_lower(first))


    def first_key(name):
        """'Surname, Given Names' -> 'surname|firstgivenname' (letters only)."""
        if "," not in str(name):
            return re.sub(r"[^a-z]", "", ascii_lower(name))
        last, first = str(name).split(",", 1)
        tok = [t for t in re.split(r"[\s.\-]+", re.sub(r"\(.*?\)", " ", first).strip()) if t]
        return re.sub(r"[^a-z]", "", ascii_lower(last)) + "|" + (re.sub(r"[^a-z]", "", ascii_lower(tok[0])) if tok else "")


    # ============================================================
    # SELECT + RESOLVE + DOWNLOAD
    # ============================================================

    def select():

        os.makedirs(PROFILE_DIR, exist_ok=True)

        members = pd.read_csv(MEMBERS)
        members = members[members["Consistent with Scopus"]]
        m_els = pd.read_csv(MEMBER_ELS)
        m_els = m_els[m_els["Match status"] == "ok"][["HCR name", "authfull", "inst_name", "sm-subfield-1",
                                                         "firstyr", "np6024", "nc9624 (ns)"]]
        members = members[["HCR name"]].merge(m_els, on="HCR name")
        print(f"HCR-listed researchers to match: {len(members)}")

        els = pd.read_csv(ELS_FILE, thousands=",", low_memory=False,
                          usecols=["authfull", "inst_name", "cntry", "np6024", "firstyr", "lastyr",
                                   "nc9624 (ns)", "h24 (ns)", "sm-subfield-1"])
        els["key"] = els["authfull"].map(name_key)

        # Exclude HCR-listed researchers (their els rows and any same-name row, to avoid namesakes).
        hcr = profiles.hcr_2025_cs()["Full Name"].str.strip()
        hcr_keys = set(hcr.map(name_key)) | set(m_els["authfull"].map(name_key))
        all25 = pd.read_excel(HCR_2025_ALL)
        hcr_keys |= {name_key(f"{l}, {f}") for l, f in zip(all25["Last Name"], all25["First Name"])}
        # also by surname and first given name, so that middle names or initials ("Jane Q.") do not hide a listing
        first_keys = {first_key(f"{l}, {f}") for l, f in zip(all25["Last Name"], all25["First Name"])} | set(hcr.map(first_key))
        pool = els[~els["key"].isin(hcr_keys) & ~els["authfull"].map(first_key).isin(first_keys) & (els["lastyr"] >= MIN_LASTYR)].copy()
        pool["log_c"] = np.log(pool["nc9624 (ns)"].clip(lower=1))

        # OpenAlex authors of 2025-listed researchers (catches names that Stanford-Elsevier spells or orders differently)
        hist = pd.read_csv(os.path.join(STUDY, "history", "per_researcher.csv"))
        listed25 = set(pd.read_csv(MEMBERS)["OpenAlex ID"].dropna()) | {a for a, v in zip(hist["OpenAlex ID"], hist["Years listed"]) if "2025" in str(v)}
        rng = np.random.default_rng(SEED)
        order = rng.permutation(len(members))
        used, rows = set(), []

        # Incremental: keep earlier matches, match only HCR-listed researchers without a row yet.
        prev_path = os.path.join(OUT, "matches.csv")
        if os.path.exists(prev_path):
            prev = pd.read_csv(prev_path)
            rows = prev.to_dict("records")
            done = set(prev["HCR name"])
            taken = set(zip(prev["Control authfull"].dropna(), prev["Control institution"].dropna()))
            used = set(pool.index[[(a, b) in taken for a, b in zip(pool["authfull"], pool["inst_name"])]])
            order = [k for k in order if members.iloc[k]["HCR name"] not in done]
            print(f"Keeping {len(prev)} earlier matches; matching {len(order)} more HCR-listed researchers")

        for k in order:
            m = members.iloc[k]
            cand = pool[(pool["sm-subfield-1"] == m["sm-subfield-1"])
                        & ((pool["firstyr"] - m["firstyr"]).abs() <= MAX_FIRSTYR_GAP)]
            cand = cand.assign(gap=(cand["log_c"] - np.log(m["nc9624 (ns)"])).abs(),
                               ygap=(cand["firstyr"] - m["firstyr"]).abs())
            cand = cand[(cand["gap"] <= np.log(CITATION_CALIPER)) & ~cand.index.isin(used)]
            cand = cand.sort_values(["gap", "ygap"]).head(MAX_TRIES)

            chosen = None
            for idx, c in cand.iterrows():
                row = {"Full Name": c["authfull"], "Primary Affiliation": c["inst_name"],
                       "Secondary Affiliations": ""}
                try:
                    a, status, inst, _ = profiles.resolve_author(pd.Series(row))
                except Exception as e:     # malformed names etc.
                    print(f"   resolve failed for {c['authfull']}: {e}")
                    continue
                if a is None or status.startswith(("weak", "not")):
                    continue
                if short_id(a["id"]) in listed25:
                    continue
                cr = a.get("cited_by_count", 0) / max(c["nc9624 (ns)"], 1)
                wr = a.get("works_count", 0) / max(c["np6024"], 1)
                if not (CITE_RATIO[0] <= cr <= CITE_RATIO[1] and WORK_RATIO[0] <= wr <= WORK_RATIO[1]):
                    continue
                t = openalex_get(f"authors/{short_id(a['id'])}", {"select": "topics"}).get("topics") or []
                n_t = sum(x["count"] for x in t) or 1
                if sum(x["count"] for x in t if x["field"]["display_name"] in ("Computer Science", "Engineering")) / n_t < 0.5:
                    continue                     # profile of a different person or a merged profile
                chosen = (idx, c, a, status, cr, wr)
                break

            if chosen is None:
                rows.append({"HCR name": m["HCR name"], "Status": "no control found"})
                print(f"  {m['HCR name'][:28]:28s} -> no control")
                continue

            idx, c, a, status, cr, wr = chosen
            used.add(idx)
            aid = short_id(a["id"])
            prof_path = os.path.join(PROFILE_DIR, f"{aid}.csv")
            if not os.path.exists(prof_path):
                works = openalex_get_all("works", {"filter": f"author.id:{aid}",
                                                   "select": WORK_FIELDS + profiles.EXTRA_WORK_FIELDS})
                merge_versions(pd.DataFrame([profiles.work_row(w) for w in works])).to_csv(prof_path, index=False)
            rows.append({
                "HCR name": m["HCR name"], "Status": status,
                "Member subfield": m["sm-subfield-1"], "Member firstyr": m["firstyr"],   # of the HCR-listed researcher
                "Member citations (Scopus ns)": m["nc9624 (ns)"],
                "Control authfull": c["authfull"], "Control institution": c["inst_name"],
                "Control country": c["cntry"], "Control firstyr": c["firstyr"],
                "Control citations (Scopus ns)": c["nc9624 (ns)"], "Control papers (Scopus)": c["np6024"],
                "Control OpenAlex ID": aid, "Control OpenAlex name": a["display_name"],
                "OpenAlex/Scopus citations (author record)": cr, "OpenAlex/Scopus works (author record)": wr,
            })
            print(f"  {m['HCR name'][:28]:28s} -> {c['authfull'][:28]:28s} ({aid}, cites x{cr:.2f})")

        sel = pd.DataFrame(rows)
        sel.to_csv(os.path.join(OUT, "matches.csv"), index=False)
        print(sel["Status"].value_counts().to_string())


    # ============================================================
    # CLASSIFY
    # ============================================================

    def classify():

        os.makedirs(CLASSIFIED_DIR, exist_ok=True)
        sel = pd.read_csv(os.path.join(OUT, "matches.csv"))
        sel = sel[sel["Control OpenAlex ID"].notna()]
        rows = []
        out_path = os.path.join(OUT, "per_control.csv")
        if len(sys.argv) > 2 and sys.argv[2] == "--missing" and os.path.exists(out_path):
            prev = pd.read_csv(out_path)
            rows = prev.to_dict("records")
            sel = sel[~sel["Control OpenAlex ID"].isin(prev["Control OpenAlex ID"])]
            print(f"Keeping {len(prev)} classified controls; classifying {len(sel)} more")
        for i, r in enumerate(sel.itertuples(index=False), 1):
            aid = r[sel.columns.get_loc("Control OpenAlex ID")]
            df = pd.read_csv(os.path.join(PROFILE_DIR, f"{aid}.csv"))
            df["citationCount"] = df["citationCount"].fillna(0).astype(int)
            papers, _ = profiles.classify_papers(df)
            papers.to_csv(os.path.join(CLASSIFIED_DIR, f"{aid}.csv"), index=False)
            m, survey, cites = profiles.author_metrics(df, papers["Category"].values)
            scopus = r[sel.columns.get_loc("Control citations (Scopus ns)")]
            cite_ratio = m["Citations"] / scopus if scopus else np.nan
            paper_ratio = m["Papers"] / r[sel.columns.get_loc("Control papers (Scopus)")]
            rows.append({
                "HCR name (matched member)": r[sel.columns.get_loc("HCR name")],
                "Control OpenAlex ID": aid,
                "Control authfull": r[sel.columns.get_loc("Control authfull")],
                "OpenAlex/Scopus citations": cite_ratio, "OpenAlex/Scopus papers": paper_ratio,
                "Consistent with Scopus": bool(0.6 <= cite_ratio <= 3.0 and 0.4 <= paper_ratio <= 3.0),
                **m,
            })
            print(f"  [{i:3d}/{len(sel)}] control for {rows[-1]['HCR name (matched member)'][:26]:26s} "
                  f"surveys {m['Survey % of citations']:5.1f}% of cites, top1 {m['Top-1% papers 2014-2024']}")
        pd.DataFrame(rows).to_csv(out_path, index=False)
        print(f"Written to {OUT}/per_control.csv")


    def exclude():
        """Drop matches who are on a 2025 list (comparison_groups.py any-list) and the 'no control found' rows, so that select
    finds new matches for those HCR-listed researchers from the current pool."""
        import shutil
        lst = pd.read_csv(os.path.join(STUDY, "hcr_any_list.csv"))
        bad = set(lst[(lst.group == "matched") & lst["on a 2025 list"]]["OpenAlex ID"])
        mt = pd.read_csv(os.path.join(OUT, "matches.csv"))
        drop = mt["Control OpenAlex ID"].isin(bad) | mt["Control OpenAlex ID"].isna()
        log = mt[mt["Control OpenAlex ID"].isin(bad)].assign(reason="on a 2025 Highly Cited Researchers list")
        log.to_csv(os.path.join(OUT, "excluded.csv"), index=False)
        mt[~drop].to_csv(os.path.join(OUT, "matches.csv"), index=False)
        pc = pd.read_csv(os.path.join(OUT, "per_control.csv"))
        pc[~pc["Control OpenAlex ID"].isin(bad)].to_csv(os.path.join(OUT, "per_control.csv"), index=False)
        for sub in ("classified", "profiles"):
            os.makedirs(os.path.join(OUT, "excluded", sub), exist_ok=True)
            for a in bad:
                f = os.path.join(OUT, sub, f"{a}.csv")
                if os.path.exists(f):
                    shutil.move(f, os.path.join(OUT, "excluded", sub, f"{a}.csv"))
        print(f"removed {len(log)} matches on a 2025 list and {int(mt['Control OpenAlex ID'].isna().sum())} 'no control found' rows")


    step = sys.argv[1] if len(sys.argv) > 1 else "select"
    {"select": select, "classify": classify, "exclude": exclude}[step]()


# =====================================================================================================================
# Random sample of the Stanford-Elsevier database
# =====================================================================================================================

def reference():
    """Reference population for survey-dependence outliers: a random sample of
    top-cited computer-science researchers who are not listed in the HCR.

    Sampling frame: els.csv (Stanford-Elsevier top-cited scientists, Aug 2025) in the
    subfields of the listed researchers, in proportion to them; 5,000-100,000 Scopus
    non-self citations; first publication 1980-2016; last publication 2022 or later;
    not on the HCR Computer Science list (by name). Each sampled researcher is
    resolved in OpenAlex with the same rules and plausibility checks as the listed
    researchers, downloaded, and classified with the final model.

        python comparison_groups.py reference select
        python comparison_groups.py reference classify
        python comparison_groups.py reference topics     # OpenAlex topics of the Scopus-consistent researchers of every group
        python comparison_groups.py reference exclude    # after comparison_groups.py any-list and topics

    The exclude step removes researchers who are on any 2025 Highly Cited Researchers list (any category, confirmed by
    affiliation, comparison_groups.py any-list) and profiles whose OpenAlex topics lie mostly outside computer science and
    engineering (less than half, hcr_study/author_topic_check.csv), which flags profiles of a different person.

    Output: hcr_study/reference/"""
    OUT = os.path.join(HERE, "hcr_study", "reference")
    PROFILE_DIR = os.path.join(OUT, "profiles")
    CLASSIFIED_DIR = os.path.join(OUT, "classified")

    STRATA = {                       # subfield -> sample size (about the listed researchers' mix)
        "Networking & Telecommunications": 330,
        "Artificial Intelligence & Image Processing": 135,
        "Industrial Engineering & Automation": 25,
        "Distributed Computing": 10,
    }
    CITES = (5_000, 100_000)
    FIRSTYR = (1980, 2016)
    MIN_LASTYR = 2022
    SEED = 7
    CITE_RATIO = (0.6, 3.0)          # OpenAlex / Scopus, author record and profile
    WORK_RATIO = (0.4, 3.0)


    def ascii_lower(s):
        return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


    def name_key(name):
        if "," not in str(name):
            return re.sub(r"[^a-z]", "", ascii_lower(name))
        last, first = str(name).split(",", 1)
        first = re.sub(r"\(.*?\)", "", first)
        return re.sub(r"[^a-z]", "", ascii_lower(last)) + "|" + re.sub(r"[^a-z]", "", ascii_lower(first))


    def select():

        os.makedirs(PROFILE_DIR, exist_ok=True)
        els = pd.read_csv(os.path.join(HERE, "inputs", "stanford_elsevier", "els.csv"), thousands=",", low_memory=False,
                          usecols=["authfull", "inst_name", "cntry", "np6024", "firstyr", "lastyr",
                                   "nc9624 (ns)", "sm-subfield-1"])
        hcr = profiles.hcr_2025_cs()["Full Name"].str.strip()
        listed_keys = set(hcr.map(name_key))
        controls = pd.read_csv(os.path.join(HERE, "hcr_study", "controls", "matches.csv"))
        control_keys = set(controls["Control authfull"].dropna().map(name_key))

        frame = els[els["sm-subfield-1"].isin(STRATA)
                    & els["nc9624 (ns)"].between(*CITES)
                    & els["firstyr"].between(*FIRSTYR)
                    & (els["lastyr"] >= MIN_LASTYR)].copy()
        frame["key"] = frame["authfull"].map(name_key)
        frame = frame[~frame["key"].isin(listed_keys | control_keys)]
        print(f"Sampling frame: {len(frame):,} researchers")

        sample = pd.concat([frame[frame["sm-subfield-1"] == s].sample(n, random_state=SEED) for s, n in STRATA.items()])

        # Resume: keep earlier results, retry only candidates that failed with an error (e.g. API budget).
        rows, done = [], set()
        prev_path = os.path.join(OUT, "sample.csv")
        if os.path.exists(prev_path):
            prev = pd.read_csv(prev_path)
            ok = prev[~prev["status"].astype(str).str.startswith("error")]
            rows = ok.to_dict("records")
            done = set(zip(ok["authfull"], ok["inst_name"].fillna("")))
            print(f"Resuming: {len(ok)} kept, {len(prev) - len(ok)} to retry")
        sample = sample[[(a, b if isinstance(b, str) else "") not in done for a, b in zip(sample["authfull"], sample["inst_name"])]]
        for i, (_, c) in enumerate(sample.iterrows(), 1):
            row = {"authfull": c["authfull"], "inst_name": c["inst_name"], "cntry": c["cntry"],
                   "subfield": c["sm-subfield-1"], "firstyr": c["firstyr"], "np6024": c["np6024"],
                   "scopus_cites": c["nc9624 (ns)"]}
            try:
                a, status, _, _ = profiles.resolve_author(pd.Series({"Full Name": c["authfull"], "Primary Affiliation": c["inst_name"],
                                                       "Secondary Affiliations": ""}))
            except Exception as e:
                rows.append({**row, "status": f"error: {e}"[:80]})
                continue
            if a is None or status.startswith(("weak", "not")):
                rows.append({**row, "status": status})
                continue
            cr = a.get("cited_by_count", 0) / max(c["nc9624 (ns)"], 1)
            wr = a.get("works_count", 0) / max(c["np6024"], 1)
            if not (CITE_RATIO[0] <= cr <= CITE_RATIO[1] and WORK_RATIO[0] <= wr <= WORK_RATIO[1]):
                rows.append({**row, "status": "implausible author record"})
                continue
            aid = short_id(a["id"])
            path = os.path.join(PROFILE_DIR, f"{aid}.csv")
            if not os.path.exists(path):
                works = openalex_get_all("works", {"filter": f"author.id:{aid}",
                                                   "select": WORK_FIELDS + profiles.EXTRA_WORK_FIELDS})
                merge_versions(pd.DataFrame([profiles.work_row(w) for w in works])).to_csv(path, index=False)
            rows.append({**row, "status": status, "openalex_id": aid, "openalex_name": a["display_name"]})
            if i % 25 == 0:
                print(f"  {i}/{len(sample)} resolved; usable so far: {sum('openalex_id' in r for r in rows)}")
        sel = pd.DataFrame(rows)
        sel.to_csv(os.path.join(OUT, "sample.csv"), index=False)
        print(sel["status"].value_counts().to_string())


    def classify():

        os.makedirs(CLASSIFIED_DIR, exist_ok=True)
        sel = pd.read_csv(os.path.join(OUT, "sample.csv"))
        sel = sel[sel["openalex_id"].notna()]
        rows = []
        for i, (_, r) in enumerate(sel.iterrows(), 1):
            path = os.path.join(CLASSIFIED_DIR, f"{r['openalex_id']}.csv")
            df = pd.read_csv(os.path.join(PROFILE_DIR, f"{r['openalex_id']}.csv"))
            df["citationCount"] = df["citationCount"].fillna(0).astype(int)
            if os.path.exists(path):
                papers = pd.read_csv(path)
            else:
                papers, _ = profiles.classify_papers(df)
                papers.to_csv(path, index=False)
            m, _, _ = profiles.author_metrics(df, papers["Category"].values)
            cr = m["Citations"] / max(r["scopus_cites"], 1)
            pr = m["Papers"] / max(r["np6024"], 1)
            rows.append({**r.to_dict(), "OpenAlex/Scopus citations": cr, "OpenAlex/Scopus papers": pr,
                         "Consistent with Scopus": bool(CITE_RATIO[0] <= cr <= CITE_RATIO[1] and WORK_RATIO[0] <= pr <= WORK_RATIO[1]),
                         **m})
            if i % 25 == 0:
                print(f"  classified {i}/{len(sel)}")
                pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_reference.csv"), index=False)
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_reference.csv"), index=False)
        print(f"Written {len(rows)} to {OUT}/per_reference.csv")


    def topics():
        """Share of each researcher's OpenAlex topics (by works) in the Computer Science and Engineering fields, for the
    Scopus-consistent researchers of every group (the rule comparison_groups.py controls applies when it chooses matched peers).
    Output: hcr_study/author_topic_check.csv (group, id, cs_eng_share, top topic, main field); author records are
    cached in hcr_study/author_topics.json."""
        import json
        from collections import Counter
        st = os.path.join(HERE, "hcr_study")
        pm = pd.read_csv(os.path.join(st, "per_member.csv"))
        pc = pd.read_csv(os.path.join(st, "controls", "per_control.csv"))
        pr = pd.read_csv(os.path.join(OUT, "per_reference.csv"))
        people = ([("listed", a) for a in pm.loc[pm["Consistent with Scopus"], "OpenAlex ID"]] +
                  [("matched", a) for a in pc.loc[pc["Consistent with Scopus"], "Control OpenAlex ID"]] +
                  [("reference", a) for a in pr.loc[pr["Consistent with Scopus"], "openalex_id"]])
        cache_path = os.path.join(st, "author_topics.json")
        cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        rows = []
        for k, (group, aid) in enumerate(people, 1):
            if aid not in cache:
                cache[aid] = openalex_get(f"authors/{aid}", {"select": "id,display_name,topics"})
                if k % 50 == 0:
                    json.dump(cache, open(cache_path, "w"))
            t = cache[aid].get("topics") or []
            n = sum(x["count"] for x in t) or 1
            fields = Counter(x["field"]["display_name"] for x in t)
            rows.append({"group": group, "id": aid,
                         "cs_eng_share": sum(x["count"] for x in t if x["field"]["display_name"] in ("Computer Science", "Engineering")) / n,
                         "top topic": t[0]["display_name"] if t else "", "main field": fields.most_common(1)[0][0] if fields else ""})
        json.dump(cache, open(cache_path, "w"))
        out = pd.DataFrame(rows)
        out.to_csv(os.path.join(st, "author_topic_check.csv"), index=False)
        print(out.groupby("group")["cs_eng_share"].apply(lambda x: int((x < 0.5).sum())).rename("below 0.5").to_string())


    def exclude():
        import shutil
        lst = pd.read_csv(os.path.join(HERE, "hcr_study", "hcr_any_list.csv"))
        top = pd.read_csv(os.path.join(HERE, "hcr_study", "author_topic_check.csv"))
        why = {a: "on a 2025 Highly Cited Researchers list" for a in lst[(lst.group == "reference") & lst["on a 2025 list"]]["OpenAlex ID"]}
        for a in top[(top.group == "reference") & (top.cs_eng_share < 0.5)]["id"]:
            why.setdefault(a, "topics mostly outside computer science and engineering")
        per = pd.read_csv(os.path.join(OUT, "per_reference.csv"))
        gone = per[per["openalex_id"].isin(why)].assign(reason=lambda d: d["openalex_id"].map(why))
        prev = os.path.join(OUT, "excluded.csv")
        if os.path.exists(prev):
            gone = pd.concat([pd.read_csv(prev), gone]).drop_duplicates("openalex_id")
        gone.to_csv(prev, index=False)
        per[~per["openalex_id"].isin(why)].to_csv(os.path.join(OUT, "per_reference.csv"), index=False)
        smp = pd.read_csv(os.path.join(OUT, "sample.csv"))
        hit = smp["openalex_id"].isin(why)
        smp.loc[hit, "status"] = "excluded: " + smp.loc[hit, "openalex_id"].map(why)
        smp.loc[hit, "openalex_id"] = np.nan
        smp.to_csv(os.path.join(OUT, "sample.csv"), index=False)
        for sub in ("classified", "profiles"):
            os.makedirs(os.path.join(OUT, "excluded", sub), exist_ok=True)
            for a in why:
                f = os.path.join(OUT, sub, f"{a}.csv")
                if os.path.exists(f):
                    shutil.move(f, os.path.join(OUT, "excluded", sub, f"{a}.csv"))
        print(gone["reason"].value_counts().to_string(), f"\nremaining: {(~per['openalex_id'].isin(why)).sum()}")


    {"select": select, "classify": classify, "topics": topics, "exclude": exclude}[sys.argv[1] if len(sys.argv) > 1 else "select"]()


# =====================================================================================================================
# Comparison researchers on any HCR list
# =====================================================================================================================

def any_list():
    """Researchers in the comparison groups (matched peers, random sample) who are on a Highly Cited Researchers list
    in ANY category (Clarivate's lists 2014-2025, all 21-22 categories). Matched peers and the random sample must not
    be on any 2025 list. Matched peers on any list from 2014 to 2025 are flagged for a sensitivity check that excludes them.

    Candidates: same surname and same first given name, or same full given names (letters only). A candidate is the
    same person when an institution of the HCR entry (primary or secondary affiliation) appears in the candidate's
    OpenAlex affiliation history, either as one of the top OpenAlex institution-search hits for the affiliation or by
    equal names, or one name containing the other, after removing parentheses and the country. Candidates whose affiliations do not connect are reported
    as namesakes and kept. In addition, every person on the Computer Science lists 2014-2025 was resolved to an OpenAlex
    author (hcr_study/history/per_researcher.csv, hcr_study/per_member.csv), so a comparison researcher whose OpenAlex ID
    is one of those authors takes that person's listing years, whatever the spelling or order of the name.

        python comparison_groups.py any-list
    Output: hcr_study/hcr_any_list_hits.csv (every name hit with its evidence)
            hcr_study/hcr_any_list.csv (one row per researcher: group, OpenAlex ID, on a 2025 list, years listed)"""
    ST = os.path.join(HERE, "hcr_study")
    LISTS = os.path.join(HERE, "inputs", "hcr_lists")      # Clarivate's lists, <year>_HCR.xlsx
    YEARS = range(2014, 2026)


    def norm(s):
        return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


    def keys(last, first):
        first = re.sub(r"\(.*?\)", " ", str(first)).strip()
        tok = [t for t in re.split(r"[\s.\-]+", first) if t]
        return {norm(last) + "|" + norm(tok[0]) if tok else norm(last), norm(last) + "||" + norm(first)}


    def split_name(full):
        return str(full).split(",", 1) if "," in str(full) else (full, "")


    def clean_affil(part):
        """'Massachusetts Institute of Technology (MIT), United States' -> 'Massachusetts Institute of Technology'."""
        part = re.sub(r"\(.*?\)", " ", str(part))
        part = part.rsplit(",", 1)[0] if "," in part else part
        return re.sub(r"\s+", " ", part).strip()


    _inst = {}
    def institution_ids(affil):
        out, names = set(), set()
        for part in [a for a in re.split(r";", str(affil)) if a.strip() and a.strip().lower() != "nan"]:
            name = clean_affil(part)
            if not name:
                continue
            names.add(norm(name))
            if name not in _inst:
                res = openalex_get("institutions", {"search": name, "per-page": 3}).get("results", [])
                _inst[name] = {short_id(r["id"]) for r in res}
            out |= _inst[name]
        return out, names


    _auth = {}
    def author_institutions(aid):
        if aid not in _auth:
            a = openalex_get(f"authors/{aid}", {})
            insts = [x["institution"] for x in a.get("affiliations", []) if x.get("institution")] + list(a.get("last_known_institutions") or [])
            _auth[aid] = ({short_id(x["id"]) for x in insts}, {norm(x["display_name"]) for x in insts},
                          sorted({x["display_name"] for x in insts}))
        return _auth[aid]


    def load_lists():
        frames = []
        for y in YEARS:
            d = pd.read_excel(os.path.join(LISTS, f"{y}_HCR.xlsx"))
            d.columns = [str(c).strip() for c in d.columns]
            first = next(c for c in d.columns if "first" in c.lower()); last = next(c for c in d.columns if "last" in c.lower() or "family" in c.lower())
            cat = next(c for c in d.columns if "categ" in c.lower())
            prim = next(c for c in d.columns if "primary" in c.lower())
            sec = next((c for c in d.columns if "secondary" in c.lower()), None)
            frames.append(pd.DataFrame({"year": y, "first": d[first], "last": d[last], "category": d[cat],
                                        "affiliation": d[prim].fillna("").astype(str) + ("; " + d[sec].fillna("").astype(str) if sec else "")}))
        return pd.concat(frames, ignore_index=True)


    def main():
        H = load_lists()
        idx = {}
        for i, r in H.iterrows():
            for k in keys(r["last"], r["first"]):
                idx.setdefault(k, []).append(i)
        pc = pd.read_csv(os.path.join(ST, "controls", "per_control.csv")); pc = pc[pc["Consistent with Scopus"]]
        pr = pd.read_csv(os.path.join(ST, "reference", "per_reference.csv")); pr = pr[pr["Consistent with Scopus"]]
        hist = pd.read_csv(os.path.join(ST, "history", "per_researcher.csv"))
        cs_years = {a: {int(y) for y in str(v).split(";") if y.strip().isdigit()} for a, v in zip(hist["OpenAlex ID"], hist["Years listed"]) if isinstance(a, str)}
        for a in pd.read_csv(os.path.join(ST, "per_member.csv"))["OpenAlex ID"].dropna():   # every resolved 2025 entry
            cs_years.setdefault(a, set()).add(2025)
        rows, people = [], []
        for grp, d, ncol, icol in (("matched", pc, "Control authfull", "Control OpenAlex ID"), ("reference", pr, "authfull", "openalex_id")):
            for _, r in d.iterrows():
                last, first = split_name(r[ncol])
                hits = sorted({i for k in keys(last, first) for i in idx.get(k, [])})
                years = set(cs_years.get(r[icol], set()))       # same OpenAlex author as a listed person
                if hits:
                    ids, names, shown = author_institutions(r[icol])
                    for i in hits:
                        h = H.loc[i]
                        their_ids, their_names = institution_ids(h["affiliation"])
                        same = bool(ids & their_ids) or bool(names & their_names) or any(   # one institution name inside the other
                            len(m) >= 12 and len(t) >= 12 and (m in t or t in m) for m in names for t in their_names)
                        years |= {int(h["year"])} if same else set()
                        rows.append({"group": grp, "OpenAlex ID": r[icol], "name": r[ncol], "year": int(h["year"]),
                                     "category": h["category"], "HCR affiliation": h["affiliation"], "same person": same,
                                     "OpenAlex affiliations": "; ".join(shown)[:300]})
                people.append({"group": grp, "OpenAlex ID": r[icol], "name": r[ncol], "on a 2025 list": 2025 in years,
                               "years listed (any category)": " ".join(map(str, sorted(years)))})
        pd.DataFrame(rows).to_csv(os.path.join(ST, "hcr_any_list_hits.csv"), index=False)
        P = pd.DataFrame(people); P.to_csv(os.path.join(ST, "hcr_any_list.csv"), index=False)
        pd.set_option("display.width", 250)
        print(P[P["years listed (any category)"] != ""].to_string(index=False))
        print("\non a 2025 list:", P.groupby("group")["on a 2025 list"].sum().to_dict(),
              " ever listed 2014-2025:", P.groupby("group")["years listed (any category)"].apply(lambda s: int((s != "").sum())).to_dict())


    main()


# =====================================================================================================================
# Matched peers
# =====================================================================================================================
#
# Matched peers: for each HCR-listed researcher of the 2025 analysis, up to five researchers of the Stanford-Elsevier
# career table (August 2025 release, els.csv) who are not on any 2025 Highly Cited Researchers list and whose career
# indicators are the most similar.
#
# Candidates share the HCR-listed researcher's subfield, first published within 3 years of them, published in 2022 or
# later, and have Scopus non-self citations within a factor of 1.25. Among them, peers are the nearest by Mahalanobis
# distance (covariance of the candidate pool) on the log of the other five indicators of the composite (h-index,
# hm-index, and citations to single-, single or first-, and single, first, or last-authored papers), the log number of
# papers, and the first publication year. HCR-listed researchers are matched in random order, peers without replacement,
# and up to 40 candidates are tried per researcher. A candidate must resolve to an OpenAlex author with the resolver of
# the main study (profiles.resolve_author), must not be the OpenAlex author of a 2025 HCR-listed researcher, and must have
# OpenAlex citations 0.6-3.0 and works 0.4-3.0 times its Scopus record and at least half of its OpenAlex topics in
# computer science or engineering. Excluded by name: anyone whose name (surname with full given names, or surname with
# first given name) appears on the 2025 list in any category.
#
# load() drops peers whose OpenAlex author was resolved from more than one Stanford-Elsevier record, and peers whose
# classified profile is not consistent with Scopus. Every other script reads the peers through load().
#
#     python comparison_groups.py peers select
#     PROFILE_DIRS=peers/profiles python ground_truth.py s2-abstracts
#     PROFILE_DIRS=peers/profiles python ground_truth.py publisher-abstracts doi      (then: ground_truth.py publisher-abstracts springer)
#     python comparison_groups.py peers classify
#     python comparison_groups.py peers analyze
#
# Output: hcr_study/peers/ (matches.csv, profiles/, classified/, per_peer.csv, analysis.md, pairs_INTERNAL.csv)

PEERS_OUT = os.path.join(STUDY, "peers")
PEERS_PROFILES, PEERS_CLASSIFIED = os.path.join(PEERS_OUT, "profiles"), os.path.join(PEERS_OUT, "classified")
ELS_FILE = os.path.join(HERE, "inputs", "stanford_elsevier", "els.csv")
LISTS = os.path.join(HERE, "inputs", "hcr_lists")      # Clarivate's lists, <year>_HCR.xlsx
K_PEERS, MAX_TRIES, PEERS_SEED = 5, 40, 2026
MAX_FIRSTYR_GAP, MIN_LASTYR, CALIPER = 3, 2022, 1.25
CITE_RATIO, WORK_RATIO = (0.6, 3.0), (0.4, 3.0)
KEY = ["authfull", "firstyr", "np6024"]
FEATURES = ["h24 (ns)", "hm24 (ns)", "ncs (ns)", "ncsf (ns)", "ncsfl (ns)", "np6024"]
BALANCE = ["nc9624 (ns)"] + FEATURES


def ascii_lower(s):
    return unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower()


def name_keys(last, first):
    first = re.sub(r"\(.*?\)", " ", str(first)).strip()
    tok = [t for t in re.split(r"[\s.\-]+", first) if t]
    l = re.sub(r"[^a-z]", "", ascii_lower(last))
    return {l + "||" + re.sub(r"[^a-z]", "", ascii_lower(first)), l + "|" + (re.sub(r"[^a-z]", "", ascii_lower(tok[0])) if tok else "")}


def authfull_keys(a):
    last, first = (str(a).split(",", 1) + [""])[:2] if "," in str(a) else (a, "")
    return name_keys(last, first)


def names_2025():
    x = pd.read_excel(os.path.join(LISTS, "2025_HCR.xlsx"))
    keys = set()
    for l, f in zip(x["Last Name"], x["First Name"]):
        keys |= name_keys(l, f)
    return keys


def career():
    return pd.read_csv(ELS_FILE, thousands=",", low_memory=False, encoding="utf-8-sig").drop_duplicates(KEY, keep=False)


def features(d):
    return np.column_stack([np.log1p(d[FEATURES].astype(float).values), d["firstyr"].astype(float).values])


def members(car):
    """HCR-listed researchers of the 2025 analysis with their Stanford-Elsevier record"""
    pm = pd.read_csv(os.path.join(STUDY, "per_member.csv"))
    pm = pm[pm["Consistent with Scopus"]]
    b = pd.read_csv(os.path.join(STUDY, "BTAS_HCR_Computer_Science.csv"))
    b = b[b["Match status"] == "ok"][["HCR name"] + KEY]
    return pm[["HCR name", "OpenAlex ID"]].merge(b, on="HCR name").merge(car, on=KEY)


def peers_select():
    os.makedirs(PEERS_PROFILES, exist_ok=True)
    car = career()
    mem = members(car)
    print(f"HCR-listed researchers to match: {len(mem)}")
    bad = names_2025()
    hist = pd.read_csv(os.path.join(STUDY, "history", "per_researcher.csv"))
    listed25 = set(pd.read_csv(os.path.join(STUDY, "per_member.csv"))["OpenAlex ID"].dropna()) | \
        {a for a, v in zip(hist["OpenAlex ID"], hist["Years listed"]) if "2025" in str(v)}
    pool = car[car["sm-subfield-1"].isin(set(mem["sm-subfield-1"])) & (car["lastyr"] >= MIN_LASTYR)].copy()
    pool = pool[[not (authfull_keys(a) & bad) for a in pool["authfull"]]]
    VI = np.linalg.inv(np.cov(features(pool), rowvar=False))
    Xm = features(mem)
    path = os.path.join(PEERS_OUT, "matches.csv")
    prev = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame()
    rows = prev.to_dict("records")
    used = set(prev["pool index"]) if len(prev) else set()
    done = prev[prev.status == "ok"].groupby("HCR name").size().to_dict() if len(prev) else {}
    rng = np.random.default_rng(PEERS_SEED)
    for k in rng.permutation(len(mem)):
        m = mem.iloc[k]
        need = K_PEERS - done.get(m["HCR name"], 0)
        if need <= 0:
            continue
        c0 = m["nc9624 (ns)"]
        cand = pool[(pool["sm-subfield-1"] == m["sm-subfield-1"]) & ((pool["firstyr"] - m["firstyr"]).abs() <= MAX_FIRSTYR_GAP)
                    & pool["nc9624 (ns)"].between(c0 / CALIPER, c0 * CALIPER)]
        diff = features(cand) - Xm[k]
        cand = cand.assign(dist=np.sqrt(np.einsum("ij,jk,ik->i", diff, VI, diff))).sort_values("dist")
        cand = cand[~cand.index.isin(used)].head(MAX_TRIES)
        got = 0
        for idx, c in cand.iterrows():
            if got >= need:
                break
            used.add(idx)
            row = {"HCR name": m["HCR name"], "pool index": idx, "distance": c["dist"], "candidates": len(cand),
                   **{f"peer {x}": c[x] for x in ["authfull", "inst_name", "cntry", "firstyr", "np6024", "nc9624 (ns)", "h24 (ns)", "c (ns)"]}}
            try:
                a, status, inst, _ = profiles.resolve_author(pd.Series({"Full Name": c["authfull"], "Primary Affiliation": c["inst_name"], "Secondary Affiliations": ""}))
            except Exception as e:
                rows.append({**row, "status": f"resolve failed: {e}"[:80]}); continue
            if a is None or status.startswith(("weak", "not")):
                rows.append({**row, "status": f"unresolved ({status})"}); continue
            aid = short_id(a["id"])
            if aid in listed25:
                rows.append({**row, "status": "HCR-listed 2025 (OpenAlex ID)"}); continue
            cr = a.get("cited_by_count", 0) / max(c["nc9624 (ns)"], 1)
            wr = a.get("works_count", 0) / max(c["np6024"], 1)
            if not (CITE_RATIO[0] <= cr <= CITE_RATIO[1] and WORK_RATIO[0] <= wr <= WORK_RATIO[1]):
                rows.append({**row, "status": "implausible profile"}); continue
            t = openalex_get(f"authors/{aid}", {"select": "topics"}).get("topics") or []
            n_t = sum(x["count"] for x in t) or 1
            if sum(x["count"] for x in t if x["field"]["display_name"] in ("Computer Science", "Engineering")) / n_t < 0.5:
                rows.append({**row, "status": "topics outside computing"}); continue
            p = os.path.join(PEERS_PROFILES, f"{aid}.csv")
            if not os.path.exists(p):
                works = openalex_get_all("works", {"filter": f"author.id:{aid}", "select": WORK_FIELDS + profiles.EXTRA_WORK_FIELDS})
                merge_versions(pd.DataFrame([profiles.work_row(w) for w in works])).to_csv(p, index=False)
            rows.append({**row, "status": "ok", "OpenAlex ID": aid, "resolver status": status, "OpenAlex/Scopus citations": cr,
                         "OpenAlex/Scopus works": wr})
            got += 1
        print(f"  {done.get(m['HCR name'], 0) + got}/{K_PEERS} peers ({len(cand)} candidates in the caliper)", flush=True)
        pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows)["status"].value_counts().to_string())


def peers_classify():
    os.makedirs(PEERS_CLASSIFIED, exist_ok=True)
    M = pd.read_csv(os.path.join(PEERS_OUT, "matches.csv"))
    M = M[M.status == "ok"].drop_duplicates("OpenAlex ID")
    rows = []
    for i, r in enumerate(M.to_dict("records"), 1):
        aid = r["OpenAlex ID"]
        df = pd.read_csv(os.path.join(PEERS_PROFILES, f"{aid}.csv"))
        df["citationCount"] = df["citationCount"].fillna(0).astype(int)
        papers, _ = profiles.classify_papers(df)
        papers.to_csv(os.path.join(PEERS_CLASSIFIED, f"{aid}.csv"), index=False)
        m, _, _ = profiles.author_metrics(df, papers["Category"].values)
        cr = m["Citations"] / max(r["peer nc9624 (ns)"], 1)
        pr = m["Papers"] / max(r["peer np6024"], 1)
        rows.append({"OpenAlex ID": aid, "Consistent with Scopus": bool(0.6 <= cr <= 3.0 and 0.4 <= pr <= 3.0), **m})
        if i % 25 == 0:
            print(f"  classified {i}/{len(M)}", flush=True)
    pd.DataFrame(rows).to_csv(os.path.join(PEERS_OUT, "per_peer.csv"), index=False)
    print(f"classified {len(rows)} peers")


def load_peers():
    """One row per usable peer: the HCR-listed researcher it is matched to, its Stanford-Elsevier record, the path of its
    classified profile, and its metrics (per_peer.csv)."""
    M = pd.read_csv(os.path.join(PEERS_OUT, "matches.csv"))
    M = M[M.status == "ok"]
    M = M[~M["OpenAlex ID"].duplicated(keep=False)]           # one OpenAlex author resolved from two Stanford-Elsevier records
    pp = pd.read_csv(os.path.join(PEERS_OUT, "per_peer.csv"))
    pp = pp[pp["Consistent with Scopus"]]
    M = M.rename(columns={f"peer {c}": c for c in ["authfull", "inst_name", "cntry", "firstyr", "np6024", "nc9624 (ns)", "h24 (ns)", "c (ns)"]})
    M = M.merge(pp, on="OpenAlex ID")
    M["path"] = [os.path.join(PEERS_CLASSIFIED, f"{a}.csv") for a in M["OpenAlex ID"]]
    return M.reset_index(drop=True)


def ever_listed(P):
    """Peers whose name (surname with full given names, or surname with first given name) appears on any Highly Cited
    Researchers list from 2014 to 2025 in any category, or whose OpenAlex author is a resolved Computer Science
    HCR-listed researcher of those years (profiles.py history): a strict flag for a sensitivity check, namesakes included."""
    keys = set()
    for y in range(2014, 2026):
        x = pd.read_excel(os.path.join(LISTS, f"{y}_HCR.xlsx"))
        last = x[next(c for c in x.columns if "last" in c.lower() or "family" in c.lower())]
        first = x[next(c for c in x.columns if "first" in c.lower())]
        for l, f in zip(last, first):
            keys |= name_keys(l, f)
    cs = set(pd.read_csv(os.path.join(STUDY, "history", "resolution.csv"))["OpenAlex ID"].dropna())
    return pd.Series([bool(authfull_keys(a) & keys) or i in cs for a, i in zip(P["authfull"], P["OpenAlex ID"])], index=P.index)


def peers_as_controls():
    """The peers with the column names of comparison_groups.py controls (matches.csv, per_control.csv), for the analysis scripts"""
    P = load_peers()
    mt = pd.DataFrame({"HCR name": P["HCR name"], "Control authfull": P["authfull"], "Control institution": P["inst_name"],
                       "Control firstyr": P["firstyr"], "Control citations (Scopus ns)": P["nc9624 (ns)"],
                       "Control papers (Scopus)": P["np6024"], "Control OpenAlex ID": P["OpenAlex ID"]})
    metrics = pd.read_csv(os.path.join(PEERS_OUT, "per_peer.csv")).columns.drop("OpenAlex ID")
    pc = pd.concat([pd.DataFrame({"HCR name (matched member)": P["HCR name"], "Control OpenAlex ID": P["OpenAlex ID"],
                                  "Control authfull": P["authfull"]}), P[list(metrics)]], axis=1)
    return mt, pc


def peers_analyze():
    from scipy.stats import wilcoxon, mannwhitneyu
    car = career()
    P = load_peers()
    H = members(car)
    pm = pd.read_csv(os.path.join(STUDY, "per_member.csv")).set_index("HCR name")
    for c in ["Survey % of citations", "Survey % of papers", "Survey % of top-1%", "h change %"]:
        H[c] = H["HCR name"].map(pm[c])
    cols = ["Survey % of citations", "Survey % of papers", "Survey % of top-1%", "h change %"]
    for d in (H, P):
        rc, sp = d["Survey % of citations"] / 100, d["Survey % of papers"] / 100
        d["dividend D"] = np.where((sp > 0) & (rc < 1), rc / (1 - rc) * (1 - sp) / sp, np.nan)
    cols.append("dividend D")
    peer_car = car.set_index(KEY).loc[[tuple(x) for x in P[KEY].values]].reset_index()
    Hm = H[H["HCR name"].isin(P["HCR name"])]
    out = ["# HCR-listed researchers against their matched peers", "",
           f"Peers: {len(P)} for {P['HCR name'].nunique()} of {len(H)} HCR-listed researchers (per researcher: "
           + ", ".join(f"{k} peers {v}" for k, v in P.groupby('HCR name').size().value_counts().sort_index().items()) + ")",
           f"Without a peer: {len(H) - len(Hm)} HCR-listed researchers (their ranks by Scopus citations among the {len(H)}: "
           f"{sorted(H['nc9624 (ns)'].rank(ascending=False)[~H['HCR name'].isin(P['HCR name'])].astype(int).tolist())})",
           f"Peers' Scopus citations relative to their HCR-listed researcher: median {np.median(P['nc9624 (ns)'].values / Hm.set_index('HCR name').loc[P['HCR name'], 'nc9624 (ns)'].values):.3f}",
           "", "## Balance on the career table (HCR-listed researchers with peers against all peers)", ""]
    bal = []
    for f in BALANCE + ["firstyr", "c (ns)"]:
        a, c = Hm[f].astype(float), peer_car[f].astype(float)
        lg = f != "firstyr"
        a2, c2 = (np.log1p(a), np.log1p(c)) if lg else (a, c)
        bal.append({"field": f, "HCR-listed median": a.median(), "peers median": c.median(),
                    "standardized difference" + (" (log)" if lg else ""): (a2.mean() - c2.mean()) / np.sqrt((a2.var() + c2.var()) / 2)})
    out += [pd.DataFrame(bal).round(3).to_markdown(index=False), "", "## Survey dependence (paired: the HCR-listed researcher against the mean of their peers)", ""]
    g = P.groupby("HCR name")[cols].mean()
    J = H.set_index("HCR name")[cols].join(g, rsuffix=" (peers)", how="inner")
    for c in cols:
        d = (J[c] - J[c + " (peers)"]).dropna()
        out.append(f"- {c}: HCR-listed median {J[c].median():.2f}, peers' mean per researcher, median {J[c + ' (peers)'].median():.2f}; "
                   f"paired difference median {d.median():+.2f}, HCR-listed higher for {int((d > 0).sum())} of {len(d)}, "
                   f"Wilcoxon p = {wilcoxon(d).pvalue:.2g}; all peers pooled: median {P[c].median():.2f}, "
                   f"Mann-Whitney p = {mannwhitneyu(H[c].dropna(), P[c].dropna()).pvalue:.2g}")
    J.to_csv(os.path.join(PEERS_OUT, "pairs_INTERNAL.csv"))
    open(os.path.join(PEERS_OUT, "analysis.md"), "w").write("\n".join(out) + "\n")
    print("\n".join(out))


def peers():
    {"select": peers_select, "classify": peers_classify, "analyze": peers_analyze}[sys.argv[1]]()


# =====================================================================================================================
# Boundary samples at the top-2% cutoff
# =====================================================================================================================
#
# Stanford-Elsevier top-2% inclusion at the boundary (research question RQ3).
#
# Samples researchers just inside the top-2% boundary of Networking and Telecommunications
# (subfield ranks 3,500-4,346 by composite c (ns); the boundary is rank 4,346, c = 2.886),
# resolves them in OpenAlex with the rules of the main study, retrieves and classifies their
# profiles, and recomputes their composite without surveys (analyses.py composite).
#
# Turnover: the top 2% has a fixed size, so researchers who fall out are replaced one for one.
# Removing every record's surveys shifts the boundary itself. We simulate the local population
# around the boundary (the observed records inside it, and a population below it with the same
# local density, which the release does not contain), give each record a change in c drawn from
# the boundary sample (observed values for sampled records), recompute the rank-4,346 threshold,
# and count exits among the records inside. Bootstrap over sampled researchers.
#
#     python comparison_groups.py boundary select     (WORKER/NWORKERS for parallel)
#     python comparison_groups.py boundary merge
#     python comparison_groups.py boundary classify
#     python comparison_groups.py boundary analyze
#     (SUBFIELD_NAME="Artificial Intelligence & Image Processing" for the second subfield)
#
# Output: hcr_study/boundary/ (Networking & Telecommunications), hcr_study/boundary_artificialin/

# Subfield (Science-Metrix name) via SUBFIELD_NAME; the band is the 20% of the top-2% ranks just inside the boundary
def set_boundary_subfield(name):
    """The subfield sampled and its output folder (hcr_study/boundary for Networking & Telecommunications)."""
    global NT, BOUNDARY_OUT, BOUNDARY_PROFILES, BOUNDARY_CLASSIFIED, BOUNDARY_BAND
    NT = name
    slug = "" if NT.startswith("Networking") else "_" + "".join(ch for ch in NT.lower() if ch.isalnum())[:12]
    BOUNDARY_OUT = os.path.join(HERE, "hcr_study", "boundary" + slug)
    BOUNDARY_PROFILES = os.path.join(BOUNDARY_OUT, "profiles")
    BOUNDARY_CLASSIFIED = os.path.join(BOUNDARY_OUT, "classified")
    BOUNDARY_BAND = (3500, 4346) if not slug else None   # set from the subfield's size in frame()


set_boundary_subfield(os.environ.get("SUBFIELD_NAME", "Networking & Telecommunications"))
BOUNDARY_N, BOUNDARY_SEED = 170, 11
K = ["authfull", "firstyr", "np6024"]


def boundary_frame():
    e = pd.read_csv(os.path.join(HERE, "inputs", "stanford_elsevier", "els.csv"), thousands=",", low_memory=False, encoding="utf-8-sig",
                    usecols=K + ["inst_name", "cntry", "lastyr", "nc9624 (ns)", "c (ns)", "c", "sm-subfield-1",
                                 "rank sm-subfield-1 (ns)", "sm-subfield-1 count"])
    return e[e["sm-subfield-1"] == NT].drop_duplicates(K, keep=False)


def band_ranks(nt):
    if BOUNDARY_BAND:
        return BOUNDARY_BAND
    k2 = int(round(0.02 * nt["sm-subfield-1 count"].iloc[0]))
    return (int(round(0.8 * k2)), k2)


def boundary_select():
    os.makedirs(BOUNDARY_PROFILES, exist_ok=True)
    nt = boundary_frame()
    band = nt[nt["rank sm-subfield-1 (ns)"].between(*band_ranks(nt))]
    sample = band.sample(BOUNDARY_N, random_state=BOUNDARY_SEED)
    k, n = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
    sample = sample.iloc[k::n]
    path = os.path.join(BOUNDARY_OUT, f"sample_part{k}.csv")
    done = pd.read_csv(path) if os.path.exists(path) else pd.DataFrame(columns=["authfull"])
    done = done[~done.get("status", pd.Series(dtype=str)).astype(str).str.startswith("error")] if len(done) else done
    rows = done.to_dict("records")
    seen = set(zip(done["authfull"], done.get("firstyr", pd.Series(dtype=float)))) if len(done) else set()
    for _, c in sample.iterrows():
        if (c["authfull"], c["firstyr"]) in seen:
            continue
        row = {**{x: c[x] for x in K + ["inst_name", "nc9624 (ns)", "c (ns)", "c", "rank sm-subfield-1 (ns)"]}}
        try:
            a, status, _, _ = profiles.resolve_author(pd.Series({"Full Name": c["authfull"], "Primary Affiliation": c["inst_name"],
                                                   "Secondary Affiliations": ""}))
        except Exception as ex:
            rows.append({**row, "status": f"error: {ex}"[:80]})
            continue
        if a is None or status.startswith(("weak", "not")):
            rows.append({**row, "status": status})
            continue
        cr = a.get("cited_by_count", 0) / max(c["nc9624 (ns)"], 1)
        wr = a.get("works_count", 0) / max(c["np6024"], 1)
        if not (0.6 <= cr <= 3.0 and 0.4 <= wr <= 3.0):
            rows.append({**row, "status": "implausible author record"})
            continue
        aid = short_id(a["id"])
        p = os.path.join(BOUNDARY_PROFILES, f"{aid}.csv")
        if not os.path.exists(p):
            works = openalex_get_all("works", {"filter": f"author.id:{aid}", "select": WORK_FIELDS + profiles.EXTRA_WORK_FIELDS})
            merge_versions(pd.DataFrame([profiles.work_row(w) for w in works])).to_csv(p, index=False)
        rows.append({**row, "status": status, "openalex_id": aid})
        pd.DataFrame(rows).to_csv(path, index=False)
    pd.DataFrame(rows).to_csv(path, index=False)
    print(pd.DataFrame(rows)["status"].value_counts().to_string())


def boundary_merge():
    parts = [os.path.join(BOUNDARY_OUT, f) for f in sorted(os.listdir(BOUNDARY_OUT)) if f.startswith("sample_part")]
    s = pd.concat([pd.read_csv(f) for f in parts], ignore_index=True).drop_duplicates(["authfull", "firstyr"])
    s.to_csv(os.path.join(BOUNDARY_OUT, "sample.csv"), index=False)
    print(len(s), s["status"].value_counts().to_dict())


def boundary_classify():
    import analyses
    os.makedirs(BOUNDARY_CLASSIFIED, exist_ok=True)
    ml = analyses.maxlog()
    s = pd.read_csv(os.path.join(BOUNDARY_OUT, "sample.csv"))
    s = s[s["openalex_id"].notna()]
    rows = []
    for _, r in s.iterrows():
        df = pd.read_csv(os.path.join(BOUNDARY_PROFILES, f"{r['openalex_id']}.csv"))
        df["citationCount"] = df["citationCount"].fillna(0).astype(int)
        cp = os.path.join(BOUNDARY_CLASSIFIED, f"{r['openalex_id']}.csv")
        if os.path.exists(cp):
            papers = pd.read_csv(cp)
        else:
            papers, _ = profiles.classify_papers(df)
            papers.to_csv(cp, index=False)
        m, _, _ = profiles.author_metrics(df, papers["Category"].values)
        cr, pr = m["Citations"] / max(r["nc9624 (ns)"], 1), m["Papers"] / max(r["np6024"], 1)
        rows.append({**r.to_dict(), "Survey % of citations": m["Survey % of citations"],
                     "Survey % of papers": m["Survey % of papers"],
                     "Consistent with Scopus": bool(0.6 <= cr <= 3.0 and 0.4 <= pr <= 3.0),
                     **analyses.profile_scores(cp, r["authfull"], ml)})
    out = pd.DataFrame(rows)
    out["dc"] = out["c_oa*"] - out["c_oa"]
    out.to_csv(os.path.join(BOUNDARY_OUT, "boundary_INTERNAL.csv"), index=False)
    print(f"{len(out)} classified; consistent {int(out['Consistent with Scopus'].sum())}")


def boundary_analyze(n_boot=2000):
    rng = np.random.default_rng(5)
    b = pd.read_csv(os.path.join(BOUNDARY_OUT, "boundary_INTERNAL.csv"))
    b = b[b["Consistent with Scopus"]].copy()
    nt = boundary_frame()
    k2 = int(round(0.02 * nt["sm-subfield-1 count"].iloc[0]))
    cut = nt.loc[nt["rank sm-subfield-1 (ns)"] <= k2, "c (ns)"].min()
    inside = np.sort(nt.loc[nt["rank sm-subfield-1 (ns)"] <= k2, "c (ns)"].values)[::-1]
    # local density just inside the boundary (records per unit c), used to extend the population below it
    band_c = nt.loc[nt["rank sm-subfield-1 (ns)"].between(*band_ranks(nt)), "c (ns)"].values
    dens = len(band_c) / (band_c.max() - band_c.min())
    span = 0.25                                   # c units below the cutoff to simulate (well beyond any drop)
    n_below = int(dens * span)
    below = cut - np.linspace(0, span, n_below + 2)[1:-1]
    dc_obs = b["dc"].values
    # records outside the band (far above the boundary) keep the reference-sample median change; only the
    # band and the population below matter for the threshold
    in_band = inside <= 1.2 * cut                  # within 20% of the cutoff: may cross it
    br = band_ranks(nt)
    lines = [f"[{NT}] Boundary sample: {len(b)} researchers consistent with Scopus (of {BOUNDARY_N} sampled) in subfield ranks "
             f"{br[0]:,}-{br[1]:,}; composite c (ns) {band_c.min():.3f}-{band_c.max():.3f}, cutoff {cut:.3f}",
             f"Survey share of citations: median {b['Survey % of citations'].median():.1f}%; composite change "
             f"median {np.median(dc_obs):+.3f} ({100 * np.median(b['dc'] / b['c_oa']):+.1f}%), "
             f"10th-90th percentile {np.percentile(dc_obs, 10):+.3f} to {np.percentile(dc_obs, 90):+.3f}"]
    # point estimate: mean over random assignments of the observed changes; interval: bootstrap of the changes
    point, exits_all, thr_shift = [], [], []
    for it in range(n_boot):
        for boot in (False, True):
            d = rng.choice(dc_obs, len(dc_obs)) if boot else dc_obs
            new_inside = inside.copy()
            new_inside[in_band] = inside[in_band] + rng.choice(d, in_band.sum())
            new_inside[~in_band] = inside[~in_band] + np.median(d)          # more than 20% above: cannot cross
            new_below = below + rng.choice(d, len(below))
            allc = np.concatenate([new_inside, new_below])
            thr = np.sort(allc)[::-1][k2 - 1]
            if boot:
                exits_all.append(int((new_inside < thr).sum()))
            else:
                point.append(int((new_inside < thr).sum()))
                thr_shift.append(thr - cut)
    exits_all, est = np.array(exits_all), np.mean(point)
    lines.append(f"Universal exclusion: boundary shifts by {np.median(thr_shift):+.3f}; researchers leaving the top 2% "
                 f"(and replaced): {est:.0f} of {k2:,} ({100 * est / k2:.1f}%), bootstrap 95% CI "
                 f"{np.percentile(exits_all, 2.5):.0f}-{np.percentile(exits_all, 97.5):.0f} "
                 f"({100 * np.percentile(exits_all, 2.5) / k2:.1f}-{100 * np.percentile(exits_all, 97.5) / k2:.1f}%)")
    # within the sampled band: share who would leave
    thr0 = cut + np.median(thr_shift)
    left = (b["c (ns)"] + b["dc"]) < thr0
    lines.append(f"Sampled band researchers who would leave: {int(left.sum())} of {len(b)} ({100 * left.mean():.0f}%); "
                 f"their median survey share of citations {b.loc[left, 'Survey % of citations'].median():.1f}% vs "
                 f"{b.loc[~left, 'Survey % of citations'].median():.1f}% for those who stay")
    with open(os.path.join(BOUNDARY_OUT, "stats.md"), "w") as fh:
        fh.write("\n".join(f"- {l}" for l in lines) + "\n")
    print("\n".join(lines))


def boundary():
    {"select": boundary_select, "merge": boundary_merge, "classify": boundary_classify, "analyze": boundary_analyze}[sys.argv[1]]()


# =====================================================================================================================
# Fellows: citation, name and region rules
# =====================================================================================================================
#
# Rules shared by the Fellows comparison (comparison_groups.py fellows): which Fellow citations are computer science-related, the
# name rule for OpenAlex candidates, world-region country lists, the countries of an OpenAlex author's institutions,
# and the content words of a citation or of an author's OpenAlex topics.

# ---------- computer science-related Fellows, from the citation text ----------
CS_CITATION = re.compile(r"comput|software|\bdata\b|database|network|wireless|communicat|learning|artificial intelligence|\bAI\b|neural|"
                r"vision|imag|video|multimedia|secur|privacy|cyber|crypt|algorithm|internet|cloud|recommender|mining|retrieval|"
                r"language|speech|robot|verification|distributed|mobile|sensing|visuali[sz]ation|human-computer|interaction|"
                r"blockchain|\bIoT\b|semantic|knowledge|information|signal processing|coding|estimation|optimization", re.I)
NOT_CS_CITATION = re.compile(r"power (system|grid|electronic|converter|amplifier|management|transmission)|\bgrids?\b|semiconductor|"
                    r"antenna|microwave|circuit|transistor|photonic|optic|laser|material|magnet|batter|motor|plasma|CMOS|"
                    r"millimeter|terahertz|electromagnetic|x-ray|nano|packag|high-voltage|energy|device|insulation|"
                    r"electric (machine|vehicle|drive)|acoustic|ultrasound|radar", re.I)

# ---------- world regions -> institution countries (ISO 3166 alpha-2) ----------
EUROPE = "AL AD AM AT AZ BY BE BA BG HR CY CZ DK EE FI FR GE DE GR HU IS IE IT KZ XK LV LI LT LU MT MD MC ME NL MK NO PL PT RO RU SM RS SK SI ES SE CH TR UA GB VA"
MIDEAST_AFRICA = ("AE BH IL IQ IR JO KW LB OM PS QA SA SY YE EG DZ AO BJ BW BF BI CM CV CF TD KM CG CD CI DJ GQ ER SZ ET GA GM GH GN "
                  "GW KE LS LR LY MG MW ML MR MU MA MZ NA NE NG RW ST SN SC SL SO ZA SS SD TZ TG TN UG ZM ZW")
LATAM = "MX GT BZ SV HN NI CR PA CU DO HT JM TT BS BB AR BO BR CL CO EC GY PY PE SR UY VE PR"
ASIA_PACIFIC = "CN HK MO TW JP KR MN IN PK BD LK NP BT MV AF SG MY TH VN PH ID KH LA MM BN TL AU NZ FJ PG UZ KG TJ TM"

GENERIC = {"for", "contribution", "contributions", "to", "the", "of", "and", "in", "on", "leadership", "development", "design",
           "system", "systems", "application", "applications", "theory", "method", "methods", "technology", "technologies",
           "engineering", "analysis", "advanced", "novel", "technique", "techniques", "their", "with", "based", "high", "large",
           "scale", "its", "from", "by", "an", "a", "at", "as", "use", "using", "new", "practice", "industry", "industrial",
           "research", "education", "field", "science", "sciences", "general"}
stem = lambda w: w[:-1] if len(w) > 4 and w.endswith("s") else w
words = lambda s: {stem(w) for w in re.findall(r"[a-z]+", ascii_lower(s)) if w not in GENERIC and len(w) > 2}
SUFFIX = {"jr", "sr", "ii", "iii", "iv"}


def split_name(name):
    t = [w for w in re.findall(r"[a-z]+", ascii_lower(name)) if w not in SUFFIX]
    return t[-1], t[:-1]


def name_ok(surname, given, author):
    """Some name variant contains the surname as a word and, apart from it, starts with the Fellow's first full
    given name (hyphens and spaces ignored, so 'Ab-Cd Ef' matches 'Abcd Ef'); an initial-only given
    name must match the variant's first initial."""
    full = [g for g in given if len(g) > 1]
    for n in [author["display_name"]] + (author.get("display_name_alternatives") or []):
        parts = re.findall(r"[a-z]+", ascii_lower(n))
        if surname not in parts:
            continue
        rest = parts[:]; rest.remove(surname); joined = "".join(rest)
        if not rest:
            continue
        if full and (full[0] in rest or joined.startswith(full[0])):
            return True
        if not full and given and rest[0][0] == given[0][0]:
            return True
    return False


def author_countries(a):
    c = {i.get("country_code") for i in (a.get("last_known_institutions") or [])}
    c |= {(x.get("institution") or {}).get("country_code") for x in (a.get("affiliations") or []) if max(x.get("years") or [0]) >= 2018}
    return c - {None}


def topic_words(a):
    s = set()
    for t in a.get("topics") or []:
        s |= words(t.get("display_name", "")) | words((t.get("subfield") or {}).get("display_name", ""))
    return s


# =====================================================================================================================
# IEEE and ACM Fellows
# =====================================================================================================================

def fellows():
    """Peer-elected comparison groups for RQ4: IEEE Fellows of the class of 2026 and ACM Fellows announced in January 2026
    (ACM's "2025 Fellows"), against the 102 HCR-listed researchers, all classified by the same final model.

    Fellows are elected by peers, whereas the HCR list is selected by citation counts. Both groups are measured on
    their current OpenAlex profiles, which is the time of election for these classes.
      1. load      inputs/fellows/IEEE-fellow-2026.pdf (name, institution, location, citation; read with pdftotext)
                   and the ACM announcements and inputs/fellows/ACM Fellows.csv (name, region); IEEE Fellows count
                   as computer science-related by their citation text (rule in comparison_groups.py), all ACM Fellows do.
      2. match     OpenAlex author by rule (no hand checking): name, then institution words and country (IEEE) or
                   world region and computing share of topics (ACM), plus citation words against the author's topics
                   (IEEE). Ambiguous or weak matches are dropped and counted.
      3. fetch     profiles (profiles.fetch_profile) and missing abstracts from Semantic Scholar, Scopus and Springer
                   Nature, the same sources as for the HCR-listed researchers.
      4. classify  SciGenre via profiles.classify_papers and author_metrics.
      5. report    academic researchers only, in every group: medians, Mann-Whitney tests with Holm correction, and survey
                   dependence adjusted for citation level and career start by a median regression fitted on the academic
                   researchers of the Stanford-Elsevier reference sample.
    Usage: comparison_groups.py fellows [load|match|academic|fetch|classify|report|all]   (classify reads hcr_study/reference/)
    Output: hcr_study/fellows2026/ (*_INTERNAL files name people; report.md is anonymous)"""
    FELLOWS = os.path.join(HERE, "inputs", "fellows")
    common.OPENALEX_MAILTO = ""
    OUT = os.path.join(HERE, "hcr_study", "fellows2026"); PROFILES = os.path.join(OUT, "profiles"); CLASSIFIED = os.path.join(OUT, "classified")
    USABLE = {"ok"}                 # ambiguous matches are dropped (official affiliations are available)
    AUTHOR_FIELDS = "id,display_name,display_name_alternatives,works_count,cited_by_count,last_known_institutions,affiliations,topics"

    # ---------- 1. lists ----------
    def load_ieee():
        txt = subprocess.run(["pdftotext", "-layout", os.path.join(FELLOWS, "IEEE-fellow-2026.pdf"), "-"], capture_output=True, text=True).stdout
        lines = [l.strip() for l in txt.splitlines() if not re.search(r"IEEE Fellows Class 2026 \| ieee\.org", l)]
        rows, i = [], 0
        while i < len(lines):
            if re.match(r"(?i)^for\b", lines[i]):
                head = [l for l in lines[max(0, i - 6):i] if l][-3:]
                cit = [lines[i]]; j = i + 1
                while j < len(lines) and lines[j]: cit.append(lines[j]); j += 1
                if len(head) == 3:
                    rows.append({"name": head[0], "affiliation": head[1], "location": head[2], "citation": " ".join(cit)})
                i = j
            else:
                i += 1
        return pd.DataFrame(rows)

    def load_acm():
        """ACM Fellows announced in January 2025 ("2024 ACM Fellows") and January 2026 ("2025 ACM Fellows"): name, affiliation
    and citation from ACM's announcements (inputs/fellows/ACM_Fellows_<class>_announcement.txt, copied from acm.org),
    world region from inputs/fellows/ACM Fellows.csv."""
        C = pd.read_csv(os.path.join(FELLOWS, "ACM Fellows.csv"), dtype=str).fillna("")
        key = lambda n: (split_name(n)[0], (split_name(n)[1] or [""])[0][:1])
        out = []
        for cls, announced in (("2024", 2025), ("2025", 2026)):
            text = open(os.path.join(FELLOWS, f"ACM_Fellows_{cls}_announcement.txt"), encoding="utf-8").read()
            blocks = [b.strip().split("\n") for b in text.split("\n\n") if b.strip()]
            A = pd.DataFrame([{"name": b[0].strip(), "affiliation": b[1].strip(), "citation": " ".join(x.strip() for x in b[2:])} for b in blocks])
            c = C[C.Year.str.strip() == cls]
            region = {key(" ".join(reversed([x.strip() for x in n.split(",", 1)]))): r for n, r in zip(c.Name, c.Region)}
            A["region"] = A.name.map(lambda n: region.get(key(n), "")); A["announced"] = announced
            out.append(A)
        A = pd.concat(out, ignore_index=True)
        A.loc[A.affiliation == "NA", "affiliation"] = ""
        return A

    US_STATES = {"USA", "United States"}
    def ieee_countries(location):
        last = location.split(",")[-1].strip()
        if last in US_STATES: return {"US"}
        m = {"Canada": "CA", "United Kingdom": "GB", "China": "CN", "Hong Kong": "HK", "Taiwan": "TW", "Korea, Republic of": "KR",
             "Japan": "JP", "Germany": "DE", "France": "FR", "Italy": "IT", "Spain": "ES", "Switzerland": "CH", "Australia": "AU",
             "India": "IN", "Singapore": "SG", "Israel": "IL", "Sweden": "SE", "Denmark": "DK", "Netherlands": "NL", "Belgium": "BE",
             "Austria": "AT", "Finland": "FI", "Norway": "NO", "Poland": "PL", "Portugal": "PT", "Ireland": "IE", "Romania": "RO",
             "Greece": "GR", "Saudi Arabia": "SA", "United Arab Emirates": "AE", "Mexico": "MX", "Brazil": "BR", "New Zealand": "NZ",
             "Iran, Islamic Republic of": "IR", "Egypt": "EG", "Turkey": "TR", "Qatar": "QA"}
        loc = location.strip()
        for k, v in m.items():
            if loc.endswith(k): return {v}
        return set()

    ACM_REGIONS = {"North America": {"US", "CA", "MX"},
                   "Europe": set(EUROPE.split()),
                   "Asia": set(ASIA_PACIFIC.split()) - {"AU", "NZ", "FJ", "PG"},
                   "Pacific Rim": {"AU", "NZ", "SG", "HK", "TW", "JP", "KR", "CN", "MY", "TH", "ID", "PH", "VN", "FJ", "PG"},
                   "Middle East": {"AE", "BH", "IL", "IQ", "IR", "JO", "KW", "LB", "OM", "PS", "QA", "SA", "SY", "YE", "EG", "TR"},
                   "Latin America": set(LATAM.split()), "Africa": set(MIDEAST_AFRICA.split())}

    def load():
        os.makedirs(OUT, exist_ok=True)
        I = load_ieee(); I["citation"] = I.citation.str.replace("\xa0", " ").str.strip()
        I["cs_related"] = I.citation.map(lambda c: bool(CS_CITATION.search(c)) and not NOT_CS_CITATION.search(c))
        I["group"] = "IEEE Fellow 2026"; I["announced"] = 2025
        A = load_acm(); A["cs_related"] = True; A["group"] = "ACM Fellow 2025-26"
        L = pd.concat([I, A], ignore_index=True); L.to_csv(os.path.join(OUT, "fellows_INTERNAL.csv"), index=False)
        print(f"IEEE class of 2026: {len(I)} Fellows, computer science-related by citation {int(I.cs_related.sum())}; ACM announced 2025 and 2026: {len(A)} "
              f"({A.announced.value_counts().to_dict()}); without region {int((A.region == '').sum())}")

    # ---------- 2. matching ----------
    def inst_overlap(affiliation, author):
        names = profiles.institution_names(author)
        ov = len(profiles.tokens(affiliation) & set().union(*[profiles.tokens(n) for n in names] or [set()]))
        k = profiles.inst_key(affiliation)
        for n in names:
            nk = profiles.inst_key(n)
            if k and nk and (k == nk or (len(nk) > 12 and nk in k) or (len(k) > 12 and k in nk)): return ov + 2
        return ov

    def resolve(f):
        surname, given = split_name(f["name"])
        res = openalex_get("authors", {"search": f["name"], "per-page": 50, "select": AUTHOR_FIELDS}).get("results", [])
        scored = []
        for a in res:
            if not name_ok(surname, given, a) or (a.get("works_count") or 0) < 10: continue
            countries = author_countries(a)
            places = ieee_countries(f["location"]) if f["group"].startswith("IEEE") else ACM_REGIONS.get(f["region"], set())
            inst = inst_overlap(f["affiliation"], a) if f["affiliation"] else 0
            place = bool(countries & places); topic = len(words(f["citation"]) & topic_words(a))
            score = 3 * min(inst, 3) + 2 * min(topic, 3) + 2 * place + np.log10(max(a.get("works_count", 1), 1))
            evidence = inst > 0 or (place and topic > 0)
            scored.append((score, evidence, inst, topic, place, a))
        if not scored: return None, "not found", {}
        scored.sort(key=lambda s: s[0], reverse=True)
        score, ev, inst, topic, place, best = scored[0]
        rivals = [s for s in scored[1:] if s[1] and s[0] > score - 1.5]
        status = "weak (no institution, place or field evidence)" if not ev else "ambiguous" if rivals else "ok"
        return best, status, {"institution overlap": inst, "topic overlap": topic, "place match": place, "candidates": len(scored)}

    def match():
        L = pd.read_csv(os.path.join(OUT, "fellows_INTERNAL.csv")).fillna("")
        hcr_ids = set(pd.read_csv(os.path.join(HERE, "hcr_study", "resolution.csv"))["OpenAlex ID"].dropna())
        hist = os.path.join(HERE, "hcr_study", "history", "resolution.csv")
        ever_ids = set(pd.read_csv(hist)["OpenAlex ID"].dropna()) if os.path.exists(hist) else set()
        rows = []
        for i, f in L[L.cs_related.astype(bool)].iterrows():
            a, status, ev = resolve(f)
            r = {**f.to_dict(), "status": status, **ev}
            if a is not None:
                aid = short_id(a["id"])
                r.update({"OpenAlex ID": aid, "OpenAlex name": a["display_name"], "works": a.get("works_count"), "cited by": a.get("cited_by_count"),
                          "HCR-listed 2025": aid in hcr_ids, "HCR-listed 2014-2025": aid in hcr_ids or aid in ever_ids})
            rows.append(r); time.sleep(0.1)
        M = pd.DataFrame(rows); M.to_csv(os.path.join(OUT, "matches_INTERNAL.csv"), index=False)
        print(M.groupby(["group", "status"]).size().to_string())

    # ---------- 2b. academic or not, the same rule for every group ----------
    ROR_CACHE = os.path.join(OUT, "ror_cache.json")
    # a university by name counts as academic whatever its registry type (ROR and OpenAlex type some universities as facilities)
    ACADEMIC_NAME = re.compile(r"universit|college|institute of technology|polytechnic|politecnico|[ée]cole", re.I)
    def ror_lookup(aff, cache):
        aff = re.sub(r"\s+", " ", str(aff)).strip()
        if not aff or aff == "NA": return None
        if aff in cache: return cache[aff]
        r = None
        for attempt in range(4):
            try:
                r = requests.get("https://api.ror.org/v2/organizations", params={"affiliation": aff}, timeout=30)
                if r.status_code == 200: break
            except requests.RequestException:
                pass
            time.sleep(2 ** attempt)
        items = r.json().get("items", []) if r is not None and r.status_code == 200 else []
        chosen = [it for it in items if it.get("chosen")]
        out = None
        if chosen:
            org = chosen[0]["organization"]
            out = {"name": next((n["value"] for n in org.get("names", []) if "ror_display" in n.get("types", [])), ""), "types": org.get("types", [])}
        cache[aff] = out; time.sleep(0.15)
        return out

    def affiliation_orgs(aff, cache):
        """ROR organizations of a printed affiliation; joint ones ('X and Y', 'X/ Y') are looked up part by part."""
        whole = ror_lookup(aff, cache)
        if whole: return [whole]
        parts = [x.strip() for x in re.split(r"\s+and\s+|/|;", str(aff)) if x.strip()]
        return [o for o in (ror_lookup(x, cache) for x in parts) if o] if len(parts) > 1 else []

    def academic():
        """A researcher is academic if the printed affiliation (IEEE and ACM announcements, Clarivate's primary affiliation,
    the Stanford-Elsevier institution) names a university, college, institute of technology or polytechnic, or is an
    education organization in ROR; where ROR has no match, if OpenAlex types one of the researcher's last known
    institutions as education."""
        cache = json.load(open(ROR_CACHE)) if os.path.exists(ROR_CACHE) else {}
        M = pd.read_csv(os.path.join(OUT, "matches_INTERNAL.csv")).fillna(""); M = M[M.status.isin(USABLE)]
        res = pd.read_csv(os.path.join(HERE, "hcr_study", "resolution.csv"))
        per = pd.read_csv(os.path.join(HERE, "hcr_study", "per_member.csv")); per = per[per["Consistent with Scopus"]]
        res = res[res["OpenAlex ID"].isin(per["OpenAlex ID"])]
        ref = pd.read_csv(os.path.join(HERE, "hcr_study", "reference", "sample.csv")).dropna(subset=["openalex_id"])
        rows = ([{"group": g, "OpenAlex ID": a, "affiliation": f} for g, a, f in zip(M.group, M["OpenAlex ID"], M.affiliation)] +
                [{"group": "HCR-listed 2025", "OpenAlex ID": a, "affiliation": f} for a, f in zip(res["OpenAlex ID"], res["HCR affiliation"])] +
                [{"group": "Reference sample", "OpenAlex ID": a, "affiliation": f} for a, f in zip(ref.openalex_id, ref.inst_name)])
        for k, r in enumerate(rows, 1):
            orgs = affiliation_orgs(r["affiliation"], cache)
            r["ror"] = "; ".join(f"{o['name']} ({'/'.join(o['types'])})" for o in orgs)
            r["academic (printed affiliation)"] = (any("education" in o["types"] for o in orgs)) if orgs else None
            if k % 100 == 0: json.dump(cache, open(ROR_CACHE, "w")); print(f"  ROR {k}/{len(rows)}", flush=True)
        json.dump(cache, open(ROR_CACHE, "w"))
        ids = sorted({r["OpenAlex ID"] for r in rows}); types = {}
        for i in range(0, len(ids), 50):
            for a in openalex_get("authors", {"filter": "openalex_id:" + "|".join(ids[i:i + 50]), "per-page": 50,
                                              "select": "id,last_known_institutions"}).get("results", []):
                types[short_id(a["id"])] = [x.get("type") for x in (a.get("last_known_institutions") or [])]
        for r in rows:
            t = types.get(r["OpenAlex ID"], [])
            r["openalex types"] = "/".join(x or "" for x in t)
            r["academic (OpenAlex)"] = ("education" in t) if t else None
            named = bool(ACADEMIC_NAME.search(str(r["affiliation"])))
            r["academic"] = True if named else r["academic (printed affiliation)"] if r["academic (printed affiliation)"] is not None else r["academic (OpenAlex)"]
        T = pd.DataFrame(rows); T.to_csv(os.path.join(OUT, "academic_INTERNAL.csv"), index=False)
        both = T.dropna(subset=["academic (printed affiliation)", "academic (OpenAlex)"])
        print(T.groupby("group").agg(n=("academic", "size"), academic=("academic", lambda x: int((x == True).sum())),
                                     ror_matched=("academic (printed affiliation)", lambda x: int(x.notna().sum()))).to_string())
        print(f"agreement of the printed-affiliation and OpenAlex rules: {100 * (both['academic (printed affiliation)'] == both['academic (OpenAlex)']).mean():.1f}% of {len(both)}")

    # ---------- 3. profiles and abstracts ----------
    def fetch():
        profiles.PROFILE_DIR = PROFILES; os.makedirs(PROFILES, exist_ok=True)
        M = pd.read_csv(os.path.join(OUT, "matches_INTERNAL.csv")); M = M[M.status.isin(USABLE)].drop_duplicates("OpenAlex ID")
        for k, aid in enumerate(M["OpenAlex ID"], 1):
            df = profiles.fetch_profile(aid)
            if k % 20 == 0: print(f"  profiles {k}/{len(M)}", flush=True)
        # same abstract sources as the HCR-listed researchers: Semantic Scholar, then Scopus and Springer Nature
        env = dict(os.environ, PROFILE_DIRS="fellows2026/profiles")
        py = sys.executable
        subprocess.run([py, os.path.join(HERE, "ground_truth.py"), "s2-abstracts"], env=env, check=False)
        subprocess.run([py, os.path.join(HERE, "ground_truth.py"), "publisher-abstracts"], env=env, check=False)
        print("FETCH DONE", flush=True)

    def career_start(df):
        y = pd.to_numeric(df.year, errors="coerce").dropna().sort_values()
        return int(y.iloc[1]) if len(y) > 1 else np.nan

    def person(df, out):
        m, survey, cites = profiles.author_metrics(df, out["Category"].values)
        cat = out["Category"].values; s, r = cat == "survey", cat == "research"
        D = cites[s].mean() / cites[r].mean() if s.any() and r.any() and cites[r].mean() > 0 else np.nan
        return {**m, "Dividend D": D, "career start": career_start(df), "R_h %": -m["h change %"]}

    # ---------- 4. classification ----------
    def classify():
        os.makedirs(CLASSIFIED, exist_ok=True)
        M = pd.read_csv(os.path.join(OUT, "matches_INTERNAL.csv")); M = M[M.status.isin(USABLE)]
        rows = []
        for k, r in enumerate(M.itertuples(index=False), 1):
            df = pd.read_csv(os.path.join(PROFILES, f"{r[M.columns.get_loc('OpenAlex ID')]}.csv")); df["citationCount"] = df["citationCount"].fillna(0).astype(int)
            out, _ = profiles.classify_papers(df); out.to_csv(os.path.join(CLASSIFIED, f"{r[M.columns.get_loc('OpenAlex ID')]}.csv"), index=False)
            rows.append({"group": r[M.columns.get_loc("group")], "OpenAlex ID": r[M.columns.get_loc("OpenAlex ID")],
                         "announced": r[M.columns.get_loc("announced")],
                         "HCR-listed 2014-2025": r[M.columns.get_loc("HCR-listed 2014-2025")], **person(df, out)})
            if k % 20 == 0: print(f"  classified {k}/{len(M)}", flush=True)
        # HCR-listed researchers (Scopus-consistent profiles), from profiles.py classify
        per = pd.read_csv(os.path.join(HERE, "hcr_study", "per_member.csv")); per = per[per["Consistent with Scopus"]]
        for aid in per["OpenAlex ID"]:
            df = pd.read_csv(os.path.join(HERE, "hcr_study", "profiles", f"{aid}.csv")); df["citationCount"] = df["citationCount"].fillna(0).astype(int)
            out = pd.read_csv(os.path.join(HERE, "hcr_study", "classified", f"{aid}.csv"))
            rows.append({"group": "HCR-listed 2025", "OpenAlex ID": aid, "HCR-listed 2014-2025": True, **person(df, out)})
        # Stanford-Elsevier reference sample, for the expectation model
        for f in glob.glob(os.path.join(HERE, "hcr_study", "reference", "classified", "*.csv")):
            aid = os.path.basename(f)[:-4]; pf = os.path.join(HERE, "hcr_study", "reference", "profiles", f"{aid}.csv")
            if not os.path.exists(pf): continue
            df = pd.read_csv(pf); df["citationCount"] = df["citationCount"].fillna(0).astype(int); out = pd.read_csv(f)
            rows.append({"group": "Reference sample", "OpenAlex ID": aid, "HCR-listed 2014-2025": False, **person(df, out)})
        pd.DataFrame(rows).to_csv(os.path.join(OUT, "per_person_INTERNAL.csv"), index=False); print("CLASSIFY DONE")

    # ---------- 5. report ----------
    def compare(Dx, title):
        """Medians, Holm-corrected Mann-Whitney tests and citation-adjusted survey dependence for one scope."""
        from scipy.stats import mannwhitneyu
        from sklearn.linear_model import QuantileRegressor
        C = "Survey % of citations (classified)"
        logit = lambda x: np.log(np.clip(x, 0.5, 99.5) / (100 - np.clip(x, 0.5, 99.5)))
        X = lambda d: np.c_[np.log10(d.Citations.clip(lower=1)), d["career start"].fillna(d["career start"].median())]
        ref = Dx[Dx.group == "Reference sample"].dropna(subset=[C])
        q = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(X(ref), logit(ref[C]))
        r0 = logit(ref[C]) - q.predict(X(ref)); sd = 1.4826 * np.median(np.abs(r0 - np.median(r0)))
        Dx = Dx.assign(z=(logit(Dx[C]) - q.predict(X(Dx))) / sd, expected=100 / (1 + np.exp(-q.predict(X(Dx)))))
        groups = ["HCR-listed 2025", "ACM Fellow 2025-26", "IEEE Fellow 2026", "Reference sample"]
        cols = ["Survey % of classified papers", C, "Dividend D", "R_h %", "Citations", "h-index", "career start", "expected", "z"]
        med = lambda d, c: f"{d[c].median():.1f} [{d[c].quantile(.25):.1f}-{d[c].quantile(.75):.1f}]"
        lines = [f"## {title}", "", "| median [IQR] | " + " | ".join(f"{g} (n={int((Dx.group == g).sum())})" for g in groups) + " |",
                 "|---|" + "---|" * len(groups)]
        lines += [f"| {c} | " + " | ".join(med(Dx[Dx.group == g], c) for g in groups) + " |" for c in cols]
        tests = []
        for g in groups[1:3]:
            for c in ["Survey % of classified papers", C, "Dividend D", "z"]:
                a, b = Dx[Dx.group == "HCR-listed 2025"][c].dropna(), Dx[Dx.group == g][c].dropna()
                u = mannwhitneyu(a, b); tests.append((g, c, u.pvalue, u.statistic / (len(a) * len(b))))
        order = np.argsort([t[2] for t in tests]); adj = np.empty(len(tests)); run = 0
        for rank, k in enumerate(order):
            run = max(run, min(1, (len(tests) - rank) * tests[k][2])); adj[k] = run
        lines += ["", "Mann-Whitney, HCR-listed against each Fellow group (two-sided, Holm across the 8 tests):", ""]
        lines += [f"- {g}, {c}: p = {p:.2g} (Holm {pa:.2g}); P(HCR-listed > Fellow) = {auc:.2f}" for (g, c, p, auc), pa in zip(tests, adj)]
        lo = Dx[Dx.group == "HCR-listed 2025"].Citations.quantile(0.1)
        lines += ["", f"Within the HCR-listed researchers' citation range (at least their 10th percentile, {lo:,.0f} citations):", ""]
        for g in groups[:3]:
            d = Dx[(Dx.group == g) & (Dx.Citations >= lo)]
            lines.append(f"- {g}: n = {len(d)}, survey % of citations {d[C].median():.1f}, dividend {d['Dividend D'].median():.1f}, "
                         f"survey % of papers {d['Survey % of classified papers'].median():.1f}")
        return lines, Dx

    def report():
        D = pd.read_csv(os.path.join(OUT, "per_person_INTERNAL.csv"))
        M = pd.read_csv(os.path.join(OUT, "matches_INTERNAL.csv"))
        A = pd.read_csv(os.path.join(OUT, "academic_INTERNAL.csv"))[["group", "OpenAlex ID", "academic"]].drop_duplicates(["group", "OpenAlex ID"])
        D = D.merge(A, on=["group", "OpenAlex ID"], how="left")
        D["also HCR"] = D["HCR-listed 2014-2025"].fillna(False).astype(bool) & D.group.str.contains("Fellow")
        Dx = D                       # people on more than one list stay in every group they belong to (overlaps are counted)
        head = ["# Peer-elected comparison: IEEE and ACM Fellows elected for 2026", "",
                "Matching: " + "; ".join(f"{g}: " + ", ".join(f"{k} {v}" for k, v in d.status.value_counts().items()) for g, d in M.groupby("group")),
                "Fellows also HCR-listed in 2014-2025 (kept in both groups): " +
                ", ".join(f"{g} {int(d['also HCR'].sum())}" for g, d in D[D.group.str.contains('Fellow')].groupby('group')),
                "Academic and kept (ROR education type of the printed affiliation, else OpenAlex): " +
                ", ".join(f"{g} {int((d.academic == True).sum())} of {len(d)}" for g, d in Dx.groupby("group")), ""]
        main, scored = compare(Dx[Dx.academic == True], "Academic researchers only (non-academic researchers dropped from every group)")
        from scipy.stats import mannwhitneyu
        acm = scored[scored.group == "ACM Fellow 2025-26"]; C = "Survey % of citations (classified)"
        a25, a26 = acm[acm.announced == 2025][C].dropna(), acm[acm.announced == 2026][C].dropna()
        both = set(M[(M.group == "ACM Fellow 2025-26") & (M.status == "ok")]["OpenAlex ID"]) & set(M[(M.group == "IEEE Fellow 2026") & (M.status == "ok")]["OpenAlex ID"])
        lines = head + main + ["", f"ACM classes: announced 2025 (n={len(a25)}) median survey % of citations {a25.median():.1f}, announced 2026 "
                                    f"(n={len(a26)}) {a26.median():.1f}, Mann-Whitney p = {mannwhitneyu(a25, a26).pvalue:.2g}",
                               f"Fellows of both societies (kept in both Fellow groups): {len(both)}"]
        open(os.path.join(OUT, "report.md"), "w").write("\n".join(lines)); print("\n".join(lines))
        scored.to_csv(os.path.join(OUT, "per_person_scored_INTERNAL.csv"), index=False)

    step = sys.argv[1] if len(sys.argv) > 1 else "all"
    for s in (["load", "match", "academic", "fetch", "classify", "report"] if step == "all" else [step]):
        {"load": load, "match": match, "academic": academic, "fetch": fetch, "classify": classify, "report": report}[s]()


STEPS = {"controls": controls, "reference": reference, "any-list": any_list, "peers": peers, "boundary": boundary, "fellows": fellows}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv.pop(1)]()
