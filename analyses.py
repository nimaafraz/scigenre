"""Stage 6: the analyses behind the paper's numbers (outputs in hcr_study/).

    python analyses.py survey-outliers      survey dependence of every group
    python analyses.py yearly-citations     citations per year of every work (OpenAlex)
    python analyses.py outlier-model        RQ4 expectation model
    python analyses.py outlier-checks       checks of the expectation model
    python analyses.py composite [figure]   Stanford-Elsevier composite without surveys
    python analyses.py boundary-scenarios   sensitivity of the boundary simulation
    python analyses.py primary-endpoints    the primary tests of RQ2-RQ4
    python analyses.py s2-types hcr         Semantic Scholar records of the HCR-listed researchers' papers
    python analyses.py s2-rule              Smyth & Cunningham's rule on those records
    python analyses.py dividend-window      survey citation dividend by publication cohort
    python analyses.py keyword-rules        survey dependence under keyword rules
    python analyses.py abstract-sources     does the abstract source change the classification?
    python analyses.py paper-numbers        remaining numbers quoted in the paper
"""
import glob
import json
import os
import re
import sys
import time
import unicodedata

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
import statsmodels.formula.api as smf
from dotenv import load_dotenv
from scipy.stats import binomtest, fisher_exact, kendalltau, mannwhitneyu, spearmanr, wilcoxon
from sklearn.linear_model import QuantileRegressor
from sklearn.metrics import cohen_kappa_score

import common
import comparison_groups
import profiles
import rq1_evaluation
from common import (
    boxes, calculate_indices, clean_text, COLORS, flag, INK, INK_2, is_non_paper, LISTED, NAMES, openalex_get_all,
    panel, PAPER_TITLE_KEYWORDS, POINT, prepare_frame, r_handle, r_sets, REFERENCE, save, SPRINGER, style, WIDE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# Survey dependence of every group
# =====================================================================================================================

def survey_outliers():
    """Survey-dependence outliers among researchers listed in the HCR (Computer Science).

    Metric (per researcher): survey dependence = the share of an indicator that
    disappears when surveys are excluded:
        C  citations          (share of citations from surveys)
        T  top-1% papers      (share of 2014-2024 top-1% papers that are surveys)
        H  h-index            (relative h-index drop)

    Reference distribution: the matched peers (up to five per HCR-listed researcher;
    comparison_groups.py peers).

    Outlier rule: robust z-score against the reference, z = (x - median) / (1.4826 MAD);
    an outlier has z >= 3. To be robust to classification error, each researcher's
    C is simulated paper by paper: a flagged survey is a survey with probability
    PPV, an unflagged paper with probability FOR (from the final model's recall and
    false-positive rate on the held-out test split, and the researcher's corrected survey share); a researcher is a
    "confident outlier" if the 5th percentile of simulated z is still >= 3.

    Decomposition: with survey paper share p and survey citation dividend D
    (mean citations per survey / per other paper), C = pD / (pD + 1 - p).

    Output: hcr_study/outliers/*.csv, figures in hcr_study/figures/."""
    OUT = os.path.join(STUDY, "outliers")
    FIG = os.path.join(STUDY, "figures")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    _CFG = json.load(open(os.path.join(HERE, "model", "scigenre.json")))
    TPR, FPR = _CFG["test_recall"] / 100, _CFG["test_fpr"] / 100     # final model, held-out test split
    Z_OUT = 3.0
    N_SIM = 2000
    RNG = np.random.default_rng(42)
    HCP_START, HCP_END = 2014, 2024

    SERIES, SERIES_2, BEFORE = "#2a78d6", "#eb6834", "#a3a29c"
    INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    matplotlib.rc_file_defaults()
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6, "legend.frameon": False,
    })


    def h_index(c):
        c = np.sort(np.asarray(c))[::-1]
        return int(np.sum(c >= np.arange(1, len(c) + 1)))


    def profile(path):
        p = pd.read_csv(path, usecols=["id", "citationCount", "Category", "year", "top1"])
        c = p["citationCount"].fillna(0).astype(int).values
        s = (p["Category"] == "survey").values
        other = (p["Category"] != "survey").values
        top1 = p["top1"].fillna(False).astype(bool).values & p["year"].between(HCP_START, HCP_END).values
        h = h_index(c)
        q = s.mean()
        # simulate classification error: P(survey | flagged) = PPV, P(survey | not flagged) = FOR
        pi = float(np.clip((q - FPR) / (TPR - FPR), 0, 1))
        ppv = min(TPR * pi / q, 1) if q > 0 else 0.0
        fom = min((1 - TPR) * pi / (1 - q), 1) if q < 1 else 0.0
        prob = np.where(s, ppv, fom)
        sims = RNG.random((N_SIM, len(c))) < prob
        total = max(c.sum(), 1)
        c_sim = 100 * (sims * c).sum(1) / total
        mean_s = c[s].mean() if s.any() else np.nan
        mean_o = c[other].mean() if other.any() else np.nan
        return {
            "papers": len(p), "p": 100 * q,
            "C": 100 * c[s].sum() / total,
            "C_sim": c_sim,
            "T": 100 * (top1 & s).sum() / top1.sum() if top1.any() else np.nan,
            "H": 100 * (h - h_index(c[~s])) / max(h, 1),
            "D": mean_s / mean_o if mean_s == mean_s and mean_o else np.nan,
            "top1": int(top1.sum()), "top1_x": int((top1 & ~s).sum()),
        }


    def load():
        rows = []
        per = pd.read_csv(os.path.join(STUDY, "per_member.csv"))
        per = per[per["Consistent with Scopus"]]
        for _, r in per.iterrows():
            rows.append({"group": "listed", "key": r["HCR name"], "id": r["OpenAlex ID"],
                         **profile(os.path.join(STUDY, "classified", f"{r['OpenAlex ID']}.csv"))})
        for _, r in comparison_groups.load_peers().iterrows():
            rows.append({"group": "non-listed", "key": r["HCR name"], "id": r["OpenAlex ID"], **profile(r["path"])})
        return pd.DataFrame(rows)


    def robust_z(x, ref):
        med = np.nanmedian(ref)
        mad = 1.4826 * np.nanmedian(np.abs(ref - med))
        return (x - med) / mad, med, mad


    def main():
        d = load()
        ref = d[d["group"] == "non-listed"]
        listed = d[d["group"] == "listed"]
        stats = []

        for m, label in (("C", "citations"), ("T", "top-1% papers"), ("H", "h-index")):
            z, med, mad = robust_z(d[m].values, ref[m].values)
            d[f"z_{m}"] = z
            stats.append((f"reference median / scaled MAD, {label}", f"{med:.1f} / {mad:.1f}"))

        # classifier-error-aware z for C
        med, mad = np.nanmedian(ref["C"]), 1.4826 * np.nanmedian(np.abs(ref["C"] - np.nanmedian(ref["C"])))
        d["z_C_p05"] = [np.percentile((cs - med) / mad, 5) for cs in d["C_sim"]]
        d["z_C_p95"] = [np.percentile((cs - med) / mad, 95) for cs in d["C_sim"]]

        for g in ("listed", "non-listed"):
            e = d[d["group"] == g]
            for m in ("C", "T", "H"):
                stats.append((f"{g}: z_{m} >= 3", int((e[f"z_{m}"] >= Z_OUT).sum())))
            stats.append((f"{g}: confident outliers on citations (5th pct of z_C >= 3)", int((e["z_C_p05"] >= Z_OUT).sum())))
            stats.append((f"{g}: n", len(e)))

        # enrichment of outliers on the list (Fisher exact, one-sided)
        for m in ("C", "T"):
            a = int((listed[f"z_{m}"] >= Z_OUT).sum()) if f"z_{m}" in listed else 0
            a = int((d.loc[d.group == "listed", f"z_{m}"] >= Z_OUT).sum())
            b = int((d.loc[d.group == "non-listed", f"z_{m}"] >= Z_OUT).sum())
            n1, n2 = int((d.group == "listed").sum()), int((d.group == "non-listed").sum())
            odds, p = fisher_exact([[a, n1 - a], [b, n2 - b]], alternative="greater")
            stats.append((f"enrichment on the list, z_{m} >= 3: listed {a}/{n1} vs non-listed {b}/{n2}", f"odds ratio {odds:.2f}, one-sided p = {p:.3f}"))

        d["outlier_C"] = d["z_C"] >= Z_OUT
        d["confident_outlier_C"] = d["z_C_p05"] >= Z_OUT
        keep = ["group", "key", "id", "papers", "p", "D", "C", "T", "H", "z_C", "z_C_p05", "z_C_p95", "z_T", "z_H",
                "top1", "top1_x", "outlier_C", "confident_outlier_C"]
        d[keep].to_csv(os.path.join(OUT, "survey_dependence_INTERNAL.csv"), index=False)

        # anonymized table of listed outliers
        out = d[(d.group == "listed") & ((d["z_C"] >= Z_OUT) | (d["z_T"] >= Z_OUT))].sort_values("z_C", ascending=False)
        out = out.reset_index(drop=True)
        out.insert(0, "label", [f"O{i + 1}" for i in range(len(out))])
        out[["label", "key", "id"]].to_csv(os.path.join(OUT, "outlier_key_INTERNAL.csv"), index=False)
        pub = out[["label", "p", "D", "C", "T", "H", "z_C", "z_C_p05", "z_T", "z_H", "confident_outlier_C"]].round(2)
        pub.to_csv(os.path.join(OUT, "listed_outliers_anonymized.csv"), index=False)

        # ---------------- Figure: decomposition p x D with iso-lines of C ----------------
        fig, ax = plt.subplots(figsize=(3.5, 2.9), constrained_layout=True)
        pp = np.linspace(0.005, 0.6, 300)
        for Cv in (10, 25, 50, 75):
            c = Cv / 100
            Dv = c * (1 - pp) / (pp * (1 - c))
            ax.plot(100 * pp, Dv, color=GRID, linewidth=1, zorder=1)
            i = np.argmin(np.abs(Dv - 30)) if Cv >= 50 else np.argmin(np.abs(100 * pp - 50))
            ax.text(100 * pp[i], Dv[i], f"C = {Cv}%", fontsize=6, color=INK_2, ha="left", va="bottom")
        nl = d[d.group == "non-listed"]
        li = d[d.group == "listed"]
        ax.scatter(nl["p"], nl["D"], s=12, color=BEFORE, alpha=0.8, linewidths=0, label="Non-listed (matched)", zorder=2)
        ax.scatter(li["p"], li["D"], s=12, color=SERIES, alpha=0.8, linewidths=0, label="Listed", zorder=3)
        o = li[li["z_C"] >= Z_OUT]
        ax.scatter(o["p"], o["D"], s=34, facecolors="none", edgecolors=SERIES_2, linewidths=1.2,
                   label=f"Listed outlier (z $\\geq$ {Z_OUT:.0f})", zorder=4)
        ax.set_yscale("log")
        ax.set_xlim(0, 45)
        ax.set_ylim(0.3, 40)
        ax.set_xlabel("Surveys as % of papers (p)")
        ax.set_ylabel("Survey citation dividend D (log)")
        ax.legend(loc="lower right", fontsize=6.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.savefig(os.path.join(FIG, "fig_outlier_decomposition.pdf"))
        fig.savefig(os.path.join(FIG, "fig_outlier_decomposition.png"), dpi=300)
        plt.close(fig)

        # ---------------- Figure: robust z with classifier-error intervals ----------------
        fig, ax = plt.subplots(figsize=(3.5, 2.6), constrained_layout=True)
        for g, color, xpos in (("non-listed", BEFORE, 0), ("listed", SERIES, 1)):
            e = d[d.group == g].sort_values("z_C").reset_index(drop=True)
            x = xpos + np.linspace(-0.35, 0.35, len(e))
            ax.vlines(x, e["z_C_p05"], e["z_C_p95"], color=color, alpha=0.35, linewidth=1)
            ax.scatter(x, e["z_C"], s=7, color=color, zorder=3)
        ax.axhline(Z_OUT, color=SERIES_2, linewidth=0.9, linestyle="--")
        ax.text(1.42, Z_OUT + 0.15, f"z = {Z_OUT:.0f}", fontsize=6.5, color=SERIES_2, ha="right")
        ax.set_xticks([0, 1], [f"Non-listed\n(matched, n = {len(ref)})", f"Listed\n(n = {len(listed)})"])
        ax.set_ylabel("Robust z of citation drop\n(vs non-listed researchers)")
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.savefig(os.path.join(FIG, "fig_outlier_z.pdf"))
        fig.savefig(os.path.join(FIG, "fig_outlier_z.png"), dpi=300)
        plt.close(fig)

        with open(os.path.join(OUT, "outlier_stats.md"), "w") as fh:
            for k, v in stats:
                fh.write(f"- {k}: {v}\n")
            fh.write("\n" + pub.to_markdown(index=False) + "\n")
        pd.set_option("display.width", 250)
        for k, v in stats:
            print(f"{k}: {v}")
        print("\nListed outliers (anonymized):")
        print(pub.to_string(index=False))


    main()


# =====================================================================================================================
# Citations per year of every work
# =====================================================================================================================

def yearly_citations():
    """Citations received per calendar year, per work (OpenAlex counts_by_year), for the
    2025 HCR-listed, matched non-listed and reference researchers and the academic
    ACM and IEEE Fellows of the Fellows comparison (comparison_groups.py fellows). Author-level counts_by_year
    in OpenAlex are unreliable (they fall ~70% from 2019 to 2024), so the yearly
    series is summed from the works instead.

        python analyses.py yearly-citations      (WORKER=k NWORKERS=n for parallel parts)

    Output: hcr_study/growth_survey/yearly/<OpenAlex ID>.csv (work id, year, citations)"""
    OUT = os.path.join(HERE, "hcr_study", "growth_survey", "yearly")


    def main():
        os.makedirs(OUT, exist_ok=True)
        sd = pd.read_csv(os.path.join(HERE, "hcr_study", "outliers", "survey_dependence_INTERNAL.csv"))
        ref = pd.read_csv(os.path.join(HERE, "hcr_study", "reference", "per_reference.csv"))
        ids = list(sd["id"].dropna()) + list(ref["openalex_id"].dropna())
        fel = os.path.join(HERE, "hcr_study", "fellows2026", "per_person_scored_INTERNAL.csv")
        if os.path.exists(fel):    # academic researchers of the Fellows comparison (analyses.py dividend-window)
            ids += list(pd.read_csv(fel)["OpenAlex ID"].dropna())
        k, n = int(os.environ.get("WORKER", 0)), int(os.environ.get("NWORKERS", 1))
        ids = list(dict.fromkeys(ids))[k::n]
        for i, aid in enumerate(ids, 1):
            path = os.path.join(OUT, f"{aid}.csv")
            if os.path.exists(path):
                continue
            works = openalex_get_all("works", {"filter": f"author.id:{aid}", "select": "id,counts_by_year"})
            rows = [{"id": w["id"], "year": y["year"], "citations": y["cited_by_count"]}
                    for w in works for y in w["counts_by_year"]]
            pd.DataFrame(rows, columns=["id", "year", "citations"]).to_csv(path, index=False)
            if i % 20 == 0:
                print(f"{i}/{len(ids)}", flush=True)
        print("done")


    main()


# =====================================================================================================================
# RQ4 expectation model
# =====================================================================================================================

def outlier_model():
    """Survey-dependence outliers relative to a reference population of top-cited
    computer-science researchers who are not listed in the HCR.

    Survey dependence C = share of a researcher's citations that come from surveys
    (the citation drop when surveys are excluded). Its expected value depends on the
    subfield, career stage, and citation level, so it is modeled on the reference
    sample by linear quantile regression of logit(C) on log10(Scopus citations),
    first publication year, and subfield group (quantiles 0.5 and 0.95).

    Outlier score: z = (logit C - q50(x)) / s, with s = 1.4826 MAD of the reference
    residuals; outlier if z >= 3. Classification error is propagated by simulating
    each paper's survey status (PPV / FOR from the final model's test-split TPR/FPR),
    and a researcher is a confident outlier if the 5th percentile of z is >= 3.

    Typology: an outlier is "volume" if its survey share of papers p exceeds the
    reference 95th percentile, "dividend" if its survey citation dividend D does,
    "both", or "neither" (the combination is unusual although each part is not).

    Output: hcr_study/outliers/model_*.csv|md and figures in hcr_study/figures/.

        python analyses.py outlier-model            # fit, statistics and figures
        python analyses.py outlier-model figure     # redraw the decomposition figure only"""
    OUT = os.path.join(STUDY, "outliers")
    FIG = os.path.join(STUDY, "figures")
    os.makedirs(OUT, exist_ok=True)
    os.makedirs(FIG, exist_ok=True)

    _CFG = json.load(open(os.path.join(HERE, "model", "scigenre.json")))
    TPR, FPR = _CFG["test_recall"] / 100, _CFG["test_fpr"] / 100     # final model, held-out test split
    Z_OUT = 3.0
    # Restrict to one subfield group ("nt", "ai", "other") or None for all; set with SUBFIELD=nt
    SUBSET = os.environ.get("SUBFIELD") or None
    N_SIM = 2000
    EPS = 0.5          # percent; keeps logit finite for C = 0 or 100
    RNG = np.random.default_rng(11)
    HCP_START, HCP_END = 2014, 2024

    SERIES, SERIES_2, BEFORE = "#2a78d6", "#eb6834", "#a3a29c"
    INK, INK_2, GRID = "#0b0b0b", "#52514e", "#e4e3df"
    plt.rcParams.update({
        "font.family": "serif", "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 8, "axes.edgecolor": INK_2, "axes.labelcolor": INK,
        "xtick.color": INK_2, "ytick.color": INK_2, "axes.linewidth": 0.6, "legend.frameon": False,
    })


    def logit_pct(c):
        c = np.clip(np.asarray(c, float), EPS, 100 - EPS) / 100
        return np.log(c / (1 - c))


    def inv_logit_pct(z):
        return 100 / (1 + np.exp(-z))


    def h_index(c):
        c = np.sort(np.asarray(c))[::-1]
        return int(np.sum(c >= np.arange(1, len(c) + 1)))


    def profile(path, rng=None):
        d = pd.read_csv(path, usecols=["citationCount", "Category", "year", "top1"])
        c = d["citationCount"].fillna(0).astype(int).values
        s = (d["Category"] == "survey").values
        top1 = d["top1"].fillna(False).astype(bool).values & d["year"].between(HCP_START, HCP_END).values
        q = s.mean()
        pi = float(np.clip((q - FPR) / (TPR - FPR), 0, 1))
        ppv = min(TPR * pi / q, 1) if q > 0 else 0.0
        fom = min((1 - TPR) * pi / (1 - q), 1) if q < 1 else 0.0
        sims = (rng or RNG).random((N_SIM, len(c))) < np.where(s, ppv, fom)
        total = max(c.sum(), 1)
        h = h_index(c)
        ms, mo = (c[s].mean() if s.any() else np.nan), (c[~s].mean() if (~s).any() else np.nan)
        return {"p": 100 * q, "D": ms / mo if mo else np.nan, "C": 100 * c[s].sum() / total,
                "C_sim": 100 * (sims * c).sum(1) / total,
                "T": 100 * (top1 & s).sum() / top1.sum() if top1.sum() >= 10 else np.nan,
                "H": 100 * (h - h_index(c[~s])) / max(h, 1), "top1": int(top1.sum())}


    def group_of(subfield):
        return {"Networking & Telecommunications": "nt", "Artificial Intelligence & Image Processing": "ai"}.get(subfield, "other")


    def load():
        rows = []
        # listed researchers
        per = pd.read_csv(os.path.join(STUDY, "per_member.csv"))
        per = per[per["Consistent with Scopus"]]
        els = pd.read_csv(os.path.join(STUDY, "BTAS_HCR_Computer_Science.csv"))
        els = els[els["Match status"] == "ok"].set_index("HCR name")
        for _, r in per.iterrows():
            if r["HCR name"] not in els.index:
                continue
            e = els.loc[r["HCR name"]]
            rows.append({"group": "listed", "key": r["HCR name"], "id": r["OpenAlex ID"],
                         "cites": e["nc9624 (ns)"], "firstyr": e["firstyr"], "sub": group_of(e["sm-subfield-1"]),
                         **profile(os.path.join(STUDY, "classified", f"{r['OpenAlex ID']}.csv"))})
        # one-to-one citation-matched peers (comparison_groups.py controls), a sensitivity check; loaded here so that the simulation draws
        # of the other groups stay as they were
        ctl = pd.read_csv(os.path.join(STUDY, "controls", "per_control.csv"))
        ctl = ctl[ctl["Consistent with Scopus"]]
        mt = pd.read_csv(os.path.join(STUDY, "controls", "matches.csv")).set_index("Control OpenAlex ID")
        for _, r in ctl.iterrows():
            m = mt.loc[r["Control OpenAlex ID"]]
            rows.append({"group": "matched_1to1", "key": r["Control authfull"], "id": r["Control OpenAlex ID"], "hcr": r["HCR name (matched member)"],
                         "cites": m["Control citations (Scopus ns)"], "firstyr": m["Control firstyr"],
                         "sub": group_of(m["Member subfield"]),
                         **profile(os.path.join(STUDY, "controls", "classified", f"{r['Control OpenAlex ID']}.csv"))})
        # reference sample
        ref = pd.read_csv(os.path.join(STUDY, "reference", "per_reference.csv"))
        ref = ref[ref["Consistent with Scopus"]]
        for _, r in ref.iterrows():
            path = os.path.join(STUDY, "reference", "classified", f"{r['openalex_id']}.csv")
            if os.path.exists(path):
                rows.append({"group": "reference", "key": r["authfull"], "id": r["openalex_id"],
                             "cites": r["scopus_cites"], "firstyr": r["firstyr"], "sub": group_of(r["subfield"]),
                             **profile(path)})
        # matched peers, up to five per HCR-listed researcher (comparison_groups.py peers), with their own random stream
        sub = els["sm-subfield-1"].to_dict()
        rng = np.random.default_rng(12)
        for _, r in comparison_groups.load_peers().iterrows():
            rows.append({"group": "matched", "key": r["authfull"], "id": r["OpenAlex ID"], "hcr": r["HCR name"],
                         "cites": r["nc9624 (ns)"], "firstyr": r["firstyr"], "sub": group_of(sub[r["HCR name"]]),
                         **profile(r["path"], rng)})
        return pd.DataFrame(rows)


    def design(df):
        X = pd.DataFrame({"logc": np.log10(df["cites"].astype(float)), "yr": (df["firstyr"] - 2000) / 10})
        if SUBSET is None:
            X["ai"] = (df["sub"] == "ai").astype(float)
            X["other"] = (df["sub"] == "other").astype(float)
        return X.values


    def main():
        d = load()
        if SUBSET:
            d = d[d["sub"] == SUBSET].reset_index(drop=True)
            print(f"Restricted to subfield group '{SUBSET}'")
        R = d[d.group == "reference"]
        print({g: int((d.group == g).sum()) for g in ("listed", "matched", "reference")})

        XR, yR = design(R), logit_pct(R["C"])
        q50 = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(XR, yR)
        q95 = QuantileRegressor(quantile=0.95, alpha=0, solver="highs").fit(XR, yR)
        resR = yR - q50.predict(XR)
        s = 1.4826 * np.median(np.abs(resR - np.median(resR)))

        X = design(d)
        d["C_expected"] = inv_logit_pct(q50.predict(X))
        d["C_q95"] = inv_logit_pct(q95.predict(X))
        d["z"] = (logit_pct(d["C"]) - q50.predict(X)) / s
        d["z_p05"] = [np.percentile((logit_pct(cs) - m) / s, 5) for cs, m in zip(d["C_sim"], q50.predict(X))]
        d["above_q95"] = d["C"] > d["C_q95"]
        d["outlier"] = d["z"] >= Z_OUT
        d["confident"] = d["z_p05"] >= Z_OUT
        p95, D95 = np.nanpercentile(R["p"], 95), np.nanpercentile(R["D"], 95)
        d["type"] = np.select([(d.p > p95) & (d.D > D95), d.p > p95, d.D > D95], ["both", "volume", "dividend"], "neither")

        lines = [f"Reference n = {len(R)}; residual scale s = {s:.3f} (logit); reference p95 of p = {p95:.1f}%, of D = {D95:.1f}",
                 f"Quantile-regression coefficients (q50) [log10 cites, (firstyr-2000)/10{'' if SUBSET else ', AI, other'}]: {np.round(q50.coef_, 3).tolist()}, intercept {q50.intercept_:.3f}"]
        for g in ("reference", "matched", "listed"):
            e = d[d.group == g]
            lines.append(f"{g}: n={len(e)}, above conditional q95: {int(e.above_q95.sum())} ({100 * e.above_q95.mean():.1f}%), "
                         f"z>=3: {int(e.outlier.sum())} ({100 * e.outlier.mean():.1f}%), confident: {int(e.confident.sum())}, "
                         f"median z {e.z.median():.2f}")
        R, L, M = d[d.group == "reference"], d[d.group == "listed"], d[d.group == "matched"]
        for name, A in (("reference", R), ("matched", M)):
            a, b = int(L.outlier.sum()), int(A.outlier.sum())
            orr, p = fisher_exact([[a, len(L) - a], [b, len(A) - b]], alternative="greater")
            lines.append(f"outliers listed vs {name}: {a}/{len(L)} vs {b}/{len(A)}: OR {orr:.2f}, one-sided p {p:.3f}")
            a, b = int(L.above_q95.sum()), int(A.above_q95.sum())
            orr, p = fisher_exact([[a, len(L) - a], [b, len(A) - b]], alternative="greater")
            lines.append(f"above q95 listed vs {name}: {a}/{len(L)} vs {b}/{len(A)}: OR {orr:.2f}, one-sided p {p:.3f}")
        lines.append(f"above q95, listed vs expected 5%: binomial one-sided p {binomtest(int(L.above_q95.sum()), len(L), 0.05, alternative='greater').pvalue:.4f}")
        lines.append(f"residual z, listed vs reference: Mann-Whitney one-sided p {mannwhitneyu(L.z, R.z, alternative='greater').pvalue:.4f}")
        lines.append(f"residual z, listed vs matched: Mann-Whitney one-sided p {mannwhitneyu(L.z, M.z, alternative='greater').pvalue:.4f}")
        for g in ("matched", "matched_1to1"):      # paired: each HCR-listed researcher against the mean of their peers
            pz = L.set_index("key").z.to_frame("listed").join(d[d.group == g].groupby("hcr").z.mean().rename("peers"), how="inner")
            dz = pz["listed"] - pz["peers"]
            lines.append(f"residual z, paired against {g} (mean per researcher): n={len(dz)}, median difference {dz.median():+.2f}, "
                         f"listed higher for {int((dz > 0).sum())}, Wilcoxon two-sided p {wilcoxon(dz).pvalue:.2g}")
        lines.append("outlier types (all groups): " + d[d.outlier].groupby(["group", "type"]).size().to_string().replace("\n", "; "))

        # Robustness: listed vs reference on several measures of survey dependence
        d["C_corrected"] = [np.mean(cs) for cs in d["C_sim"]]
        R, L, M = d[d.group == "reference"], d[d.group == "listed"], d[d.group == "matched"]
        for col, label in (("C", "survey % of citations"), ("C_corrected", "survey % of citations, error-corrected"),
                           ("T", "survey % of top-1% papers"), ("H", "h-index drop %"), ("p", "survey % of papers"),
                           ("D", "survey citation dividend"), ("z", "residual z")):
            a, b, c = L[col].dropna(), R[col].dropna(), M[col].dropna()
            lines.append(f"robustness, {label}: listed median {a.median():.2f} (n={len(a)}) vs reference {b.median():.2f} "
                         f"(n={len(b)}), two-sided p={mannwhitneyu(a, b).pvalue:.2g}; matched "
                         f"{c.median():.2f}, listed vs matched p={mannwhitneyu(a, c).pvalue:.2g}")
        lines.append(f"odds of survey citations per 10x Scopus citations: x{np.exp(q50.coef_[0]):.2f}; "
                     f"per decade later first publication: x{np.exp(q50.coef_[1]):.2f}")
        # Individual tests: empirical one-sided p against the reference residuals, Benjamini-Hochberg
        res_all = (logit_pct(d["C"]) - q50.predict(X))
        d["p_emp"] = [(1 + np.sum(resR >= r)) / (1 + len(resR)) for r in res_all]
        for g in ("listed", "matched"):
            e = d[d.group == g].sort_values("p_emp")
            m = len(e)
            q = np.minimum.accumulate((e["p_emp"].values * m / np.arange(1, m + 1))[::-1])[::-1]
            d.loc[e.index, "q_BH"] = np.minimum(q, 1)
            lines.append(f"{g}: smallest empirical p {e['p_emp'].min():.4f}; smallest BH q {np.min(q):.2f}; "
                         f"q < 0.05: {int((q < 0.05).sum())}, q < 0.10: {int((q < 0.10).sum())}")

        keep = ["group", "key", "id", "hcr", "cites", "firstyr", "sub", "p", "D", "C", "C_expected", "C_q95", "z", "z_p05",
                "above_q95", "outlier", "confident", "type", "T", "H", "top1"]
        d[keep].to_csv(os.path.join(OUT, "model_scores_INTERNAL.csv"), index=False)
        lo = d[(d.group == "listed") & d.outlier].sort_values("z", ascending=False).reset_index(drop=True)
        lo.insert(0, "label", [f"O{i + 1}" for i in range(len(lo))])
        lo[["label", "key", "id"]].to_csv(os.path.join(OUT, "model_outlier_key_INTERNAL.csv"), index=False)
        pub = lo[["label", "p", "D", "C", "C_expected", "z", "z_p05", "confident", "type", "T", "H"]].round(2)
        pub.to_csv(os.path.join(OUT, "model_listed_outliers_anonymized.csv"), index=False)
        with open(os.path.join(OUT, "model_stats.md"), "w") as fh:
            fh.write("\n".join(f"- {l}" for l in lines) + "\n\n" + pub.to_markdown(index=False) + "\n")
        print("\n".join(lines)); print(pub.to_string(index=False))

        # ---------- Figure A: observed vs expected, with the reference q95 band ----------
        fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
        xs = np.linspace(np.log10(5000), np.log10(100000), 100)
        base = np.column_stack([xs, np.full(100, 0.0)] + ([] if SUBSET else [np.zeros(100), np.zeros(100)]))   # first year 2000
        ax.plot(10 ** xs, inv_logit_pct(q50.predict(base)), color=INK_2, linewidth=1, label="Expected (median), reference")
        ax.plot(10 ** xs, inv_logit_pct(q95.predict(base)), color=INK_2, linewidth=1, linestyle="--", label="Reference 95th percentile")
        ax.scatter(R["cites"], R["C"], s=8, color=BEFORE, alpha=0.6, linewidths=0, label="Non-listed reference")
        rf0 = L["key"].isin(r_sets()["names"]).values
        ax.scatter(L["cites"][~rf0], L["C"][~rf0], s=10, color=SERIES, alpha=0.85, linewidths=0, label="HCR-listed")
        flag(ax, L["cites"][rf0], L["C"][rf0])
        o = L[L.outlier]
        if len(o):
            ax.scatter(o["cites"], o["C"], s=34, facecolors="none", edgecolors=SERIES_2, linewidths=1.2,
                       label="HCR-listed outlier (z $\\geq$ 3)")
        ax.set_xscale("log")
        ax.set_xlabel("Scopus non-self citations (log)")
        ax.set_ylabel("Citation drop without surveys (%)")
        ax.set_ylim(0, 100)
        ax.legend(loc="upper left", fontsize=6)
        ax.grid(True, color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.savefig(os.path.join(FIG, "fig_model_expected.pdf")); fig.savefig(os.path.join(FIG, "fig_model_expected.png"), dpi=300)
        plt.close(fig)

        decomposition_figure()

        # ---------- Figure C: residual z by group, with classifier-error intervals for outliers ----------
        fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
        for k, (g, color) in enumerate((("reference", BEFORE), ("matched", BEFORE), ("listed", SERIES))):
            e = d[d.group == g]
            x = k + RNG.uniform(-0.25, 0.25, len(e))
            ax.scatter(x, e["z"], s=6, color=color, alpha=0.7, linewidths=0)
            f = e["outlier"].values
            ax.vlines(x[f], e.loc[f, "z_p05"], e.loc[f, "z"], color=SERIES_2, linewidth=1)
        ax.axhline(Z_OUT, color=SERIES_2, linewidth=0.9, linestyle="--")
        ax.set_xticks([0, 1, 2], [f"Non-listed\nreference ({len(R)})", f"Matched\nnon-listed ({len(M)})", f"HCR-listed\n({len(L)})"])
        ax.set_ylabel("Survey-dependence z\n(vs expected for field, career, citations)")
        ax.tick_params(axis="x", length=0)
        ax.grid(axis="y", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        fig.savefig(os.path.join(FIG, "fig_model_z.pdf")); fig.savefig(os.path.join(FIG, "fig_model_z.png"), dpi=300)
        plt.close(fig)


    def decomposition_figure():
        """Survey share of papers p (volume) against the survey citation dividend D, with curves of equal survey
    dependence R_C = pD / (pD + 1 - p). Drawn from model_scores_INTERNAL.csv, so it can be redrawn alone."""
        d = pd.read_csv(os.path.join(OUT, "model_scores_INTERNAL.csv"))
        R, L = d[d.group == "reference"], d[d.group == "listed"]
        p95, D95 = np.nanpercentile(R["p"], 95), np.nanpercentile(R["D"], 95)
        fig, ax = plt.subplots(figsize=(3.5, 2.9), constrained_layout=True)
        pp = np.linspace(0.003, 0.6, 400)
        for Cv in (25, 50, 75):
            c = Cv / 100
            Dv = c * (1 - pp) / (pp * (1 - c))
            ax.plot(100 * pp, Dv, color=common.GRID, linewidth=1.1, zorder=1)
            ax.text(44.5, Dv[np.argmin(np.abs(pp - 0.445))], f"$R_C$ = {Cv}%", fontsize=6.5, color=common.INK_2, va="center", ha="left")
        ax.scatter(R["p"], R["D"], color=REFERENCE, zorder=2, label=NAMES["reference"], **{**POINT, "s": 8})
        rf = L["key"].isin(r_sets()["names"]).values
        ax.scatter(L["p"][~rf], L["D"][~rf], color=LISTED, zorder=3, label=NAMES["listed"], **{**POINT, "s": 9, "alpha": 0.85})
        flag(ax, L["p"][rf], L["D"][rf], label=False)
        ax.axvline(p95, color=common.INK_2, linewidth=0.7, linestyle=":")
        ax.axhline(D95, color=common.INK_2, linewidth=0.7, linestyle=":")
        ax.text(p95 + 0.4, 0.11, "random sample\n95th pct.", fontsize=6, color=common.INK_2, va="bottom", ha="left")
        ax.text(44.5, D95 * 1.06, "random sample 95th pct.", fontsize=6, color=common.INK_2, va="bottom", ha="right")
        ax.set_yscale("log")
        ax.set_xlim(0, 45)
        ax.set_ylim(0.1, 50)
        ax.set_yticks([0.1, 0.2, 0.5, 1, 2, 5, 10, 20])
        ax.set_yticklabels(["0.1", "0.2", "0.5", "1", "2", "5", "10", "20"])
        ax.yaxis.set_minor_formatter(plt.NullFormatter())
        ax.set_xlabel("Volume: surveys as % of papers ($v$)")
        ax.set_ylabel("Dividend: citations per survey /\ncitations per other record ($D$, log scale)")
        h, _ = ax.get_legend_handles_labels()
        ax.legend(handles=h + [r_handle()], loc="upper right", fontsize=6.3, borderaxespad=0.1)
        style(ax, None)
        save(fig, "fig_model_decomposition")


    if sys.argv[1:] == ["figure"]:      # redraw from the saved outputs without refitting
        decomposition_figure()
    else:
        main()


# =====================================================================================================================
# Checks of the expectation model
# =====================================================================================================================

def outlier_model_checks():
    """Checks of the RQ4 expectation model (secondary): how much the residual scores depend on the model's choices.

    1. Range: HCR-listed researchers' citation levels and first-publication years relative to the random sample on which
       the model is fitted.
    2. Zero handling: researchers whose survey share of citations is 0 (logit clipped at EPS); refits with other EPS values.
    3. Specifications: as in the paper; with a squared log-citation term; with subfield-specific citation slopes; fitted on
       the random sample and matched peers together; HCR-listed researchers inside the random sample's citation range only.
    4. Refit within the bootstrap: the random sample is resampled, the model and scale refitted, and the HCR-listed minus
       random-sample difference in median residual z recomputed (2,000 resamples).

    Input: hcr_study/outliers/model_scores_INTERNAL.csv (written by analyses.py outlier-model)
    Output: hcr_study/outliers/model_checks.md"""
    OUT = os.path.join(HERE, "hcr_study", "outliers")
    RNG = np.random.default_rng(4)
    B = 2000


    def logit_pct(c, eps=0.5):
        c = np.clip(np.asarray(c, float), eps, 100 - eps) / 100
        return np.log(c / (1 - c))


    def design(df, spec):
        logc, yr = np.log10(df["cites"].astype(float)), (df["firstyr"] - 2000) / 10
        ai, other = (df["sub"] == "ai").astype(float), (df["sub"] == "other").astype(float)
        cols = [logc, yr, ai, other]
        if spec == "squared":
            cols.append((logc - 4.3) ** 2)
        if spec == "slopes":
            cols += [logc * ai, logc * other]
        return np.column_stack(cols)


    def scores(fit_on, d, spec="paper", eps=0.5):
        q = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(design(fit_on, spec), logit_pct(fit_on["C"], eps))
        res = logit_pct(fit_on["C"], eps) - q.predict(design(fit_on, spec))
        s = 1.4826 * np.median(np.abs(res - np.median(res)))
        return (logit_pct(d["C"], eps) - q.predict(design(d, spec))) / s, q


    def compare(z, d):
        L, R, M = z[d.group == "listed"], z[d.group == "reference"], z[d.group == "matched"]
        return (f"median z HCR-listed {np.median(L):.2f}, random sample {np.median(R):.2f}, matched peers {np.median(M):.2f}; "
                f"HCR-listed vs random sample two-sided p={mannwhitneyu(L, R).pvalue:.2g}, vs matched peers "
                f"p={mannwhitneyu(L, M).pvalue:.2g}")


    def main():
        d = pd.read_csv(os.path.join(OUT, "model_scores_INTERNAL.csv"))
        d = d[d.group.isin(["listed", "reference", "matched"])].reset_index(drop=True)
        R, L, M = (d[d.group == g] for g in ("reference", "listed", "matched"))
        lo, hi = R["cites"].min(), R["cites"].max()
        out = ["# Checks of the RQ4 expectation model", "", "## 1. Range",
               f"- random sample: Scopus citations {lo:,.0f} to {hi:,.0f}; first publication {R.firstyr.min():.0f} to "
               f"{R.firstyr.max():.0f}",
               f"- HCR-listed: citations {L.cites.min():,.0f} to {L.cites.max():,.0f} (median {L.cites.median():,.0f}); above "
               f"the random sample's maximum: {(L.cites > hi).sum()} of {len(L)}; first publication outside its range: "
               f"{((L.firstyr < R.firstyr.min()) | (L.firstyr > R.firstyr.max())).sum()}",
               f"- matched peers above the random sample's maximum: {(M.cites > hi).sum()} of {len(M)}", "",
               "## 2. Zero survey share",
               "- researchers with no survey citations (logit clipped): " + ", ".join(
                   f"{g} {(e['C'] == 0).sum()} of {len(e)}" for g, e in (("HCR-listed", L), ("random sample", R),
                                                                          ("matched peers", M)))]
        for eps in (0.1, 0.5, 1.0, 2.0):
            z, _ = scores(R, d, eps=eps)
            out.append(f"- clipping at {eps}%{' (as in the paper)' if eps == 0.5 else ''}: {compare(z, d)}")
        nz = d[d["C"] > 0].reset_index(drop=True)
        z, _ = scores(nz[nz.group == "reference"], nz)
        out.append(f"- researchers with survey citations only (refitted): {compare(z, nz.reset_index(drop=True))}")

        out += ["", "## 3. Specifications"]
        z0, q0 = scores(R, d)
        out.append(f"- as in the paper (coefficients log10 citations {q0.coef_[0]:.3f}, decade of first publication "
                   f"{q0.coef_[1]:.3f}, AI {q0.coef_[2]:.3f}, other {q0.coef_[3]:.3f}): {compare(z0, d)}")
        for spec, lab in (("squared", "with a squared log-citation term"), ("slopes", "with subfield-specific citation slopes")):
            z, _ = scores(R, d, spec)
            out.append(f"- {lab}: {compare(z, d)}")
        z, _ = scores(d[d.group != "listed"].drop_duplicates("id"), d)      # a researcher in both groups counts once
        out.append(f"- fitted on the random sample and matched peers together: {compare(z, d)}")
        inr = d[d.group != "listed"].index.union(L.index[L.cites.between(lo, hi)])
        out.append(f"- HCR-listed inside the random sample's citation range only ({L.cites.between(lo, hi).sum()} of "
                   f"{len(L)}): {compare(z0[d.index.get_indexer(inr)], d.loc[inr])}")
        above = L.cites > hi
        out.append(f"- HCR-listed above that range: median z {np.median(z0[L.index[above]]):.2f} (n={above.sum()})")

        diffs, dm = [], []
        for _ in range(B):
            Rb = R.sample(len(R), replace=True, random_state=RNG.integers(1 << 31))
            z, _ = scores(Rb, d)
            diffs.append(np.median(z[d.group == "listed"]) - np.median(z[d.group == "reference"]))
            dm.append(np.median(z[d.group == "listed"]) - np.median(z[d.group == "matched"]))
        diffs, dm = np.array(diffs), np.array(dm)
        out += ["", "## 4. Model refitted in each bootstrap resample of the random sample",
                f"- HCR-listed minus random sample, median z: {np.median(z0[d.group == 'listed']) - np.median(z0[d.group == 'reference']):.2f} "
                f"(95% CI {np.percentile(diffs, 2.5):.2f} to {np.percentile(diffs, 97.5):.2f}; resamples with a difference "
                f"<= 0: {(diffs <= 0).sum()} of {B})",
                f"- HCR-listed minus matched peers, median z: {np.median(z0[d.group == 'listed']) - np.median(z0[d.group == 'matched']):.2f} "
                f"(95% CI {np.percentile(dm, 2.5):.2f} to {np.percentile(dm, 97.5):.2f})"]
        with open(os.path.join(OUT, "model_checks.md"), "w") as fh:
            fh.write("\n".join(out) + "\n")
        print("\n".join(out))


    main()


# =====================================================================================================================
# Stanford-Elsevier composite indicator with and without surveys
# =====================================================================================================================
#
# The Stanford-Elsevier composite citation indicator with and without surveys.
#
# The composite (Ioannidis et al. 2019) sums six log-normalized career indicators:
# total citations (NC), h-index (H), Schreiber's co-authorship-adjusted hm-index (Hm),
# citations to single-authored papers (NCS), to single- or first-authored papers (NCSF),
# and to single-, first- or last-authored papers (NCSFL):
#
#     c = sum_i log(1 + x_i) / maxlog_i
#
# with maxlog_i from the 2024 release (Table 3, citations including self-citations, since
# OpenAlex counts do). Each indicator is recomputed from the researcher's classified OpenAlex
# profile with all papers and without surveys (non-papers stay in both), the researcher's
# author position found by name in each paper's author list.
#
# Rank and inclusion: the release lists the top 2% of each subfield by c (ns). For the
# Networking and Telecommunications records, each researcher's change in c is applied to
# their published c (ns). Two scenarios: (unilateral) only that researcher's surveys are
# removed; (universal) every record in the subfield loses surveys, with the change for
# records outside our samples predicted from the reference sample (median change as a
# linear function of c). Rank and the subfield cutoff are recomputed under each.
#
# Inputs: inputs/stanford_elsevier/els.csv and Table_3_maxlog_career_2024_pubs_since_1788_wopp_extracted_202508.xlsx
# Output: hcr_study/composite/*, hcr_study/figures/fig_composite.*
#
#     python analyses.py composite            # compute, statistics and figure
#     python analyses.py composite figure     # redraw the figure from the saved outputs

SE = os.path.join(HERE, "inputs", "stanford_elsevier")
COMPOSITE_OUT = os.path.join(STUDY, "composite")
os.makedirs(COMPOSITE_OUT, exist_ok=True)
K = ["authfull", "firstyr", "np6024"]
NT = "Networking & Telecommunications"
IND = ["NC", "H", "Hm", "NCS", "NCSF", "NCSFL"]


def name_norm(s):
    return re.sub(r"[^a-z]", "", unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode().lower())


def maxlog():
    x = pd.read_excel(os.path.join(SE, "Table_3_maxlog_career_2024_pubs_since_1788_wopp_extracted_202508.xlsx"),
                      engine="calamine")
    r = x.iloc[0]
    return dict(zip(IND, [r["nc9624"], r["h24"], r["hm24"], r["ncs"], r["ncsf"], r["ncsfl"]]))


def position(authors, last, first_initial):
    """'single', 'first', 'last', 'middle', or None if the researcher is not found."""
    names = [a.strip() for a in str(authors).split(";") if a.strip()]
    if not names:
        return None, 1
    for i, n in enumerate(names):
        toks = n.replace("‐", "-").split()
        if not toks:
            continue
        cands = {name_norm("".join(toks[-k:])) for k in range(1, min(3, len(toks)) + 1)} | {name_norm(toks[0])}
        if last in cands and name_norm(toks[0])[:1] == first_initial or (last in cands and len(toks) == 1):
            if len(names) == 1:
                return "single", 1
            return ("first" if i == 0 else "last" if i == len(names) - 1 else "middle"), len(names)
    return None, len(names)


def h_index(c):
    c = np.sort(np.asarray(c))[::-1]
    return int(np.sum(c >= np.arange(1, len(c) + 1)))


def hm_index(c, n_auth):
    o = np.argsort(-np.asarray(c))
    c, a = np.asarray(c)[o], np.asarray(n_auth, float)[o]
    r_eff = np.cumsum(1 / np.maximum(a, 1))
    ok = c >= r_eff
    return float(r_eff[ok].max()) if ok.any() else 0.0


def indicators(df):
    c = df["citationCount"].values
    return {"NC": c.sum(), "H": h_index(c), "Hm": hm_index(c, df["n_auth"].values),
            "NCS": c[df["pos"].eq("single").values].sum(),
            "NCSF": c[df["pos"].isin(["single", "first"]).values].sum(),
            "NCSFL": c[df["pos"].isin(["single", "first", "last"]).values].sum()}


def composite_score(ind, ml):
    return sum(np.log1p(ind[k]) / ml[k] for k in IND)


def profile_scores(path, name, ml):
    last, _, first = str(name).partition(",")
    last, fi = name_norm(last), name_norm(first.strip())[:1]
    df = pd.read_csv(path, usecols=["authors", "citationCount", "Category"])
    df["citationCount"] = df["citationCount"].fillna(0).astype(int)
    pos = [position(a, last, fi) for a in df["authors"]]
    df["pos"], df["n_auth"] = [p[0] for p in pos], [p[1] for p in pos]
    survey = df["Category"].eq("survey")
    a, b = indicators(df), indicators(df[~survey])
    out = {f"{k}": a[k] for k in IND} | {f"{k}*": b[k] for k in IND}
    out["c_oa"], out["c_oa*"] = composite_score(a, ml), composite_score(b, ml)
    out["position found %"] = 100 * df["pos"].notna().mean()
    return out


def load_composite(ml):
    sd = pd.read_csv(os.path.join(STUDY, "outliers", "survey_dependence_INTERNAL.csv"))
    els = pd.read_csv(os.path.join(SE, "els.csv"), thousands=",", low_memory=False, encoding="utf-8-sig",
                      usecols=K + ["c", "c (ns)", "sm-subfield-1", "rank sm-subfield-1 (ns)", "sm-subfield-1 count"])
    els = els.drop_duplicates(K, keep=False)
    hm = pd.read_csv(os.path.join(STUDY, "BTAS_HCR_Computer_Science.csv"))
    hm = hm[hm["Match status"] == "ok"][["HCR name"] + K]
    mt = comparison_groups.peers_as_controls()[0].rename(columns={
        "Control authfull": "authfull", "Control firstyr": "firstyr", "Control papers (Scopus)": "np6024"})
    rf = pd.read_csv(os.path.join(STUDY, "reference", "per_reference.csv"))
    rf = rf[rf["Consistent with Scopus"]]
    rows = []
    for _, r in sd.iterrows():
        if r["group"] == "listed":
            m = hm[hm["HCR name"] == r["key"]]
            path = os.path.join(STUDY, "classified", f"{r['id']}.csv")
            name = r["key"]
            grp = "listed"
        else:
            m = mt[mt["Control OpenAlex ID"] == r["id"]]
            path = os.path.join(comparison_groups.PEERS_CLASSIFIED, f"{r['id']}.csv")
            name = m["authfull"].iloc[0] if len(m) else r["key"]
            grp = "matched"
        rows.append({"group": grp, "id": r["id"], "C": r["C"], **(m[K].iloc[0].to_dict() if len(m) else {}),
                     **profile_scores(path, name, ml)})
    for _, r in rf.iterrows():
        rows.append({"group": "reference", "id": r["openalex_id"], "C": r["Survey % of citations"],
                     **{k: r[k] for k in K},
                     **profile_scores(os.path.join(STUDY, "reference", "classified", f"{r['openalex_id']}.csv"),
                                      r["authfull"], ml)})
    d = pd.DataFrame(rows).merge(els, on=K, how="left")
    d["dc"] = d["c_oa*"] - d["c_oa"]
    d["dc %"] = 100 * d["dc"] / d["c_oa"]
    for k in IND:
        d[f"{k} drop %"] = 100 * (d[k] - d[f"{k}*"]) / d[k].replace(0, np.nan)
    return d, els


def recompute_composite():
    ml = maxlog()
    d, els = load_composite(ml)
    lines = [f"n: " + ", ".join(f"{g} {int((d.group == g).sum())}" for g in ("listed", "matched", "reference")),
             f"author position found: median {d['position found %'].median():.1f}% of papers per researcher "
             f"(10th percentile {d['position found %'].quantile(0.1):.1f}%)"]
    ok = d.dropna(subset=["c"])
    lines.append(f"validation: OpenAlex composite vs published composite (with self-citations): Spearman "
                 f"{spearmanr(ok['c_oa'], ok['c'])[0]:.2f}, n={len(ok)}")
    R = d[d.group == "reference"]
    for g in ("listed", "matched", "reference"):
        e = d[d.group == g]
        lines.append(f"{g}: median composite drop {-e['dc %'].median():.2f}% (IQR {-e['dc %'].quantile(0.75):.2f} to "
                     f"{-e['dc %'].quantile(0.25):.2f}); median absolute drop {-e['dc'].median():.3f}; "
                     + ", ".join(f"{k} {e[f'{k} drop %'].median():.1f}%" for k in IND))
    L, M = d[d.group == "listed"], d[d.group == "matched"]
    lines.append(f"composite drop, listed vs reference: two-sided p={mannwhitneyu(-L['dc %'], -R['dc %']).pvalue:.2g}; "
                 f"listed vs matched p={mannwhitneyu(-L['dc %'], -M['dc %']).pvalue:.2g}")
    rho, p = spearmanr(d["C"], -d["dc %"])
    lines.append(f"composite drop vs survey share of citations: Spearman {rho:.2f} (p={p:.2g})")

    # ---- rank and inclusion, per computer science-related subfield of the HCR-listed researchers ----
    Rall = R.dropna(subset=["c (ns)"])
    parts = []
    for sub in d.loc[d.group == "listed", "sm-subfield-1"].dropna().value_counts().index:
        nt = els[els["sm-subfield-1"] == sub].copy()
        n_sub = int(nt["sm-subfield-1 count"].iloc[0])
        k2 = int(round(0.02 * n_sub))
        cutoff = nt.loc[nt["rank sm-subfield-1 (ns)"] <= k2, "c (ns)"].min()
        Rn = Rall[Rall["sm-subfield-1"] == sub]
        src = "own" if len(Rn) >= 50 else "pooled"
        Rn = Rn if len(Rn) >= 50 else Rall
        qr = QuantileRegressor(quantile=0.5, alpha=0, solver="highs").fit(Rn[["c (ns)"]].values, Rn["dc"].values)
        base = np.sort(nt["c (ns)"].values)
        base_univ = np.sort(nt["c (ns)"].values + qr.predict(nt[["c (ns)"]].values))
        S = d[(d["sm-subfield-1"] == sub)].dropna(subset=["c (ns)"]).copy()
        if S.empty:
            continue
        S["subfield size"], S["k2"] = n_sub, k2
        S["rank before"] = [1 + np.sum(base > v) for v in S["c (ns)"]]
        S["c new"] = S["c (ns)"] + S["dc"]
        S["rank unilateral"] = [1 + np.sum(base > v) for v in S["c new"]]
        S["rank universal"] = [1 + np.sum(base_univ > v) for v in S["c new"]]
        cut_univ = cutoff + qr.predict([[cutoff]])[0]
        S["in top 2% before"] = S["c (ns)"] >= cutoff
        S["out unilateral"] = S["in top 2% before"] & (S["c new"] < cutoff)
        S["out universal"] = S["in top 2% before"] & (S["c new"] < cut_univ)
        S["near cutoff"] = (S["c (ns)"] >= cutoff) & (S["c (ns)"] < 1.1 * cutoff)
        lines.append(f"[{sub}] {len(nt):,} records; top 2% = rank {k2:,} of {n_sub:,}; cutoff c (ns) {cutoff:.3f}; "
                     f"change model from {src} reference (n={len(Rn)}); groups here: "
                     + ", ".join(f"{g} {int((S.group == g).sum())}" for g in ("listed", "matched", "reference")))
        parts.append(S)
    S = pd.concat(parts, ignore_index=True)
    for g in ("listed", "matched", "reference"):
        e = S[S.group == g]
        ch_u = np.log2(e["rank unilateral"] / e["rank before"])
        ch_v = np.log2(e["rank universal"] / e["rank before"])
        lines.append(f"ALL subfields {g} (n={len(e)}): universal exclusion: median rank change "
                     f"{np.median(e['rank universal'] - e['rank before']):+.0f} places (median x{2 ** np.median(ch_v):.2f}); "
                     f"worse by 2x or more: {int((ch_v >= 1).sum())}; unilateral: median x{2 ** np.median(ch_u):.2f}; "
                     f"in the top 2% before: {int(e['in top 2% before'].sum())}; would fall below the 2% cutoff: "
                     f"universal {int(e['out universal'].sum())}, unilateral {int(e['out unilateral'].sum())}; "
                     f"within 10% above the cutoff: {int(e['near cutoff'].sum())}")
    tau = kendalltau(S["rank before"] / S["k2"], S["rank universal"] / S["k2"])[0]
    lines.append(f"Kendall tau, relative rank before vs after universal exclusion (all our records): {tau:.2f}")
    a = np.log2(S[S.group == "listed"]["rank universal"] / S[S.group == "listed"]["rank before"])
    b = np.log2(S[S.group == "reference"]["rank universal"] / S[S.group == "reference"]["rank before"])
    lines.append(f"relative rank loss under universal exclusion, listed vs reference: two-sided p="
                 f"{mannwhitneyu(a, b).pvalue:.2g}")
    k2 = None

    d.round(4).to_csv(os.path.join(COMPOSITE_OUT, "composite_INTERNAL.csv"), index=False)
    S.round(4).to_csv(os.path.join(COMPOSITE_OUT, "rank_INTERNAL.csv"), index=False)
    with open(os.path.join(COMPOSITE_OUT, "stats.md"), "w") as fh:
        fh.write("\n".join(f"- {l}" for l in lines) + "\n")
    print("\n".join(lines))
    composite_figure(d, S, k2)


def composite_figure(d, S, k2):
    R_IDS = r_sets()["ids"]
    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.6 if SPRINGER else 3.0), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.25]})
    ax = axes[0]
    groups = ["listed", "matched", "reference"]                          # HCR-listed first, as in every figure
    data, flags = [], []
    for g in groups:
        e = d[d.group == g].dropna(subset=["dc %"])
        data.append(-e["dc %"].values); flags.append((e["id"].isin(R_IDS) & (g == "listed")).values)
    boxes(ax, data, range(3), [COLORS[g] for g in groups], flags=flags, rng=np.random.default_rng(3))
    ax.set_xticks(range(3), [f"{NAMES[g]}\n(n = {int((d.group == g).sum())})" for g in groups])
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel("Drop in the composite indicator\nwithout surveys (%)")
    ax.set_ylim(bottom=-0.5)
    ax.legend(handles=[r_handle()], loc="upper left", frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    panel(ax, "a", "Composite drop by group")
    style(ax)

    # (b) one row per HCR-listed researcher: rank relative to the top-2% boundary, with and without surveys
    ax = axes[1]
    e = S[S.group == "listed"].copy()
    e["rank before"] = e["rank before"] / e["k2"]
    e["rank universal"] = e["rank universal"] / e["k2"]
    e = e.sort_values("rank before").reset_index(drop=True)
    y = np.arange(len(e))
    rf = e["id"].isin(R_IDS).values
    out = e["out universal"].values
    for k, r in e.iterrows():
        ax.plot([r["rank before"], r["rank universal"]], [k, k], color=LISTED if not out[k] else INK,
                linewidth=0.7 if not out[k] else 1.3, alpha=0.55 if not out[k] else 1, zorder=2)
    ax.scatter(e["rank before"], y, s=10, facecolors="white", edgecolors=LISTED, linewidths=0.7, zorder=3,
               label="With surveys (published composite)")
    ax.scatter(e["rank universal"][~rf], y[~rf], s=10, color=LISTED, zorder=4, linewidths=0,
               label="Without surveys (estimated)")
    flag(ax, e["rank universal"][rf], y[rf])
    ax.axvline(1.0, color=INK, linewidth=0.8, linestyle=":")
    ax.text(1.08, len(e) * 0.03, "top-2%\nboundary", fontsize=6.5, color=INK_2, va="bottom")
    ax.set_xscale("log")
    ax.xaxis.set_major_locator(plt.FixedLocator([0.001, 0.01, 0.1, 1]))
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(plt.NullFormatter())
    ax.set_xlim(0.0002, 3)
    ax.set_ylim(len(e), -1)
    ax.set_yticks([])
    ax.set_xlabel("Rank in own subfield / rank at the top-2% boundary\n(log scale, left is better)" if SPRINGER else
                  "Rank in own subfield / rank at the top-2% boundary (log scale, left is better)")
    ax.set_ylabel(f"HCR-listed researchers, ordered by rank (n = {len(e)})")
    ax.plot([], [], color=INK, linewidth=1.3, label="Falls below the top-2% boundary")
    ax.legend(loc="lower left")
    panel(ax, "b", "Rank in own subfield, with and without surveys")
    style(ax, "x")
    save(fig, "fig_composite")


def composite():
    if sys.argv[1:] == ["figure"]:      # redraw from the saved outputs without recomputing
        composite_figure(pd.read_csv(os.path.join(COMPOSITE_OUT, "composite_INTERNAL.csv")), pd.read_csv(os.path.join(COMPOSITE_OUT, "rank_INTERNAL.csv")), None)
    else:
        recompute_composite()


# =====================================================================================================================
# Sensitivity of the boundary simulation
# =====================================================================================================================

def boundary_scenarios():
    """Sensitivity of the boundary simulation (comparison_groups.py boundary analyze) to its assumptions (secondary).

    Varied one at a time, with everything else as in the paper:
    - density of the simulated population below the cutoff: 0.5, 1.5 and 2 times the density just inside it;
    - composite changes of the records below the cutoff: half and one and a half times those drawn from the boundary sample;
    - records that may cross the cutoff: within 10% or 30% of it, instead of 20%;
    - simulated span below the cutoff: 0.5 instead of 0.25 composite units;
    - changes as a share of each record's composite instead of absolute changes.

    Output: hcr_study/boundary/scenarios.md"""
    SUBFIELDS = ("Networking & Telecommunications", "Artificial Intelligence & Image Processing")
    SCENARIOS = [("as in the paper", {}),
                 ("density below the cutoff x0.5", {"dens": 0.5}), ("density below the cutoff x1.5", {"dens": 1.5}),
                 ("density below the cutoff x2", {"dens": 2.0}),
                 ("changes below the cutoff x0.5", {"below": 0.5}), ("changes below the cutoff x1.5", {"below": 1.5}),
                 ("records within 10% of the cutoff may cross", {"band": 1.1}),
                 ("records within 30% of the cutoff may cross", {"band": 1.3}),
                 ("span 0.5 below the cutoff", {"span": 0.5}), ("relative changes", {"relative": True})]


    def simulate(n_boot=2000, dens=1.0, below=1.0, band=1.2, span=0.25, relative=False):
        rng = np.random.default_rng(5)
        b = pd.read_csv(os.path.join(comparison_groups.BOUNDARY_OUT, "boundary_INTERNAL.csv"))
        b = b[b["Consistent with Scopus"]]
        nt = comparison_groups.boundary_frame()
        k2 = int(round(0.02 * nt["sm-subfield-1 count"].iloc[0]))
        cut = nt.loc[nt["rank sm-subfield-1 (ns)"] <= k2, "c (ns)"].min()
        inside = np.sort(nt.loc[nt["rank sm-subfield-1 (ns)"] <= k2, "c (ns)"].values)[::-1]
        band_c = nt.loc[nt["rank sm-subfield-1 (ns)"].between(*comparison_groups.band_ranks(nt)), "c (ns)"].values
        n_below = int(dens * len(band_c) / (band_c.max() - band_c.min()) * span)
        lower = cut - np.linspace(0, span, n_below + 2)[1:-1]
        obs = (b["dc"] / b["c (ns)"]).values if relative else b["dc"].values
        cross = inside <= band * cut
        # point estimate: mean over random assignments of the observed changes; interval: bootstrap of the changes
        point, exits = [], []
        for it in range(n_boot):
            for boot in (False, True):
                d = rng.choice(obs, len(obs)) if boot else obs
                draw = lambda x: x * rng.choice(d, len(x)) if relative else rng.choice(d, len(x))
                new_inside = inside.copy()
                new_inside[cross] = inside[cross] + draw(inside[cross])
                new_inside[~cross] = inside[~cross] + (np.median(d) * inside[~cross] if relative else np.median(d))
                new_lower = lower + below * draw(lower)
                thr = np.sort(np.concatenate([new_inside, new_lower]))[::-1][k2 - 1]
                (exits if boot else point).append(int((new_inside < thr).sum()))
        return np.mean(point), np.percentile(exits, 2.5), np.percentile(exits, 97.5), k2


    def main():
        res = {}
        for sf in SUBFIELDS:
            comparison_groups.set_boundary_subfield(sf)
            res[sf] = {name: simulate(**kw) for name, kw in SCENARIOS}
        out = ["# Boundary simulation: researchers who would leave the top 2% and be replaced, by assumption", "",
               "| Scenario | " + " | ".join(SUBFIELDS) + " | Both |", "|---|---|---|---|"]
        for name, _ in SCENARIOS:
            cells, tot = [], 0
            for sf in SUBFIELDS:
                e, lo, hi, k2 = res[sf][name]
                cells.append(f"{100 * e / k2:.1f}% ({100 * lo / k2:.1f}-{100 * hi / k2:.1f})")
                tot += e
            out.append(f"| {name} | " + " | ".join(cells) + f" | {tot:,.0f} |")
        pct = [100 * res[sf][n][0] / res[sf][n][3] for sf in SUBFIELDS for n, _ in SCENARIOS]
        out += ["", f"Range of point estimates over scenarios and subfields: {min(pct):.1f}% to {max(pct):.1f}%"]
        path = os.path.join(STUDY, "boundary", "scenarios.md")
        with open(path, "w") as fh:
            fh.write("\n".join(out) + "\n")
        print("\n".join(out))


    main()


# =====================================================================================================================
# Primary tests of RQ2-RQ4
# =====================================================================================================================

def primary_endpoints():
    """One primary endpoint per research question, with Holm correction across the three.

        RQ2  survey share of citations vs survey share of papers, 2025 HCR-listed (Wilcoxon)
        RQ3  material drop: share of 2025 HCR-listed researchers whose OpenAlex top-1% papers fall by at least a tenth
            without surveys (one-sided exact sign test that the median drop exceeds 10%)
        RQ4  survey dependence of HCR-listed vs random-sample researchers, adjusted for citation level, first publication
            year, and subfield: median regression of logit(R_C) on an indicator for HCR-listed researchers and these
            covariates, fitted on both groups together, so that the uncertainty of the adjustment enters the test (two-sided)

    All other analyses in the paper are secondary or exploratory.

    Output: hcr_study/primary_endpoints.md"""
    def main():
        pm = pd.read_csv(os.path.join(STUDY, "per_member.csv"))
        pm = pm[pm["Consistent with Scopus"]]
        top1 = [c for c in pm.columns if c.startswith("Top-1% papers 20")][0]
        rows = []
        rows.append(("RQ2", f"survey % of citations {pm['Survey % of citations'].median():.1f} vs of papers "
                           f"{pm['Survey % of papers'].median():.1f}",
                     wilcoxon(pm["Survey % of citations"], pm["Survey % of papers"]).pvalue))
        drop = 100 * (pm[top1] - pm["Top-1% papers*"]) / pm[top1]
        k, n = int((drop >= 10).sum()), len(drop)
        ci = binomtest(k, n).proportion_ci(method="wilson")
        rows.append(("RQ3", f"top-1% papers down >= 10%: {k} of {n} ({100 * k / n:.0f}%, Wilson 95% CI {100 * ci.low:.0f}-{100 * ci.high:.0f}%), "
                           f"median drop {drop.median():.1f}%", binomtest(k, n, 0.5, alternative="greater").pvalue))
        m = pd.read_csv(os.path.join(STUDY, "outliers", "model_scores_INTERNAL.csv"))
        d = m[m.group.isin(["listed", "reference"])].copy()
        c = np.clip(d["C"].astype(float), 0.5, 99.5) / 100                       # logit clipped as in analyses.py outlier-model
        d = d.assign(y=np.log(c / (1 - c)), lc=np.log10(d["cites"].astype(float)), yr=(d["firstyr"] - 2000) / 10,
                     hcr=(d.group == "listed").astype(int), ai=(d["sub"] == "ai").astype(int), oth=(d["sub"] == "other").astype(int))
        r = smf.quantreg("y ~ hcr + lc + yr + ai + oth", d).fit(q=0.5, max_iter=5000)
        rows.append(("RQ4", f"HCR-listed coefficient {r.params['hcr']:.2f} on the logit scale (odds ratio {np.exp(r.params['hcr']):.2f}, 95% CI "
                           f"{np.exp(r.conf_int().loc['hcr', 0]):.2f}-{np.exp(r.conf_int().loc['hcr', 1]):.2f}); median expected R_C "
                           f"{m[m.group == 'listed'].C_expected.median():.1f}% vs observed {m[m.group == 'listed'].C.median():.1f}%",
                     r.pvalues["hcr"]))   # two-sided: the difference lies opposite to the prediction
        t = pd.DataFrame(rows, columns=["question", "estimate", "p"])
        order = t["p"].argsort().values
        adj = np.empty(len(t))
        running = 0
        for rank, i in enumerate(order):
            running = max(running, min(1, (len(t) - rank) * t["p"].iloc[i]))
            adj[i] = running
        t["Holm p"] = adj
        with open(os.path.join(STUDY, "primary_endpoints.md"), "w") as fh:
            fh.write(t.to_markdown(index=False, floatfmt=".2g") + "\n")
        print(t.to_string(index=False))


    main()


# =====================================================================================================================
# Semantic Scholar records of the HCR-listed researchers' papers
# =====================================================================================================================

def s2_types():
    """Semantic Scholar records for the Smyth & Cunningham rule, which is defined on Semantic Scholar's
    publicationTypes (Review / MetaAnalysis), title, abstract and venue, not on OpenAlex data. Batch API with the key
    SEMANTIC_SCHOLAR_API_KEY from .env (header x-api-key), at most one request per second, exponential backoff
    on errors. Covers the papers of the HCR-listed researchers in hcr_study/outliers/survey_dependence_INTERNAL.csv.
    Usage: analyses.py s2-types hcr   -> cache hcr_study/s2_types/hcr.json
           {our_id: {types, venue, journal, publication_venue, title, abstract}}; older entries without title are refetched"""
    OUT = os.path.join(HERE, "hcr_study", "s2_types"); os.makedirs(OUT, exist_ok=True)
    URL = "https://api.semanticscholar.org/graph/v1/paper/batch"
    load_dotenv(os.path.join(HERE, ".env"), override=True)
    KEY = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    HEADERS = {"x-api-key": KEY} if KEY else {}
    _last = [0.0]
    def post(payload):
        wait = 1.1 - (time.time() - _last[0])          # stay below 1 request per second
        if wait > 0: time.sleep(wait)
        _last[0] = time.time()
        return requests.post(URL, params={"fields": "title,abstract,publicationTypes,venue,journal,publicationVenue,referenceCount"}, json=payload, headers=HEADERS, timeout=60)

    def fetch(idmap, cache_path):
        """idmap: our_id -> S2 id string (DOI:...). Returns our_id -> {types, venue} (None if not found)."""
        cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
        # records cached before reference counts were needed get only their reference count; other fields stay as first fetched
        todo = [k for k in idmap if k not in cache or (cache[k] is not None and ("title" not in cache[k] or "reference_count" not in cache[k]))]
        for i in range(0, len(todo), 500):
            chunk = todo[i:i + 500]
            for attempt in range(12):
                r = post({"ids": [idmap[k] for k in chunk]})
                if r.status_code == 200: break
                time.sleep(min(60, 3 * 2 ** attempt))
            else:
                raise SystemExit(f"failed at {i}: {r.status_code} {r.text[:200]}")
            for k, res in zip(chunk, r.json()):
                if cache.get(k) is not None and "title" in cache[k]:
                    cache[k]["reference_count"] = None if res is None else res.get("referenceCount")
                    continue
                cache[k] = None if res is None else {"reference_count": res.get("referenceCount"),
                                                     "types": res.get("publicationTypes") or [], "venue": res.get("venue") or "",
                                                     "journal": (res.get("journal") or {}).get("name") or "",
                                                     "publication_venue": (res.get("publicationVenue") or {}).get("name") or "",
                                                     "title": res.get("title") or "", "abstract": res.get("abstract") or ""}
            json.dump(cache, open(cache_path, "w"))
            print(f"  {min(i + 500, len(todo))}/{len(todo)}", flush=True)
        return cache

    which = sys.argv[1] if len(sys.argv) > 1 else "hcr"
    if which != "hcr":
        raise SystemExit("usage: analyses.py s2-types hcr")
    s = pd.read_csv(os.path.join(HERE, "hcr_study", "outliers", "survey_dependence_INTERNAL.csv"))
    idmap = {}
    for i in s[s.group == "listed"].id:
        f = os.path.join(HERE, "hcr_study", "classified", f"{i}.csv")
        if not os.path.exists(f): continue
        for r in pd.read_csv(f, usecols=["id", "doi"]).itertuples():
            if isinstance(r.doi, str) and r.doi:
                idmap[r.id] = "DOI:" + r.doi.replace("https://doi.org/", "")
    print(which, len(idmap), "ids")
    res = fetch(idmap, os.path.join(OUT, f"{which}.json"))
    found = sum(v is not None for v in res.values())
    print(f"{which}: found {found}/{len(res)}; with Review/MetaAnalysis type: "
          f"{sum(1 for v in res.values() if v and any(t in ('Review', 'MetaAnalysis') for t in v['types']))}")


# =====================================================================================================================
# Smyth & Cunningham's rule on the HCR-listed researchers' papers
# =====================================================================================================================

def s2_rule():
    """Smyth & Cunningham rule (the authors' released code) with Semantic Scholar records (analyses.py s2-types hcr) on the
    HCR-listed researchers' profiles, next to the SciGenre and the OpenAlex-based baselines. Per researcher: share
    of records excluded, share of citations, and h-index drop; averaged over researchers. Records not found in Semantic
    Scholar count as untyped. Variants: with each work's full venue name (OpenAlex) added to Semantic Scholar's, and the
    rule as implemented from the published description (sensitivity checks).
    Output: hcr_study/s2_types/hcr_rule_metrics.json"""
    cache_p = os.path.join(HERE, "hcr_study", "s2_types", "hcr.json")
    T = json.load(open(cache_p)) if os.path.exists(cache_p) else {}
    s = pd.read_csv(os.path.join(HERE, "hcr_study", "outliers", "survey_dependence_INTERNAL.csv"))
    rows = []
    for i in s[s.group == "listed"].id:
        f = os.path.join(HERE, "hcr_study", "classified", f"{i}.csv")
        if not os.path.exists(f): continue
        df = pd.read_csv(f); df["citationCount"] = df["citationCount"].fillna(0).astype(int)
        h0 = calculate_indices(df)[0]; C = max(1, df.citationCount.sum())
        def stats(mask):
            return 100 * mask.mean(), 100 * df.loc[mask, "citationCount"].sum() / C, h0 - calculate_indices(df[~mask])[0]
        t = df.id.map(lambda k: T.get(k) or {})
        typed = t.map(lambda x: any(y in ("Review", "MetaAnalysis") for y in x.get("types", [])))
        sc = np.array([rq1_evaluation.is_review_released(T.get(k), ti, ab) for k, ti, ab in zip(df.id, df.title, df.abstract.fillna(""))])
        sc_full = np.array([rq1_evaluation.is_review_released(T.get(k), ti, ab, v) for k, ti, ab, v in
                            zip(df.id, df.title, df.abstract.fillna(""), df.venue.fillna("").str.replace("®", "", regex=False))])
        sc_pub = np.array([rq1_evaluation.is_review(*rq1_evaluation.from_s2_record(T.get(k), ti, ab)) for k, ti, ab in zip(df.id, df.title, df.abstract.fillna(""))])
        r = {"found": t.map(bool).mean()}
        for name, m in [("scigenre", df.Category == "survey"),     # the final model
                        ("openalex_review", df["Survey (indexer type)"].astype(bool)),
                        ("sc_openalex", df["Survey (Smyth & Cunningham)"].astype(bool)),
                        ("s2_review", typed), ("sc_s2", sc), ("sc_s2_full_venue", sc_full), ("sc_s2_published", sc_pub)]:
            r[name] = stats(np.asarray(m, dtype=bool))
        rows.append(r)
    KEYS = ["scigenre", "openalex_review", "sc_openalex", "s2_review", "sc_s2", "sc_s2_full_venue", "sc_s2_published"]
    out = {k: [float(np.mean([r[k][j] for r in rows])) for j in range(3)] for k in KEYS}
    out["coverage"] = float(np.mean([r["found"] for r in rows])); out["n"] = len(rows); out["cache_size"] = len(T)
    print(json.dumps(out, indent=1))
    json.dump(out, open(os.path.join(HERE, "hcr_study", "s2_types", "hcr_rule_metrics.json"), "w"), indent=1)


# =====================================================================================================================
# Survey citation dividend by publication cohort
# =====================================================================================================================

def dividend_window():
    """Survey citation dividend by publication cohort, measured two ways, to test whether the citation advantage of
    surveys declines over time (Smyth & Cunningham 2026) among the researchers we study.

    (A) Within researcher: for each researcher and publication cohort, mean citations per survey / mean citations per
        research paper of the same researcher and cohort (at least one survey and three research papers). Citations are
        counted (i) in a fixed window, the publication year and the two years after it (OpenAlex counts_by_year, which
        start in 2012, so cohorts 2012-2023), and (ii) in total up to the profile date, for comparison.
    (B) Timing: papers of 2012-2014 followed for eleven years. (C) Career stage: papers written within ten years of
        the first publication vs later (career start from the Fellows comparison file, academic researchers).

    Groups: the HCR-listed researchers (R1-R8 also separately), the reference researchers, and the academic ACM and
    IEEE Fellows, as in the Fellows comparison.
    Inputs: classified profiles, hcr_study/growth_survey/yearly/<id>.csv (analyses.py yearly-citations).
    Output: hcr_study/dividend_window/{within.csv, within_researcher.csv, speed.csv, results.md}"""
    ST = os.path.join(HERE, "hcr_study"); YEARLY = os.path.join(ST, "growth_survey", "yearly"); OUT = os.path.join(ST, "dividend_window")
    WIN = [(2012, 2014), (2015, 2017), (2018, 2020), (2021, 2023)]                     # fixed-window cohorts
    ALL = [(a, a + 2) for a in range(2000, 2022, 3)]                                    # 2000-02 ... 2021-23, total citations
    lab = lambda c: f"{c[0]}-{str(c[1])[2:]}"
    rng = np.random.default_rng(2026)


    def groups():
        per = pd.read_csv(os.path.join(ST, "per_member.csv")); per = per[per["Consistent with Scopus"]]
        S = pd.read_csv(os.path.join(ST, "fellows2026", "per_person_scored_INTERNAL.csv"))
        fel = lambda p: [(a, os.path.join(ST, "fellows2026", "classified", f"{a}.csv")) for a in S[S.group.str.startswith(p)]["OpenAlex ID"]]
        return {"HCR-listed": [(a, os.path.join(ST, "classified", f"{a}.csv")) for a in per["OpenAlex ID"]],
                "Reference": [(os.path.basename(f)[:-4], f) for f in sorted(glob.glob(os.path.join(ST, "reference", "classified", "*.csv")))],
                "ACM Fellows": fel("ACM"), "IEEE Fellows": fel("IEEE")}


    def papers(aid, path):
        d = pd.read_csv(path, usecols=["id", "title", "abstract", "year", "citationCount", "Category"])
        d = d[d.Category.isin(["survey", "research"]) & d.year.between(2000, 2023)].copy()
        d["year"] = d.year.astype(int); d["c"] = d.citationCount.fillna(0)
        y = pd.read_csv(os.path.join(YEARLY, f"{aid}.csv")); y["id"] = y.id.str.replace("https://openalex.org/", "", regex=False)
        m = d[["id", "year"]].merge(y.rename(columns={"year": "cy"}), on="id"); m["age"] = m.cy - m.year
        A = m[m.age.between(0, 10)].pivot_table(index="id", columns="age", values="citations", aggfunc="sum")
        A = A.reindex(index=list(set(m.id)), columns=range(11)).fillna(0).cumsum(axis=1)   # works with yearly rows
        for a in range(11):                                       # cumulative citations by the end of year `a` after publication
            d[f"cum{a}"] = d.id.map(A[a])
            d.loc[d[f"cum{a}"].isna() & (d.c == 0), f"cum{a}"] = 0   # never cited: no yearly rows
            d.loc[(d.year < 2012) | (d.year + a > 2025), f"cum{a}"] = np.nan   # yearly counts start in 2012, 2025 is the last full year
        d["w3"] = d.cum2                                          # publication year and the two years after it
        d["aid"] = aid
        return d


    def boot_median(x, n=2000):
        x = np.asarray(x); b = np.median(rng.choice(x, (n, len(x))), axis=1); return np.percentile(b, [2.5, 97.5])


    def within(P, col, cohorts):
        rows = []
        for (aid, R), d in P.groupby(["aid", "R"]):
            for c in cohorts:
                e = d[d.year.between(*c) & d[col].notna()]; s, r = e[e.Category == "survey"], e[e.Category == "research"]
                if len(s) >= 1 and len(r) >= 3 and r[col].mean() > 0:
                    rows.append({"aid": aid, "R": R, "cohort": lab(c), "D": s[col].mean() / r[col].mean()})
        return pd.DataFrame(rows)


    def speed(P):
        """Papers of 2012-2014, followed for eleven years: ratio of mean cumulative citations (surveys / research papers of
    the same researcher) by year after publication, citations in the first three years, and the share of the first
    eleven years' citations received in the first three."""
        rows, txt = [], []
        for g, d in P.items():
            for name, sub in ((g, d), ("R1-R8", d[d.R])) if g == "HCR-listed" else ((g, d),):
                e = sub[sub.year.between(2012, 2014) & sub.cum10.notna()]
                res = []
                for aid, x in e.groupby("aid"):
                    s, r = x[x.Category == "survey"], x[x.Category == "research"]
                    if len(s) >= 1 and len(r) >= 3 and r.cum10.mean() > 0:
                        res.append({**{a: s[f"cum{a}"].mean() / r[f"cum{a}"].mean() if r[f"cum{a}"].mean() > 0 else np.nan for a in range(11)},
                                    "s3": s.cum2.mean(), "r3": r.cum2.mean()})
                X = pd.DataFrame(res)
                rows += [{"group": name, "years after publication": a, "median ratio": X[a].median(), "n": int(X[a].notna().sum())} for a in range(11)]
                f = e[e.cum10 > 0].assign(early=lambda z: z.cum2 / z.cum10).groupby("Category").early.median()
                txt.append(f"- {name} (n={len(X)}): ratio in year 0 {X[0].median():.2f}, by year 2 {X[2].median():.2f}, by year 10 {X[10].median():.2f}. "
                           f"First three years: {X.s3.median():.0f} citations per survey vs {X.r3.median():.0f} per research paper (medians over researchers). "
                           f"Share of the first eleven years' citations in the first three: surveys {100 * f['survey']:.1f}%, research papers {100 * f['research']:.1f}%")
        return pd.DataFrame(rows), txt


    def career(P):
        """3-year-window dividend for papers written within ten years of the researcher's first publication vs later."""
        S = pd.read_csv(os.path.join(ST, "fellows2026", "per_person_scored_INTERNAL.csv"))
        start = S.dropna(subset=["career start"]).set_index("OpenAlex ID")["career start"].to_dict()
        txt = []
        for g, d in P.items():
            e = d[d.w3.notna() & d.aid.isin(start)].copy(); e["early"] = (e.year - e.aid.map(start)) <= 10
            res = []
            for (aid, ec), x in e.groupby(["aid", "early"]):
                s, r = x[x.Category == "survey"], x[x.Category == "research"]
                if len(s) >= 1 and len(r) >= 3 and r.w3.mean() > 0:
                    res.append({"early": ec, "D": s.w3.mean() / r.w3.mean()})
            X = pd.DataFrame(res)
            a, b = X[X.early].D, X[~X.early].D
            txt.append(f"- {g}: first ten career years median D {a.median():.2f} (n={len(a)}, D>1 for {100 * (a > 1).mean():.0f}%), "
                       f"later {b.median():.2f} (n={len(b)}, {100 * (b > 1).mean():.0f}%)")
        return txt


    def main():
        os.makedirs(OUT, exist_ok=True)
        rids = set(r_sets()["ids"]); md = []
        P = {g: pd.concat([papers(a, f).assign(R=(g == "HCR-listed" and a in rids)) for a, f in L if os.path.exists(f)]) for g, L in groups().items()}
        for g, d in P.items():
            print(f"{g}: {d.aid.nunique()} researchers, {len(d)} papers 2000-2023, window missing for "
                  f"{100 * d[d.year >= 2012].w3.isna().mean():.1f}% of 2012-2023 papers", flush=True)

        # (A) within researcher
        rows, Wall = [], []
        for g, d in P.items():
            for name, sub in ((g, d), ("R1-R8", d[d.R])) if g == "HCR-listed" else ((g, d),):
                for col, cohorts, kind in (("w3", WIN, "3-year window"), ("c", WIN, "total citations"), ("c", ALL, "total citations, 2000-2023")):
                    W = within(sub, col, cohorts)
                    for c in cohorts:
                        x = W[W.cohort == lab(c)].D
                        if len(x) == 0: continue
                        lo, hi = boot_median(x) if len(x) >= 5 else (np.nan, np.nan)
                        rows.append({"group": name, "measure": kind, "cohort": lab(c), "n": len(x), "median D": x.median(),
                                     "CI low": lo, "CI high": hi, "share D>1": (x > 1).mean()})
                    if kind == "3-year window":
                        a, b = W[W.cohort == lab(WIN[0])].set_index("aid").D, W[W.cohort == lab(WIN[-1])].set_index("aid").D
                        k = a.index.intersection(b.index)
                        if len(k) >= 5:
                            md.append(f"- {name}, 3-year window, same researchers in {lab(WIN[0])} and {lab(WIN[-1])}: n={len(k)}, median D "
                                      f"{a[k].median():.2f} -> {b[k].median():.2f}, D rose for {int((b[k] > a[k]).sum())}, paired Wilcoxon p={wilcoxon(a[k], b[k]).pvalue:.2g}")
                        Wall.append(W.assign(group=name))
        A = pd.DataFrame(rows); A.to_csv(os.path.join(OUT, "within.csv"), index=False)
        pd.concat(Wall).to_csv(os.path.join(OUT, "within_researcher.csv"), index=False)

        fmt = lambda df: df.to_markdown(index=False, floatfmt=".2f")
        out = ["# Survey citation dividend by publication cohort", "", "## (A) Within researcher (median over researchers of mean survey / mean research citations)", ""]
        for kind in ("3-year window", "total citations", "total citations, 2000-2023"):
            t = A[A.measure == kind].pivot(index="group", columns="cohort", values="median D")
            n = A[A.measure == kind].pivot(index="group", columns="cohort", values="n")
            sh = A[A.measure == kind].pivot(index="group", columns="cohort", values="share D>1")
            out += [f"### {kind}: median D", "", fmt(t.reset_index()), "", "researchers (n)", "", fmt(n.reset_index()), "", "share with D > 1", "", fmt(sh.reset_index()), ""]
        ci = A[A.measure == "3-year window"][["group", "cohort", "median D", "CI low", "CI high"]]
        out += ["### 3-year window, bootstrap 95% intervals of the median", "", fmt(ci), "", *md, ""]
        V, vt = speed(P); V.to_csv(os.path.join(OUT, "speed.csv"), index=False)
        out += ["## (B) Timing: papers of 2012-2014 followed for eleven years", "", *vt, "", "## (C) Career stage (3-year window, 2012-2023)", "", *career(P), ""]
        open(os.path.join(OUT, "results.md"), "w").write("\n".join(out)); print("\n".join(out))


    main()


# =====================================================================================================================
# Survey dependence under keyword rules
# =====================================================================================================================

def keyword_rules():
    """Survey dependence of the HCR-listed researchers under keyword rules, against the SciGenre.

    For each HCR-listed researcher in hcr_study/outliers/survey_dependence_INTERNAL.csv (analyses.py survey-outliers), over all
    records of the classified profile:
      * share of citations from papers flagged (R_C) by the SciGenre, by the title keyword rule
        (common.PAPER_TITLE_KEYWORDS), and by the title-or-abstract rule (the title rule, or survey/review/overview
        and their inflections in the abstract);
      * share of papers flagged by each.
    Medians over researchers.
    Output: hcr_study/keyword_rules/stats.md"""
    OUT = os.path.join(STUDY, "keyword_rules")
    os.makedirs(OUT, exist_ok=True)
    WORD = re.compile(r"\b(?:survey|surveys|surveyed|review|reviews|reviewed|reviewing|overview)\b", re.I)
    lines = []
    def out(s=""):
        print(s); lines.append(s)

    sd = pd.read_csv(os.path.join(STUDY, "outliers", "survey_dependence_INTERNAL.csv"))
    ids = sd[sd.group == "listed"].id.tolist()
    rows = []
    for i in ids:
        f = os.path.join(STUDY, "classified", f"{i}.csv")
        if not os.path.exists(f): continue
        g = pd.read_csv(f).drop_duplicates("id")  # all records, as in the paper's indicators
        tt = g.title.map(clean_text); aa = g.abstract.map(clean_text)
        tr = tt.map(lambda x: bool(PAPER_TITLE_KEYWORDS.search(x)))
        wr = tr | aa.map(lambda x: bool(WORD.search(x)))
        c = g.citationCount.fillna(0); C = c.sum()
        rows.append({"model": c[g.Category == "survey"].sum() / C, "title": c[tr].sum() / C, "word": c[wr].sum() / C,
                     "p_m": (g.Category == "survey").mean(), "p_t": tr.mean(), "p_w": wr.mean()})
    r = pd.DataFrame(rows)
    out(f"## Survey share of citations of the {len(r)} HCR-listed researchers (all records in the denominator, medians)")
    out(f"- SciGenre {r.model.median():.1%} | title rule {r.title.median():.1%} | title or abstract word {r.word.median():.1%}")
    out(f"- share of papers flagged: SciGenre {r.p_m.median():.1%} | title rule {r.p_t.median():.1%} | word rule {r.p_w.median():.1%}")
    open(os.path.join(OUT, "stats.md"), "w").write("\n".join(lines) + "\n")


# =====================================================================================================================
# Abstract sources
# =====================================================================================================================

def abstract_sources():
    """Does the abstract source change the classification?
    1. Cross-source consistency: a random 1,000 HCR-study papers with an OpenAlex abstract and a DOI; their Scopus
       (Search API, view COMPLETE) and Semantic Scholar abstracts are fetched, each version is classified by the final
       model, and agreement is measured (share agreeing, Cohen's kappa, mean |difference| in P(survey)).
    2. Source as a covariate: over all classified papers, survey rate and mean confidence |P - 0.5| by abstract source.
    Keys ELSEVIER_API_KEY and SEMANTIC_SCHOLAR_API_KEY from .env (never printed).
    Output: hcr_study/abstracts/source_check.{json,md}"""
    load_dotenv(os.path.join(HERE, ".env"), override=True)
    EK = os.getenv("ELSEVIER_API_KEY", "").strip(); S2K = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "").strip()
    OUT = os.path.join(HERE, "hcr_study", "abstracts"); CACHE = os.path.join(OUT, "source_check_cache.json")
    GROUP_DIRS = {"listed": "classified", "matched": "controls/classified", "reference": "reference/classified",
                  "boundary": "boundary/classified", "boundary_ai": "boundary_artificialin/classified"}

    # ---------- 1. cross-source consistency ----------
    fr = [pd.read_csv(f, usecols=["title", "doi", "abstract", "type", "venue"]) for f in glob.glob(os.path.join(HERE, "hcr_study", "profiles", "*.csv"))]
    P = pd.concat(fr, ignore_index=True)
    P["doi"] = P.doi.fillna("").str.replace("https://doi.org/", "", regex=False).str.lower()
    P = P[(P.doi != "") & P.abstract.fillna("").map(profiles.has_abstract) & ~is_non_paper(prepare_frame(P))].drop_duplicates("doi")
    S = P.sample(1000, random_state=2026).reset_index(drop=True)
    cache = json.load(open(CACHE)) if os.path.exists(CACHE) else {}
    sc = cache.setdefault("scopus", {}); todo = [d for d in S.doi if d not in sc]     # incremental: fetch DOIs not cached yet
    if todo:
        for i in range(0, len(todo), 25):
            chunk = todo[i:i + 25]
            for d in chunk: sc.setdefault(d, "")
            for attempt in range(6):
                r = requests.get("https://api.elsevier.com/content/search/scopus", params={"query": " OR ".join(f'DOI("{d}")' for d in chunk),
                                 "view": "COMPLETE", "count": 25}, headers={"X-ELS-APIKey": EK, "Accept": "application/json"}, timeout=90)
                if r.status_code == 200: break
                time.sleep(5 * 2 ** attempt)
            for e in r.json().get("search-results", {}).get("entry", []):
                if "error" not in e: sc[str(e.get("prism:doi", "")).lower()] = e.get("dc:description") or ""
            time.sleep(0.5)
    s2 = cache.setdefault("s2", {}); todo = [d for d in S.doi if d not in s2]
    if todo:
        for i in range(0, len(todo), 500):
            chunk = todo[i:i + 500]
            for attempt in range(6):
                r = requests.post("https://api.semanticscholar.org/graph/v1/paper/batch", params={"fields": "abstract"},
                                  json={"ids": ["DOI:" + d for d in chunk]}, headers={"x-api-key": S2K} if S2K else {}, timeout=90)
                if r.status_code == 200: break
                time.sleep(5 * 2 ** attempt)
            for d, x in zip(chunk, r.json()): s2[d] = (x or {}).get("abstract") or "" if isinstance(x, dict) or x is None else ""
            time.sleep(1.1)
    json.dump(cache, open(CACHE, "w"))
    S["scopus"] = S.doi.map(lambda d: cache["scopus"].get(d, "")); S["s2"] = S.doi.map(lambda d: cache["s2"].get(d, ""))
    probs = {"openalex": profiles.proba(S.title.tolist(), S.abstract.tolist())}
    for src in ("scopus", "s2"):
        ok = S[src].map(profiles.has_abstract)
        p = np.full(len(S), np.nan); p[ok.values] = profiles.proba(S.title[ok].tolist(), S[src][ok].tolist()); probs[src] = p
    res = {"n": len(S)}
    for src in ("scopus", "s2"):
        ok = ~np.isnan(probs[src]); a, b = probs["openalex"][ok] >= profiles.THRESHOLD, probs[src][ok] >= profiles.THRESHOLD
        res[src] = {"n_both": int(ok.sum()), "agreement": float((a == b).mean()), "kappa": float(cohen_kappa_score(a, b)),
                    "mean_abs_diff_p": float(np.abs(probs["openalex"][ok] - probs[src][ok]).mean()),
                    "survey_rate_openalex": float(a.mean()), f"survey_rate_{src}": float(b.mean()),
                    "flips_to_survey": int((~a & b).sum()), "flips_to_research": int((a & ~b).sum())}

    # ---------- 2. source as a covariate (all classified papers) ----------
    rows = []
    for g, d in GROUP_DIRS.items():
        for f in glob.glob(os.path.join(HERE, "hcr_study", d, "*.csv")):
            x = pd.read_csv(f, usecols=lambda c: c in ("Category", "SurveyScore", "Abstract source"))
            if "Abstract source" in x: rows.append(x.assign(group=g))
    if rows:
        C = pd.concat(rows, ignore_index=True); C = C[C.Category.isin(["survey", "research"])]
        cov = C.groupby("Abstract source").agg(papers=("Category", "size"), survey_rate=("Category", lambda s: (s == "survey").mean()),
                                               mean_confidence=("SurveyScore", lambda p: (p - 0.5).abs().mean()))
        res["by_source"] = cov.round(4).reset_index().to_dict("records")
    json.dump(res, open(os.path.join(OUT, "source_check.json"), "w"), indent=1)
    md = [f"Cross-source consistency on {res['n']} random papers with an OpenAlex abstract (final model, threshold {profiles.THRESHOLD}):"]
    for src in ("scopus", "s2"):
        r = res[src]
        md.append(f"- OpenAlex vs {src}: {r['n_both']} with both; agreement {100*r['agreement']:.1f}%, kappa {r['kappa']:.3f}, "
                  f"mean |dP| {r['mean_abs_diff_p']:.3f}; flips to survey {r['flips_to_survey']}, to research {r['flips_to_research']}")
    if "by_source" in res: md += ["", pd.DataFrame(res["by_source"]).to_markdown(index=False)]
    open(os.path.join(OUT, "source_check.md"), "w").write("\n".join(md)); print("\n".join(md))


# =====================================================================================================================
# Remaining numbers quoted in the paper
# =====================================================================================================================

def paper_numbers():
    """Numbers quoted in the paper that no analysis script prints directly: the list's Tukey fences, the RQ4 expectation
    and robustness comparisons, the matched comparisons with Holm correction, and R1-R8 ranges.
    Inputs: hcr_study/outliers/model_scores_INTERNAL.csv (analyses.py outlier-model), survey_dependence_INTERNAL.csv
    (analyses.py survey-outliers), per_member.csv, peers/ (comparison_groups.py peers), v2_top10_key_INTERNAL.csv (figures.py hcr),
    history/ (profiles.py history), hcr_any_list.csv (comparison_groups.py any-list).   Output: hcr_study/paper_numbers.md"""
    S = os.path.join(HERE, "hcr_study")
    M = pd.read_csv(os.path.join(S, "outliers", "model_scores_INTERNAL.csv"))
    SD = pd.read_csv(os.path.join(S, "outliers", "survey_dependence_INTERNAL.csv"))
    out = []
    def say(k, v): out.append(f"- {k}: {v}")
    L, R, Mt = (M[M.group == g] for g in ("listed", "reference", "matched"))
    Ls, Rs, Ms = L, R, Mt                       # model_scores has C, H, T for all three groups
    Lx = SD[SD.group == "listed"].set_index("id")   # top-1% papers without surveys

    # Tukey fences within the list
    def fence(x): q1, q3 = np.nanpercentile(x, [25, 75]); return q3 + 1.5 * (q3 - q1)
    fC, fH, fT = fence(Ls.C), fence(Ls.H), fence(Ls["T"])
    say("Tukey fence R_C (list)", f"{fC:.1f}%; HCR-listed above: {int((Ls.C > fC).sum())} ({Ls.C[Ls.C > fC].min():.0f}-{Ls.C[Ls.C > fC].max():.0f}%); "
        f"matched above: {int((Ms.C > fC).sum())} of {len(Ms)}; reference above: {int((Rs.C > fC).sum())} of {len(Rs)}")
    say("Tukey fence R_h (list)", f"{fH:.1f}%; HCR-listed above: {int((Ls.H > fH).sum())} ({Ls.H[Ls.H > fH].min():.0f}-{Ls.H[Ls.H > fH].max():.0f}%)")
    say("Tukey fence top-1% survey share (list)", f"{fT:.1f}%; HCR-listed above: {int((Ls['T'] > fT).sum())} ({Ls['T'][Ls['T'] > fT].min():.0f}-{Ls['T'][Ls['T'] > fT].max():.0f}%)")
    # R1-R8 = the records with more than half of their citations from surveys, in the order of figures.py hcr (v2_top10_key_INTERNAL.csv)
    K = pd.read_csv(os.path.join(S, "v2_top10_key_INTERNAL.csv")); order = {k: i for i, k in enumerate(K["key"])}
    top10 = Ls[Ls["key"].isin(order)].assign(o=lambda z: z["key"].map(order)).sort_values("o").reset_index(drop=True)
    assert len(top10) == 8, len(top10)
    say("R1-R8 beyond each fence (C, H, T)", f"{[(int(r.C > fC), int(r.H > fH), int(r.T > fT)) for r in top10.itertuples()]}")
    say("R1-R8: R_h range", f"{top10.H.min():.0f}-{top10.H.max():.0f}%; beyond the R_h fence: {int((top10.H > fH).sum())}, "
        f"beyond the top-1% fence: {int((top10['T'] > fT).sum())}, beyond the R_C fence: {int((top10.C > fC).sum())}")
    say("R1-R8: survey % of papers / citations", f"{top10.p.min():.0f}-{top10.p.max():.0f}% / {top10.C.min():.0f}-{top10.C.max():.0f}%")
    say("R1-R8: top-1% papers without surveys (min)", f"{int(Lx.loc[top10.id, 'top1_x'].min())}")
    say("max residual z, HCR-listed", f"{L.z.max():.2f}; records with R_C >= 75%: z = {', '.join(f'{z:.1f}' for z in L[L.C >= 75].z)}")

    # RQ4: expectation and robustness
    say("expected R_C (median), HCR / matched / reference", f"{L.C_expected.median():.1f} / {Mt.C_expected.median():.1f} / {R.C_expected.median():.1f}")
    say("HCR-listed: observed minus expected (median of differences)", f"{(L.C - L.C_expected).median():.1f} points; above expectation: {100 * (L.z > 0).mean():.0f}%")
    say("median R_C, HCR / matched / reference", f"{L.C.median():.1f} / {Mt.C.median():.1f} / {R.C.median():.1f}")
    lo = L.cites.quantile(0.1); Rh = R[R.cites >= lo]
    say("reference restricted to >= HCR 10th percentile of Scopus citations", f"n={len(Rh)}, median R_C {Rh.C.median():.1f} vs {L.C.median():.1f}, "
        f"two-sided p={mannwhitneyu(L.C, Rh.C).pvalue:.1g}")
    for sub, d in L.groupby("sub"):
        r = R[R["sub"] == sub]
        if len(d) >= 5 and len(r) >= 5:
            say(f"residual z within subfield group {sub}", f"HCR n={len(d)} median {d.z.median():.2f} vs reference n={len(r)} {r.z.median():.2f}, "
                f"two-sided p={mannwhitneyu(d.z, r.z).pvalue:.2g}")
    for c, lab in (("p", "survey % of papers"), ("D", "dividend"), ("C", "R_C"), ("T", "survey % of top-1%"), ("H", "R_h"), ("z", "residual z")):
        a, b, m = L[c].dropna(), R[c].dropna(), Mt[c].dropna()
        say(f"Table reference row: {lab}", f"{a.median():.1f} / {m.median():.1f} / {b.median():.1f}; p vs ref {mannwhitneyu(a, b).pvalue:.1g}, "
            f"vs matched {mannwhitneyu(a, m).pvalue:.1g} (two-sided)")
    q = SD[SD.group != "reference"]; Ld = L.set_index("id"); Rd = R
    T10 = lambda d: d[d.top1 >= 10]
    say("top-1% share, researchers with >= 10 top-1% papers", f"{T10(L)['T'].median():.1f} (n={len(T10(L))}) / {T10(Mt)['T'].median():.1f} (n={len(T10(Mt))}) / {T10(R)['T'].median():.1f} (n={len(T10(R))})")
    D = L.D.dropna(); say("dividend D, HCR-listed: median, IQR", f"{D.median():.2f} ({D.quantile(.25):.1f} to {D.quantile(.75):.1f}); D > 1 for {int((L.D > 1).sum())} of {len(L)}")
    # volume vs dividend decomposition at the medians
    rc = lambda p, d: 100 * p * d / (p * d + 1 - p)
    pL, dL, pR, dR = L.p.median() / 100, L.D.median(), R.p.median() / 100, R.D.median()
    say("decomposition at medians (R_C)", f"HCR {rc(pL, dL):.1f}, reference {rc(pR, dR):.1f}, HCR volume with reference dividend {rc(pL, dR):.1f}, "
        f"reference volume with HCR dividend {rc(pR, dL):.1f}")

    # matched peers: each HCR-listed researcher against the mean of their peers; paired Wilcoxon, rank-biserial r, Holm
    # across four measures
    P = comparison_groups.load_peers()
    key = L.set_index("key")["id"].to_dict()
    pairs = pd.DataFrame({"hcr": P["HCR name"].map(key), "ctrl": P["OpenAlex ID"].values}).dropna()
    A, B = L.set_index("id"), Mt.set_index("id")
    pairs = pairs[pairs.hcr.isin(A.index) & pairs.ctrl.isin(B.index)]


    def paired(X, Y, c, prs, fill=None):
        """the HCR-listed researcher's value and the mean of their peers' values (peers with a value)"""
        y = Y.loc[prs.ctrl, c]
        y = y.fillna(fill) if fill is not None else y
        ym = pd.Series(y.values, index=prs.hcr.values).groupby(level=0).mean().dropna()
        x = X.loc[ym.index, c]
        x = x.fillna(fill) if fill is not None else x
        return x.values, ym.values


    def holm(ps):
        o = np.argsort(ps); adj = np.empty(len(ps)); run = 0
        for rank, k in enumerate(o): run = max(run, min(1, (len(ps) - rank) * ps[k])); adj[k] = run
        return adj


    res = []
    for c, lab in (("C", "R_C"), ("p", "survey % of papers"), ("T", "survey % of top-1%"), ("H", "R_h")):
        x, y = paired(A, B, c, pairs); ok = ~np.isnan(x); x, y = x[ok], y[ok]; d = x - y; d = d[d != 0]
        rk = pd.Series(np.abs(d)).rank().values; rbc = (rk[d > 0].sum() - rk[d < 0].sum()) / rk.sum()
        res.append((lab, wilcoxon(x, y).pvalue, rbc, np.median(x), np.median(y), len(x), int((x > y).sum())))
    for (lab, p, rbc, mx, my, n, k), pa in zip(res, holm([r[1] for r in res])):
        say(f"matched peers, {lab}", f"n={n} HCR-listed researchers, medians {mx:.1f} vs {my:.1f} (peers' mean), HCR-listed higher for {k}, "
            f"paired Wilcoxon p={p:.2g}, Holm {pa:.2g}, rank-biserial r={rbc:.2f}")
    x, y = paired(A, B, "D", pairs); ok = ~np.isnan(x)
    say("matched peers, dividend D (secondary)", f"n={ok.sum()}, medians {np.median(x[ok]):.2f} vs {np.median(y[ok]):.2f} (peers' mean), "
        f"paired Wilcoxon p={wilcoxon(x[ok], y[ok]).pvalue:.2g}")
    say("matched peers", f"{len(pairs)} peers for {pairs.hcr.nunique()} HCR-listed researchers; peers per researcher: "
        + ", ".join(f"{k}: {v}" for k, v in pairs.groupby('hcr').size().value_counts().sort_index().items()))
    # Tukey fence, top-1% share and majority-survey records among the peers (all peers, each once)
    say("matched peers beyond the list's R_C fence", f"{int((Mt.C > fC).sum())} of {len(Mt)}")
    for g, e in (("HCR-listed", L), ("matched peers", Mt), ("random sample", R)):
        t = T10(e)
        say(f"{g} with >= 10 top-1% papers, surveys the majority of them", f"{int((t['T'] > 50).sum())} of {len(t)}")
    say("dividend D defined (at least one survey)", f"HCR-listed {int(L.D.notna().sum())}, matched peers {int(Mt.D.notna().sum())}, "
        f"random sample {int(R.D.notna().sum())}")

    # Holm family with top-1% papers without surveys: R_C, survey % of papers, survey % of top-1% (missing = 0), top-1% papers without surveys
    SDl, SDn = SD[SD.group == "listed"].set_index("id"), SD[SD.group == "non-listed"].drop_duplicates("id").set_index("id")
    pairs2 = pairs[pairs.hcr.isin(SDl.index) & pairs.ctrl.isin(SDn.index)]
    fam = []
    for c, lab, src in (("C", "R_C", "M"), ("p", "survey % of papers", "M"), ("T", "survey % of top-1% (missing = 0)", "SD"), ("top1_x", "top-1% papers without surveys", "SD")):
        X, Y = (A, B) if src == "M" else (SDl, SDn)
        x, y = paired(X, Y, c, pairs2, fill=0)
        fam.append((lab, wilcoxon(x, y).pvalue, np.median(x), np.median(y)))
    for (lab, p, mx, my), pa in zip(fam, holm([f[1] for f in fam])):
        out.append(f"- matched comparison Holm family, {lab}: medians {mx:.1f} vs {my:.1f}, p={p:.2g}, Holm {pa:.2g}")
    # sensitivity: peers whose name appears on any HCR list 2014-2025 (any category) excluded
    ev = comparison_groups.ever_listed(P).values
    nv = pairs[~pairs.ctrl.isin(set(P.loc[ev, "OpenAlex ID"]))]
    x, y = paired(A, B, "C", nv)
    out.append(f"- matched peers without anyone named on an HCR list 2014-2025: {len(nv)} peers (excluded {len(pairs) - len(nv)}) for {len(x)} "
               f"HCR-listed researchers, R_C {np.median(x):.1f} vs {np.median(y):.1f}, paired Wilcoxon p={wilcoxon(x, y).pvalue:.2g}")
    # sensitivity: one peer per HCR-listed researcher, matched on Scopus citations alone (comparison_groups.py controls)
    M1 = M[M.group == "matched_1to1"].set_index("id")
    p1 = pd.DataFrame({"hcr": M1["hcr"].map(key), "ctrl": M1.index}).dropna()
    x, y = paired(A, M1, "C", p1)
    out.append(f"- one-to-one citation-matched peers: n={len(x)} pairs, R_C {np.median(x):.1f} vs {np.median(y):.1f}, HCR-listed higher for "
               f"{int((x > y).sum())}, paired Wilcoxon p={wilcoxon(x, y).pvalue:.2g}")
    # share of the HCR-listed researchers' top-1% advantage over their peers that comes from surveys, with a 95% bootstrap
    # interval over HCR-listed researchers: A = 1 - mean(advantage without surveys) / mean(advantage with surveys)
    xb, yb = paired(SDl, SDn, "top1", pairs2)
    xa, ya = paired(SDl, SDn, "top1_x", pairs2)
    A_ = lambda i: 100 * (1 - (xa[i] - ya[i]).mean() / (xb[i] - yb[i]).mean())
    rng = np.random.default_rng(2026); boots = [A_(rng.integers(0, len(xb), len(xb))) for _ in range(2000)]
    out.append(f"- share of the top-1% advantage over matched peers from surveys: A = {A_(np.arange(len(xb))):.1f}% "
               f"(95% CI {np.percentile(boots, 2.5):.0f} to {np.percentile(boots, 97.5):.0f}%), n={len(xb)} HCR-listed researchers")
    open(os.path.join(S, "paper_numbers.md"), "w").write("\n".join(out)); print("\n".join(out))

    # ---------- additions after the pre-submission review ----------
    lg = lambda c: np.log(np.clip(c, 0.5, 99.5) / (100 - np.clip(c, 0.5, 99.5)))
    il = lambda z: 100 / (1 + np.exp(-z))
    # the abstract's "a fifth"
    f5 = L[L.C >= 41]
    out.append(f"- HCR-listed with R_C >= 41%: {len(f5)} of {len(L)} ({100 * len(f5) / len(L):.0f}%), range {f5.C.min():.1f}-{f5.C.max():.1f}%")
    # the survey share at which z = 3 (residual scale from model_stats.md)
    s_res = float(re.search(r"residual scale s = ([0-9.]+)", open(os.path.join(S, "outliers", "model_stats.md")).read()).group(1))
    need = il(lg(L.C_expected) + 3 * s_res)
    out.append(f"- R_C at which z = 3, HCR-listed: median {need.median():.1f}% (range {need.min():.1f}-{need.max():.1f}%); highest R_C in any group "
               f"{M[M.group.isin(['listed', 'matched', 'reference'])].C.max():.1f}%")
    # records beyond the list's R_C fence: Fisher's exact tests
    for g, A_ in (("matched peers", Mt), ("random sample", R)):
        a, b = int((L.C > fC).sum()), int((A_.C > fC).sum())
        out.append(f"- beyond the R_C fence, HCR-listed {a}/{len(L)} vs {g} {b}/{len(A_)}: Fisher two-sided p={fisher_exact([[a, len(L) - a], [b, len(A_) - b]])[1]:.2g}")
    # pooled quantile regressions of logit(R_C) with an indicator for HCR-listed researchers (HCR-listed and random sample)
    Q = M[M.group.isin(["listed", "reference"])].rename(columns={"C": "RC"}).copy()
    Q["y"], Q["lc"], Q["yr"], Q["hcr"] = lg(Q.RC), np.log10(Q.cites.astype(float)), (Q.firstyr - 2000) / 10, (Q.group == "listed").astype(int)
    Q["ai"], Q["oth"] = (Q["sub"] == "ai").astype(int), (Q["sub"] == "other").astype(int)
    FQ = "y ~ hcr + lc + yr + ai + oth"
    for tau in (0.1, 0.25, 0.5, 0.75, 0.9):
        r = smf.quantreg(FQ, Q).fit(q=tau, max_iter=5000)
        out.append(f"- pooled quantile regression, tau {tau}: HCR coefficient {r.params['hcr']:.2f} (95% CI {r.conf_int().loc['hcr', 0]:.2f} to "
                   f"{r.conf_int().loc['hcr', 1]:.2f}), p={r.pvalues['hcr']:.2g}")
    rng_q = np.random.default_rng(7); bq = []
    gl, gr = Q[Q.hcr == 1], Q[Q.hcr == 0]
    for _ in range(1000):
        bs_ = pd.concat([gl.sample(len(gl), replace=True, random_state=rng_q.integers(1 << 31)), gr.sample(len(gr), replace=True, random_state=rng_q.integers(1 << 31))])
        bq.append(smf.quantreg(FQ, bs_).fit(q=0.5, max_iter=5000).params["hcr"])
    bq = np.array(bq)
    out.append(f"- pooled median regression, bootstrap stratified by group (1,000): 95% CI {np.percentile(bq, 2.5):.2f} to {np.percentile(bq, 97.5):.2f}, "
               f"resamples <= 0: {(bq <= 0).sum()}")
    Lz, Rz = L.z, R.z
    out.append(f"- residual z quantiles (10/25/50/75/90), HCR-listed: {np.round(Lz.quantile([.1, .25, .5, .75, .9]).values, 2).tolist()}; random sample: "
               f"{np.round(Rz.quantile([.1, .25, .5, .75, .9]).values, 2).tolist()}; HCR-listed below the random sample's 25th percentile: "
               f"{100 * (Lz < Rz.quantile(.25)).mean():.1f}%, above its 95th: {100 * (Lz > Rz.quantile(.95)).mean():.1f}%")
    # comparison at equal counts of top-1% papers, Clarivate's selection variable (non-listed: matched peers and random sample, each once)
    N_ = M[M.group.isin(["matched", "reference"])].drop_duplicates("id")
    for t in (L.top1.quantile(0.1), L.top1.median()):
        a, b = L[L.top1 >= t], N_[N_.top1 >= t]
        out.append(f"- at >= {t:.0f} top-1% papers: HCR-listed n={len(a)} median R_C {a.C.median():.1f} vs non-listed n={len(b)} {b.C.median():.1f}, "
                   f"two-sided p={mannwhitneyu(a.C, b.C).pvalue:.2g}")
    T_ = pd.concat([L, N_]).rename(columns={"C": "RC"}); T_ = T_[T_.top1 > 0].copy()
    T_["y"], T_["lt"], T_["yr"], T_["hcr"] = lg(T_.RC), np.log10(T_.top1), (T_.firstyr - 2000) / 10, (T_.group == "listed").astype(int)
    T_["ai"], T_["oth"] = (T_["sub"] == "ai").astype(int), (T_["sub"] == "other").astype(int)
    r = smf.quantreg("y ~ hcr + lt + yr + ai + oth", T_).fit(q=0.5, max_iter=5000)
    out.append(f"- median regression on log top-1% papers, first year, subfield: HCR coefficient {r.params['hcr']:.2f} (95% CI "
               f"{r.conf_int().loc['hcr', 0]:.2f} to {r.conf_int().loc['hcr', 1]:.2f}), p={r.pvalues['hcr']:.2g}, n={len(T_)}")
    # matched peers: paired difference in logit(R_C) adjusted for the differences in log papers and log citations
    els_ = pd.read_csv(os.path.join(S, "BTAS_HCR_Computer_Science.csv")).set_index("HCR name")
    Pd = P.assign(ly=lg(P["Survey % of citations"]), lp=np.log(P.np6024.astype(float)), lcit=np.log(P["nc9624 (ns)"].astype(float)))
    Pm = Pd.groupby("HCR name")[["ly", "lp", "lcit"]].mean()
    Lk = L.set_index("key")
    Pm["dy"] = lg(Lk.loc[Pm.index, "C"]).values - Pm.ly
    Pm["dp"] = np.log(els_.loc[Pm.index, "np6024"].astype(float)).values - Pm.lp
    Pm["dc"] = np.log(els_.loc[Pm.index, "nc9624 (ns)"].astype(float)).values - Pm.lcit
    r = smf.quantreg("dy ~ dp + dc", Pm).fit(q=0.5, max_iter=5000)
    out.append(f"- matched peers, paired median regression of the logit difference on differences in log papers and log citations: intercept "
               f"{r.params['Intercept']:.2f} (95% CI {r.conf_int().loc['Intercept', 0]:.2f} to {r.conf_int().loc['Intercept', 1]:.2f}), unadjusted median "
               f"{Pm.dy.median():.2f}, n={len(Pm)}")
    # sensitivity: survey share of citations without magazine records (venues named magazine, IEEE Network, IEEE Wireless Communications)
    MAG = re.compile(r"magazine|^ieee network$|^ieee wireless communications$", re.I)
    def no_mag(path):
        d = pd.read_csv(path, usecols=["venue", "citationCount", "Category"])
        c, sv, m = d.citationCount.fillna(0), d.Category == "survey", d.venue.fillna("").str.strip().str.match(MAG)
        return 100 * c[sv & ~m].sum() / max(c[~m].sum(), 1), 100 * c[sv & m].sum() / max(c[sv].sum(), 1)
    pm_ = pd.read_csv(os.path.join(S, "per_member.csv")); pm_ = pm_[pm_["Consistent with Scopus"]]
    rf_ = pd.read_csv(os.path.join(S, "reference", "per_reference.csv")); rf_ = rf_[rf_["Consistent with Scopus"]]
    NM = {"HCR-listed": [no_mag(os.path.join(S, "classified", f"{i}.csv")) for i in pm_["OpenAlex ID"]],
          "random sample": [no_mag(os.path.join(S, "reference", "classified", f"{i}.csv")) for i in rf_.openalex_id
                            if os.path.exists(os.path.join(S, "reference", "classified", f"{i}.csv"))],
          "matched peers": [no_mag(p_) for p_ in P.path]}
    NM = {k: np.array(v) for k, v in NM.items()}
    out.append("- without magazine records, median R_C: " + ", ".join(f"{k} {np.median(v[:, 0]):.1f}%" for k, v in NM.items())
               + f"; HCR-listed vs random sample two-sided p={mannwhitneyu(NM['HCR-listed'][:, 0], NM['random sample'][:, 0]).pvalue:.2g}, "
               f"vs matched peers p={mannwhitneyu(NM['HCR-listed'][:, 0], NM['matched peers'][:, 0]).pvalue:.2g}; magazine share of HCR-listed researchers' "
               f"survey citations: median {np.median(NM['HCR-listed'][:, 1]):.1f}%, mean {np.mean(NM['HCR-listed'][:, 1]):.1f}%")
    # sensitivity: identity resolution (OpenAlex/Scopus citation ratio)
    ratio = pm_["OpenAlex/Scopus citations"]
    rho, prho = spearmanr(pm_["Survey % of citations"], ratio)
    a, b = pm_[ratio <= 1.5]["Survey % of citations"], rf_[rf_["OpenAlex/Scopus citations"] <= 1.5]["Survey % of citations"]
    out.append(f"- R_C vs OpenAlex/Scopus citation ratio, HCR-listed: Spearman {rho:.2f} (p={prho:.2g}); profiles within 1.5 times their Scopus citations: "
               f"HCR-listed n={len(a)} median {a.median():.1f}% vs random sample n={len(b)} {b.median():.1f}%, two-sided p={mannwhitneyu(a, b).pvalue:.2g}")
    open(os.path.join(S, "paper_numbers.md"), "w").write("\n".join(out)); print("\n".join(out[-22:]))


STEPS = {"survey-outliers": survey_outliers, "yearly-citations": yearly_citations, "outlier-model": outlier_model, "outlier-checks": outlier_model_checks, "composite": composite, "boundary-scenarios": boundary_scenarios, "primary-endpoints": primary_endpoints, "s2-types": s2_types, "s2-rule": s2_rule, "dividend-window": dividend_window, "keyword-rules": keyword_rules, "abstract-sources": abstract_sources, "paper-numbers": paper_numbers}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv.pop(1)]()
