"""Stage 7: the paper's figures (hcr_study/figures/).

    python figures.py hcr        figures of the HCR-listed researchers and their peers
    python figures.py fellows    HCR-listed researchers against ACM and IEEE Fellows
    python figures.py dividend   survey citation dividend
"""
import os
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import FuncFormatter
from scipy.stats import kendalltau, wilcoxon

import common
import comparison_groups
from common import (
    ACM, flag, IEEE, INK, INK_2, LISTED, LW, MATCHED, MS, panel, POINT, r_handle, R_MARK, r_sets, REFERENCE,
    save, SPRINGER, style, WIDE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# Figures of the HCR-listed researchers and their peers
# =====================================================================================================================

def hcr_figures():
    """Figures and statistics of the paper for the 2025 HCR Computer Science list.

    Inputs (all produced with the final model):
        hcr_study/per_member.csv, hcr_study/classified/<ID>.csv          (profiles.py classify)
        hcr_study/peers/ (comparison_groups.py peers: up to five matched peers per HCR-listed researcher)
        hcr_study/composite/composite_INTERNAL.csv                        (analyses.py composite; optional, panel e of
                                                                           fig_top10_relative, so run this script again
                                                                           after analyses.py composite)

    HCR-listed researchers: Scopus-consistent profiles (as in the group tables of profiles.py classify).
    Matched peers: up to five per HCR-listed researcher, passing the same checks; each HCR-listed researcher is
    compared with the mean of their peers.

    Output: hcr_study/figures/*.pdf (+ .png previews)
            hcr_study/v2_stats.md (numbers quoted in the paper)
            hcr_study/v2_top10_key_INTERNAL.csv (R1-R8, read by common.r_sets)"""
    CTRL = os.path.join(STUDY, "peers")
    FIG = os.path.join(STUDY, "figures")
    os.makedirs(FIG, exist_ok=True)

    HCP_START, HCP_END = 2014, 2024
    TOPK_MAX = 50
    TREND_YEARS = range(2010, 2025)

    METHOD_COLS = {
        "SciGenre (ours)": None,                      # Category == survey (final model)
        "Indexer type (Review)": "Survey (indexer type)",
        "Smyth & Cunningham rule": "Survey (Smyth & Cunningham)",
    }

    RNG = np.random.default_rng(42)
    SERIES, SERIES_2, BEFORE = LISTED, MATCHED, REFERENCE


    # ============================================================
    # HELPERS
    # ============================================================

    def boxes(ax, data, positions=None, color=SERIES, width=0.5, fmt="{:.1f}", flags=None):
        """Shared box style (common.boxes): neutral boxes over one dot per researcher in the group's colour."""
        positions = list(positions) if positions is not None else list(range(1, len(data) + 1))
        common.boxes(ax, data, positions, [color] * len(data), flags=flags, width=width, fmt=fmt, rng=RNG)


    def rflags(series_list):
        """R1-R8 masks aligned with Series taken from M (by index)."""
        return [M.loc[s.index, "key"].isin(R_NAMES).values for s in series_list]


    def h_index(c):
        c = np.sort(np.asarray(c))[::-1]
        return int(np.sum(c >= np.arange(1, len(c) + 1)))


    def p_text(p):
        return "p < 0.001" if p < 0.001 else f"p = {p:.3f}"


    def profile_stats(p):
        """All per-profile quantities used below, from one classified profile."""
        p = p.copy()
        p["citationCount"] = p["citationCount"].fillna(0).astype(int)
        survey = (p["Category"] == "survey").values
        research = (p["Category"] == "research").values
        cites = p["citationCount"].values
        window = p["year"].between(HCP_START, HCP_END).values
        top1 = p["top1"].fillna(False).astype(bool).values & window

        order = np.argsort(-cites, kind="stable")
        h = h_index(cites)
        core = order[:h]
        s_sorted = survey[order]
        topk = [s_sorted[:k].mean() * 100 if k <= len(s_sorted) else np.nan for k in range(1, TOPK_MAX + 1)]

        n_auth = p["authors"].fillna("").map(lambda s: len([a for a in str(s).split(";") if a.strip()])).values

        out = {
            "papers": len(p), "surveys": int(survey.sum()),
            "share_papers": 100 * survey.mean(),
            "share_hcore": 100 * survey[core].mean() if h else np.nan,
            "share_top1": 100 * survey[top1].mean() if top1.any() else np.nan,
            "share_cites": 100 * cites[survey].sum() / max(cites.sum(), 1),
            "cites_per_survey": cites[survey].mean() if survey.any() else np.nan,
            "cites_per_research": cites[research].mean() if research.any() else np.nan,
            "authors_survey": np.median(n_auth[survey]) if survey.any() else np.nan,
            "authors_research": np.median(n_auth[research]) if research.any() else np.nan,
            "h": h, "h_x": h_index(cites[~survey]),
            "cites": int(cites.sum()), "cites_x": int(cites[~survey].sum()),
            "top1": int(top1.sum()), "top1_x": int((top1 & ~survey).sum()),
            "topk": topk,
        }
        for method, col in METHOD_COLS.items():
            flag = survey if col is None else p[col].fillna(False).astype(bool).values
            out[f"{method}|cites"] = 100 * cites[flag].sum() / max(cites.sum(), 1)
            out[f"{method}|papers"] = 100 * flag.mean()
            out[f"{method}|h"] = 100 * (h - h_index(cites[~flag])) / max(h, 1)
        return out


    def load_group(meta, id_col, cls_dir):
        rows, works = [], []
        for _, r in meta.iterrows():
            path = os.path.join(cls_dir, f"{r[id_col]}.csv")
            if not os.path.exists(path):
                continue
            p = pd.read_csv(path)
            s = profile_stats(p)
            s["key"] = r["key"]
            rows.append(s)
            works.append(p.assign(_member=r["key"]))
        return pd.DataFrame(rows), pd.concat(works, ignore_index=True)


    # ============================================================
    # DATA
    # ============================================================

    members_meta = pd.read_csv(os.path.join(STUDY, "per_member.csv"))
    members_meta = members_meta[members_meta["Consistent with Scopus"]].copy()
    members_meta["key"] = members_meta["HCR name"]
    M, M_works = load_group(members_meta, "OpenAlex ID", os.path.join(STUDY, "classified"))
    n = len(M)
    print(f"HCR-listed researchers: {n}")

    have_controls = os.path.exists(os.path.join(CTRL, "per_peer.csv"))
    if have_controls:
        ctrl_meta = comparison_groups.peers_as_controls()[1]          # Scopus-consistent peers, up to five per HCR-listed researcher
        ctrl_meta["key"] = ctrl_meta["HCR name (matched member)"]
        ctrl_meta = ctrl_meta[ctrl_meta["key"].isin(M["key"])]
        C, _ = load_group(ctrl_meta, "Control OpenAlex ID", comparison_groups.PEERS_CLASSIFIED)
        # each HCR-listed researcher against the mean of their peers
        pairs = M.merge(C.groupby("key").mean(numeric_only=True).reset_index(), on="key", suffixes=("_m", "_c"))
        print(f"Matched pairs: {len(pairs)}")

    for col, before, after in (("h_drop", "h", "h_x"), ("cites_drop", "cites", "cites_x"), ("top1_drop", "top1", "top1_x")):
        M[col] = np.where(M[before] > 0, 100 * (M[before] - M[after]) / M[before], np.nan)
    M["yield_ratio"] = M["cites_per_survey"] / M["cites_per_research"]
    # R1-R8: the records that draw more than half of their citations from surveys (citation drop > 50% when surveys are
    # excluded = survey share of citations), in order of the drop,
    # from this run's data (the key file is rewritten at the end, so reading it at start-up could be stale)
    R_ORDER = M[M["cites_drop"] > 50].sort_values("cites_drop", ascending=False, kind="stable")["key"].tolist()
    R_NAMES = set(R_ORDER)

    stats = []


    def note(label, value):
        stats.append((label, value))


    # ============================================================
    # FIG: drops per metric
    # ============================================================

    i10 = members_meta.set_index("HCR name")[["i10", "i10*"]]
    M["i10_drop"] = M["key"].map(100 * (i10["i10"] - i10["i10*"]) / i10["i10"])

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    cols = [("h_drop", "h-index"), ("i10_drop", "i10-index"), ("cites_drop", "Citations"),
            ("top1_drop", "Top-1% papers\n(OpenAlex, 2014–2024)")]
    data_ = [M[c].dropna() for c, _ in cols]
    boxes(ax, data_, fmt="{:.1f}%", flags=rflags(data_))
    ax.set_xticks(range(1, 5), [l for _, l in cols])
    ax.set_ylabel("Drop when surveys are excluded (%)")
    ax.set_ylim(0, 100)
    ax.legend(handles=[r_handle()], loc="upper left")
    ax.tick_params(axis="x", length=0)
    style(ax)
    save(fig, "fig_drops")
    for c, l in cols:
        note(f"Median drop, {l.replace(chr(10), ' ')} (%)", M[c].median())
        note(f"IQR drop, {l.replace(chr(10), ' ')} (%)", f"{M[c].quantile(.25):.1f}-{M[c].quantile(.75):.1f}")
    for b, a, l in (("h", "h_x", "h-index"), ("cites", "cites_x", "citations"), ("top1", "top1_x", "top-1% papers")):
        note(f"Wilcoxon one-sided p, {l}", wilcoxon(M[b], M[a], alternative="greater").pvalue)
        note(f"Median {l} before / after", f"{M[b].median():.0f} / {M[a].median():.0f}")


    # ============================================================
    # FIG: survey share by layer of the record
    # ============================================================

    fig, ax = plt.subplots(figsize=(3.5, 2.5), constrained_layout=True)
    layers = [("share_papers", "All papers"), ("share_hcore", "h-core\npapers"),
              ("share_top1", "Top-1% papers\n(OpenAlex, 2014–2024)"), ("share_cites", "Citations")]
    data_ = [M[c].dropna() for c, _ in layers]
    boxes(ax, data_, fmt="{:.1f}%", flags=rflags(data_))
    ax.plot(range(1, 5), [np.median(x) for x in data_], color=INK, linewidth=0.9, linestyle="--", zorder=4,
            label="Median of HCR-listed researchers")
    ax.set_xticks(range(1, 5), [l for _, l in layers])
    ax.set_ylabel("Share that comes from surveys (%)")
    ax.set_ylim(0, 100)
    h_, l_ = ax.get_legend_handles_labels()
    ax.legend(handles=h_ + [r_handle()], loc="upper left")
    ax.tick_params(axis="x", length=0)
    style(ax)
    save(fig, "fig_layers")
    for c, l in layers:
        note(f"Median survey share of {l.replace(chr(10), ' ')} (%)", M[c].median())
        note(f"Mean survey share of {l.replace(chr(10), ' ')} (%)", M[c].mean())
    note("HCR-listed researchers with surveys >= 50% of top-1% papers", int((M["share_top1"] >= 50).sum()))
    note("HCR-listed researchers with surveys >= 25% of top-1% papers", int((M["share_top1"] >= 25).sum()))
    note("HCR-listed researchers with surveys >= 50% of citations", int((M["share_cites"] >= 50).sum()))


    # ============================================================
    # FIG: top-k curve
    # ============================================================

    K = np.arange(1, TOPK_MAX + 1)
    topk = np.vstack(M["topk"].values)
    med, q1, q3 = np.nanmean(topk, 0), np.nanpercentile(topk, 25, 0), np.nanpercentile(topk, 75, 0)
    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    ax.fill_between(K, q1, q3, color=SERIES, alpha=0.18, linewidth=0, label="Interquartile range")
    ax.plot(K, med, color=SERIES, linewidth=2, label="Mean over HCR-listed researchers")
    base = M["share_papers"].median()
    ax.axhline(base, color=INK_2, linewidth=0.8, linestyle="--")
    ax.text(TOPK_MAX, base + 1.5, f"surveys among all papers ({base:.0f}%)", ha="right", va="bottom",
            fontsize=6.5, color=INK_2)
    ax.set_xlabel("k most-cited papers of the researcher")
    ax.set_ylabel("Share that are surveys (%)")
    ax.set_xlim(1, TOPK_MAX)
    ax.set_ylim(0, 100)
    ax.legend(loc="upper right", fontsize=7)
    style(ax, "both")
    save(fig, "fig_topk")
    for k in (1, 5, 10, 20, 50):
        note(f"Mean survey share of top-{k} papers (%)", med[k - 1])
        note(f"Median survey share of top-{k} papers (%)", np.nanmedian(topk, 0)[k - 1])
    note("HCR-listed researchers whose single most-cited paper is a survey", int(np.nansum(topk[:, 0] == 100)))


    # ============================================================
    # FIG: citations per survey vs per research paper
    # ============================================================

    y = M.dropna(subset=["cites_per_survey", "cites_per_research"])
    fig, ax = plt.subplots(figsize=(3.5, 2.7), constrained_layout=True)
    lo, hi = 5, max(y["cites_per_survey"].max(), y["cites_per_research"].max()) * 1.5
    ax.plot([lo, hi], [lo, hi], color=INK_2, linewidth=0.8, linestyle="--", zorder=1)
    ax.scatter(y["cites_per_research"], y["cites_per_survey"], s=14, color=SERIES, alpha=0.75,
               linewidths=0.4, edgecolors="white", zorder=3)
    ratio = (y["cites_per_survey"] / y["cites_per_research"])
    ax.text(lo * 1.2, hi / 1.6, f"surveys cited {ratio.median():.1f}x more per paper (median)\n"
            f"{int((ratio > 1).sum())} of {len(y)} listed researchers above the line", fontsize=7, color=INK, va="top")
    ax.text(hi / 6, hi / 6 / 1.6, "equal citations\nper paper", fontsize=6.5, color=INK_2, ha="left", va="top")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(lo, hi)
    ax.set_ylim(lo, hi)
    ax.set_xlabel("Mean citations per research paper (log)")
    ax.set_ylabel("Mean citations per survey (log)")
    style(ax, "both")
    save(fig, "fig_yield")
    note("Median citations-per-paper ratio, survey / research", ratio.median())
    note("IQR citations-per-paper ratio", f"{ratio.quantile(.25):.1f}-{ratio.quantile(.75):.1f}")
    note("HCR-listed researchers whose surveys are cited more per paper than their research", int((ratio > 1).sum()))
    note("Median authors per survey / per research paper (median over listed researchers)",
         f"{M['authors_survey'].median():.1f} / {M['authors_research'].median():.1f}")


    # ============================================================
    # FIG: ranking and qualification threshold
    # ============================================================

    rb = M["top1"].rank(ascending=False, method="min")
    ra = M["top1_x"].rank(ascending=False, method="min")
    tau = kendalltau(M["top1"], M["top1_x"]).statistic
    moved = (rb - ra).abs()
    hb = M["h"].rank(ascending=False, method="min")
    ha = M["h_x"].rank(ascending=False, method="min")
    tau_h = kendalltau(M["h"], M["h_x"]).statistic

    fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.2 if SPRINGER else 2.7), constrained_layout=True)
    ax = axes[0]
    ax.plot([1, n], [1, n], color=INK_2, linewidth=0.8, linestyle="--", zorder=1)
    rf = M["key"].isin(R_NAMES).values
    ax.scatter(rb[~rf], ra[~rf], s=12, color=SERIES, alpha=0.8, linewidths=0.4, edgecolors="white", zorder=3,
               label="HCR-listed researcher")
    flag(ax, rb[rf], ra[rf])
    ax.legend(**({"loc": "upper left", "bbox_to_anchor": (0.0, 0.80), "frameon": True, "facecolor": "white", "edgecolor": "none", "framealpha": 0.9} if SPRINGER else {"loc": "lower right"}))
    ax.invert_xaxis()
    ax.invert_yaxis()
    ax.set_xlabel("Rank by OpenAlex top-1% papers, with surveys")
    ax.set_ylabel("Rank without surveys")
    ax.text(n * 0.97, 3, f"Kendall's $\\tau$ = {tau:.2f}\n{int((moved >= 10).sum())} of {n} researchers move\n10 or more places" if SPRINGER else
            f"Kendall's $\\tau$ = {tau:.2f}\n{int((moved >= 10).sum())} of {n} researchers move 10 or more places",
            fontsize=7, color=INK, va="top", ha="left", bbox=dict(facecolor="white", edgecolor="none", alpha=0.9, pad=1.5) if SPRINGER else None)
    panel(ax, "a", "Rank by top-1% papers, with and without surveys")
    style(ax, "both")

    ax = axes[1]
    T = np.arange(0, int(M["top1"].quantile(0.75)) + 1)
    below_b = [(M["top1"] < t).sum() for t in T]
    below_a = [(M["top1_x"] < t).sum() for t in T]
    ax.step(T, below_b, where="post", color=SERIES, linewidth=LW, linestyle="--", label="With surveys")
    ax.step(T, below_a, where="post", color=SERIES, linewidth=LW, label="Without surveys")
    ax.fill_between(T, below_b, below_a, step="post", color=SERIES, alpha=0.12, linewidth=0)
    floor = int(M["top1"].min())
    ax.axvline(floor, color=INK_2, linewidth=0.8, linestyle=":")
    ax.text(floor + 0.7, n * 0.95, f"lowest count\non the list ({floor})", fontsize=6.5, color=INK_2, va="top")
    ax.set_xlabel("Hypothetical threshold:\nOpenAlex top-1% papers needed" if SPRINGER else "Hypothetical threshold: OpenAlex top-1% papers needed")
    ax.set_ylabel("HCR-listed researchers below threshold")
    ax.set_ylim(0, n)
    ax.legend(loc="lower right")
    panel(ax, "b", "Below a hypothetical threshold" if SPRINGER else "Researchers below a hypothetical threshold")
    style(ax, "both")
    save(fig, "fig_rank_threshold")
    note("Kendall tau, rank by top-1% papers before vs after", tau)
    note("Kendall tau, rank by h-index before vs after", tau_h)
    note("HCR-listed researchers moving >= 10 places (top-1% rank)", int((moved >= 10).sum()))
    note("Median rank change (top-1% rank)", moved.median())
    note("Lowest listed researcher top-1% count (OpenAlex)", floor)
    for t in (floor, int(M["top1"].quantile(0.1)), int(M["top1"].quantile(0.25)), int(M["top1"].median())):
        note(f"HCR-listed researchers below {t} top-1% papers: before / after",
             f"{int((M['top1'] < t).sum())} / {int((M['top1_x'] < t).sum())}")
    top10_b = set(M.loc[rb <= 10, "key"])
    top10_a = set(M.loc[ra <= 10, "key"])
    note("Overlap of top-10 by top-1% papers, before vs after", len(top10_b & top10_a))


    # ============================================================
    # FIG: HCR-listed researchers vs matched non-listed peers
    # ============================================================

    if have_controls and len(pairs) >= 10:
        fig, axes = plt.subplots(1, 2, figsize=(WIDE, 3.25 if SPRINGER else 2.7), constrained_layout=True)
        ax = axes[0]
        fm = pairs["key"].isin(R_NAMES).values
        boxes(ax, [pairs["share_cites_m"]], positions=[1], fmt="{:.1f}%", flags=[fm], color=SERIES)
        boxes(ax, [pairs["share_cites_c"]], positions=[2], fmt="{:.1f}%", color=SERIES_2)
        ax.set_xlim(0.4, 2.6)
        ax.legend(handles=[r_handle()], loc="upper left")
        ax.set_xticks([1, 2], [f"HCR-listed with peers\n(n = {len(pairs)})", f"Their peers, mean per\nresearcher ({len(C)} peers)"])
        ax.set_ylabel("Survey share of citations (%)")
        ax.set_ylim(0, 100)
        ax.tick_params(axis="x", length=0)
        p_sc = wilcoxon(pairs["share_cites_m"], pairs["share_cites_c"]).pvalue
        panel(ax, "a", "Survey share of citations")
        style(ax)

        ax = axes[1]
        for x0, col, b_, a_ in ((1, SERIES, "top1_m", "top1_x_m"), (2.5, SERIES_2, "top1_c", "top1_x_c")):
            ax.boxplot([pairs[b_]], positions=[x0 - 0.25], widths=0.4, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor="white", edgecolor=col, linewidth=1, linestyle="--"),
                       medianprops=dict(color=col, linewidth=1.5), whiskerprops=dict(color=col, linewidth=1, linestyle="--"),
                       capprops=dict(color=col, linewidth=1))
            ax.boxplot([pairs[a_]], positions=[x0 + 0.25], widths=0.4, showfliers=False, patch_artist=True,
                       boxprops=dict(facecolor=col + "55", edgecolor=col, linewidth=1), medianprops=dict(color=col, linewidth=1.5),
                       whiskerprops=dict(color=col, linewidth=1), capprops=dict(color=col, linewidth=1))
            mb, ma = pairs[b_].median(), pairs[a_].median()
            ax.annotate("", xy=(x0 + 0.25, ma), xytext=(x0 - 0.25, mb),
                        arrowprops=dict(arrowstyle="-|>", color=INK, lw=0.9, shrinkA=0, shrinkB=0), zorder=8)
            ax.text(x0 - 0.47, mb, f"{mb:.0f}", ha="right", va="center", fontsize=6.5, color=INK)
            ax.text(x0 + 0.47, ma, f"{ma:.0f}", ha="left", va="center", fontsize=6.5, color=INK)
        ax.set_xticks([1, 2.5], [f"HCR-listed with peers\n(n = {len(pairs)})", f"Their peers, mean per\nresearcher ({len(C)} peers)"])
        ax.set_ylabel("OpenAlex top-1% papers, 2014–2024")
        ax.set_xlim(0.3, 3.2)
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="x", length=0)
        from matplotlib.patches import Patch
        ax.legend(handles=[Patch(facecolor="white", edgecolor=INK_2, linestyle="--", label="With surveys"),
                           Patch(facecolor=INK_2 + "55", edgecolor=INK_2, label="Without surveys")],
                  loc="upper right")
        panel(ax, "b", "OpenAlex top-1% papers, with and without surveys")
        style(ax)
        save(fig, "fig_controls")

        d_before = pairs["top1_m"] - pairs["top1_c"]
        d_after = pairs["top1_x_m"] - pairs["top1_x_c"]
        note("Matched pairs", len(pairs))
        note("Median survey share of citations, listed researchers / non-listed (%)",
             f"{pairs['share_cites_m'].median():.1f} / {pairs['share_cites_c'].median():.1f}")
        note("Paired Wilcoxon p, survey share of citations", p_sc)
        note("Median survey share of papers, listed researchers / non-listed (%)",
             f"{pairs['share_papers_m'].median():.1f} / {pairs['share_papers_c'].median():.1f}")
        note("Paired Wilcoxon p, survey share of papers",
             wilcoxon(pairs["share_papers_m"], pairs["share_papers_c"]).pvalue)
        note("Median survey share of top-1% papers, listed researchers / non-listed (%)",
             f"{pairs['share_top1_m'].median():.1f} / {pairs['share_top1_c'].median():.1f}")
        note("Median top-1% papers, listed researchers / non-listed", f"{pairs['top1_m'].median():.0f} / {pairs['top1_c'].median():.0f}")
        note("Median top-1% papers without surveys, listed researchers / non-listed",
             f"{pairs['top1_x_m'].median():.0f} / {pairs['top1_x_c'].median():.0f}")
        note("Mean listed researcher advantage in top-1% papers, all / surveys excluded",
             f"{d_before.mean():.1f} / {d_after.mean():.1f}")
        note("Share of the listed researcher advantage in top-1% papers accounted for by surveys (%)",
             100 * (1 - d_after.mean() / d_before.mean()) if d_before.mean() else np.nan)
        note("Paired Wilcoxon p, top-1% papers (listed researchers vs non-listed), all papers",
             wilcoxon(pairs["top1_m"], pairs["top1_c"]).pvalue)
        note("Paired Wilcoxon p, top-1% papers, surveys excluded",
             wilcoxon(pairs["top1_x_m"], pairs["top1_x_c"]).pvalue)
        note("Pairs where the listed researcher has more top-1% papers: all / surveys excluded",
             f"{int((d_before > 0).sum())} / {int((d_after > 0).sum())}")
        note("HCR-listed researchers with higher survey share of citations than their match", int((pairs["share_cites_m"] > pairs["share_cites_c"]).sum()))
        note("Median survey share of h-core, listed researchers / non-listed (%)",
             f"{pairs['share_hcore_m'].median():.1f} / {pairs['share_hcore_c'].median():.1f}")
        note("Paired Wilcoxon p, survey share of top-1% papers",
             wilcoxon(pairs["share_top1_m"].fillna(0), pairs["share_top1_c"].fillna(0)).pvalue)
        note("Median citations-per-paper ratio, listed researchers / non-listed",
             f"{(pairs['cites_per_survey_m'] / pairs['cites_per_research_m']).median():.1f} / "
             f"{(pairs['cites_per_survey_c'] / pairs['cites_per_research_c']).median():.1f}")
        note("Median citations, listed researchers / non-listed (OpenAlex)",
             f"{pairs['cites_m'].median():.0f} / {pairs['cites_c'].median():.0f}")


    # ============================================================
    # FIG: trend by publication year (unique works across HCR-listed researchers)
    # ============================================================

    W = M_works.drop_duplicates("id")
    W = W[W["year"].isin(TREND_YEARS)]
    W = W.assign(survey=W["Category"] == "survey", top1=W["top1"].fillna(False).astype(bool))
    by = W.groupby("year").agg(papers=("survey", "size"), surveys=("survey", "sum"))
    by1 = W[W["top1"]].groupby("year").agg(top1=("survey", "size"), top1_surveys=("survey", "sum"))
    by = by.join(by1).fillna(0)
    by["share_all"] = 100 * by["surveys"] / by["papers"]
    by["share_top1"] = 100 * by["top1_surveys"] / by["top1"].replace(0, np.nan)

    fig, ax = plt.subplots(figsize=(3.5, 2.4), constrained_layout=True)
    ax.plot(by.index, by["share_top1"], color=SERIES_2, linewidth=2, marker="o", markersize=3.5)
    ax.plot(by.index, by["share_all"], color=SERIES, linewidth=2, marker="o", markersize=3.5)
    ax.text(by.index[-1] + 0.3, by["share_top1"].iloc[-1], "top-1% papers", fontsize=7, color=INK, va="center")
    ax.text(by.index[-1] + 0.3, by["share_all"].iloc[-1], "all papers", fontsize=7, color=INK, va="center")
    ax.set_xlim(TREND_YEARS[0] - 0.5, TREND_YEARS[-1] + 3.2)
    ax.set_ylim(0, max(30, by["share_top1"].max() * 1.25))
    ax.set_xticks(range(TREND_YEARS[0], TREND_YEARS[-1] + 1, 2))
    ax.set_xlabel("Publication year")
    ax.set_ylabel("Share that are surveys (%)")
    ax.plot([], [], color=SERIES_2, linewidth=2, label="Top-1% papers")
    ax.plot([], [], color=SERIES, linewidth=2, label="All papers")
    ax.legend(loc="upper left", fontsize=7)
    style(ax, "both")
    save(fig, "fig_trend")
    for yr in (2010, 2015, 2020, 2024):
        if yr in by.index:
            note(f"{yr}: survey share of listed researchers' papers / top-1% papers (%)",
                 f"{by.loc[yr, 'share_all']:.1f} / {by.loc[yr, 'share_top1']:.1f} (n top-1% = {int(by.loc[yr, 'top1'])})")
    note("Unique works across listed researchers (2010-2024)", len(W))


    # ============================================================
    # FIG: most affected records, relative changes only
    # ============================================================

    top = M.set_index("key").loc[R_ORDER].reset_index()           # R1-R8 in order of citation drop
    top["Label"] = [f"R{i}" for i in range(1, len(top) + 1)]
    # Stanford-Elsevier composite drop (analyses.py composite), by OpenAlex ID
    comp_path = os.path.join(STUDY, "composite", "composite_INTERNAL.csv")
    id_of = dict(zip(members_meta["key"], members_meta["OpenAlex ID"]))
    M["comp_drop"] = np.nan
    if os.path.exists(comp_path):
        comp = pd.read_csv(comp_path)
        comp = comp[comp.group == "listed"].drop_duplicates("id").set_index("id")["dc %"]
        M["comp_drop"] = -M["key"].map(id_of).map(comp)
        top["comp_drop"] = -top["key"].map(id_of).map(comp)

    if SPRINGER:            # (a) to (c) in the first row, (d), (e) and the legend in the second
        fig = plt.figure(figsize=(WIDE, 1.2 + 0.4 * len(top)), constrained_layout=True)
        gs = fig.add_gridspec(2, 3, width_ratios=[1.3, 1, 1.6])
        _a0 = fig.add_subplot(gs[0, 0])
        axes = [_a0] + [fig.add_subplot(gs[i, j], sharey=_a0) for i, j in ((0, 1), (0, 2), (1, 0), (1, 1))]
        _leg = fig.add_subplot(gs[1, 2]); _leg.axis("off")
    else:
        fig, axes = plt.subplots(1, 5, figsize=(7.16, 2.9), constrained_layout=True, sharey=True,
                                 gridspec_kw={"width_ratios": [1.3, 1, 1.6, 1, 0.9]})
    ypos = np.arange(len(top))[::-1]

    # (a) survey share of papers -> of citations (dumbbell)
    ax = axes[0]
    for yv, a_, b_ in zip(ypos, top["share_papers"], top["share_cites"]):
        ax.plot([a_, b_], [yv, yv], color=LISTED, linewidth=1.2, alpha=0.6, zorder=2)
    ax.scatter(top["share_papers"], ypos, s=18, facecolors="white", edgecolors=LISTED, linewidths=1.1, zorder=3)
    ax.scatter(top["share_cites"], ypos, **{**R_MARK, "s": 22})
    for yv, v in zip(ypos, top["share_cites"]):
        ax.text(v + 5, yv, f"{v:.0f}%", va="center", fontsize=6, color=INK)
    for col, ls in (("share_papers", ":"), ("share_cites", "--")):
        ax.axvline(M[col].median(), color=INK_2, linewidth=0.7, linestyle=ls, zorder=1)
    ax.set_xlim(0, 100)
    ax.set_xticks([0, 50, 100])
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
    panel(ax, "a", "Survey share")
    style(ax, "x")
    ax.tick_params(axis="y", length=0)

    panels = [("cites_drop", "b", "Citation drop"), ("h_drop", "c", "h-index with and without surveys"),
              ("top1_drop", "d", "Top-1% paper drop"), ("comp_drop", "e", "Composite drop")]
    from matplotlib.patches import Patch  # noqa: E402
    for ax, (col, letter, title) in zip(axes[1:], panels):
        if col == "h_drop":             # absolute h-index: dashed outline with surveys, solid bar without, both values and the drop
            hb, ha = top["h"].values, top["h_x"].values
            xmax = hb.max() * 1.95
            ax.barh(ypos, hb, height=0.62, facecolor="white", edgecolor=LISTED, linestyle="--", linewidth=0.8, zorder=2)
            ax.barh(ypos, ha, height=0.62, color=LISTED, zorder=3)
            for yv, b_, a_, d_ in zip(ypos, hb, ha, top["h_drop"]):
                ax.text(b_ + xmax * 0.025, yv, f"{b_:.0f} $\\rightarrow$ {a_:.0f} ($-${d_:.0f}%)", va="center", fontsize=6, color=INK)
            ax.set_xlim(0, xmax)
            ax.xaxis.set_major_locator(plt.MultipleLocator(50))
            panel(ax, letter, title)
            style(ax, "x")
            ax.tick_params(axis="y", length=0)
            continue
        vals = top[col].values
        xmax = 100 if col != "comp_drop" else max(20, np.nanmax(vals) * 1.35)
        ax.barh(ypos, vals, color=LISTED, height=0.62, zorder=2)
        for yv, v in zip(ypos, vals):
            if not np.isnan(v):
                ax.text(v + xmax * 0.02, yv, f"{v:.0f}%", va="center", fontsize=6, color=INK)
        ax.axvline(M[col].median(), color=INK_2, linewidth=0.8, linestyle="--", zorder=3)
        ax.set_xlim(0, xmax)
        ax.set_xticks([0, 50, 100] if col != "comp_drop" else [0, 10, 20])
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}%"))
        panel(ax, letter, title)
        style(ax, "x")
        ax.tick_params(axis="y", length=0)
    axes[0].set_yticks(ypos, top["Label"])
    if SPRINGER:
        for _ax in (axes[1], axes[2], axes[4]):
            _ax.tick_params(labelleft=False)
    (_leg.legend if SPRINGER else fig.legend)(handles=[plt.Line2D([], [], linestyle="none", marker="o", markersize=4, markerfacecolor="white",
                                   markeredgecolor=LISTED, markeredgewidth=1.1, label="Survey share of papers"),
                        plt.Line2D([], [], linestyle="none", marker="D", markersize=4, markerfacecolor=LISTED,
                                   markeredgecolor=INK, markeredgewidth=0.6, label="Survey share of citations"),
                        plt.Line2D([], [], color=INK_2, linewidth=0.8, linestyle=":", label="Median of all HCR-listed, papers"),
                        plt.Line2D([], [], color=INK_2, linewidth=0.8, linestyle="--", label="Median of all HCR-listed, citations or drop"),
                        Patch(facecolor="white", edgecolor=LISTED, linestyle="--", linewidth=0.8, label="h-index with surveys"),
                        Patch(facecolor=LISTED, label="h-index without surveys")],
               **({"loc": "center left", "ncol": 1} if SPRINGER else {"loc": "outside lower center", "ncol": 3}))
    save(fig, "fig_top10_relative")
    top[["Label", "key"]].to_csv(os.path.join(STUDY, "v2_top10_key_INTERNAL.csv"), index=False)
    note("R1-R8 median survey share of top-1% papers (%)", top["share_top1"].median())
    note("R1-R8 range of h-index drop (%)", f"{top['h_drop'].min():.0f}-{top['h_drop'].max():.0f}")
    note("R1-R8 range of citation drop (%)", f"{top['cites_drop'].min():.1f}-{top['cites_drop'].max():.1f}")
    note("R1-R8 range of survey share of papers (%)", f"{top['share_papers'].min():.0f}-{top['share_papers'].max():.0f}")
    note("R1-R8 range of composite drop (%)", f"{top['comp_drop'].min():.0f}-{top['comp_drop'].max():.0f}")
    note("R1-R8 minimum top-1% papers without surveys", int(top['top1_x'].min()))
    note("R1-R8 range of top-1% paper drop (%)", f"{top['top1_drop'].min():.0f}-{top['top1_drop'].max():.0f}")


    # ============================================================
    # TABLE: what each detection method would remove (HCR-listed researchers)
    # ============================================================

    mt = pd.DataFrame([{
        "Method": m,
        "Papers (%)": M[f"{m}|papers"].mean(),
        "Citations (%)": M[f"{m}|cites"].mean(),
        "h-index drop (%)": M[f"{m}|h"].mean(),
    } for m in METHOD_COLS])


    # ============================================================
    # WRITE STATS
    # ============================================================

    with open(os.path.join(STUDY, "v2_stats.md"), "w") as fh:
        fh.write(f"# Numbers for the paper (n = {n} researchers listed in the HCR)\n\n")
        for label, value in stats:
            v = f"{value:.4g}" if isinstance(value, (float, np.floating)) else str(value)
            fh.write(f"- {label}: {v}\n")
        fh.write("\n## Methods on HCR-listed researchers (means over researchers)\n\n")
        fh.write(mt.round(2).to_markdown(index=False) + "\n")
    print(open(os.path.join(STUDY, "v2_stats.md")).read())


# =====================================================================================================================
# Figure: HCR-listed researchers, ACM and IEEE Fellows
# =====================================================================================================================

def fellows_figure():
    """Figure for RQ4: survey dependence of academic HCR-listed researchers, peer-elected ACM and IEEE Fellows, and the
    academic Stanford-Elsevier reference sample (comparison_groups.py fellows report). Three panels: survey share of papers, survey
    share of citations (the citation drop when surveys are excluded), and the survey citation dividend.
    In every group, points beyond the group's own Tukey fence for the survey share of citations have a dark outline, and
    the researcher with the highest survey share of citations is enlarged in all panels, with only the value shown.
    R1-R8 keep the shared diamond marker. No researcher is named.
    Output: hcr_study/figures/fig_fellows.{pdf,png}"""
    GROUPS = [("HCR-listed 2025", "HCR-\nlisted", LISTED), ("ACM Fellow 2025-26", "ACM\nFellows", ACM),
              ("IEEE Fellow 2026", "IEEE\nFellows", IEEE), ("Reference sample", "Random\nsample", REFERENCE)]
    P, C, D = "Survey % of papers", "Survey % of citations", "Dividend D"   # all records, as everywhere in the paper

    d = pd.read_csv(os.path.join(HERE, "hcr_study", "fellows2026", "per_person_scored_INTERNAL.csv"))
    rids = r_sets()["ids"]
    rng = np.random.default_rng(7)
    d["x_jit"] = rng.uniform(-0.17, 0.17, len(d))
    d["is_R"] = (d.group == "HCR-listed 2025") & d["OpenAlex ID"].isin(rids)
    d["fence_out"] = False; d["top"] = False
    for g, _, _ in GROUPS:
        m = d.group == g
        q1, q3 = d.loc[m, C].quantile([0.25, 0.75])
        d.loc[m, "fence_out"] = d.loc[m, C] > q3 + 1.5 * (q3 - q1)
        d.loc[d.loc[m, C].idxmax(), "top"] = True

    if SPRINGER:            # 2 x 2: three panels and the legend
        fig, _ax4 = plt.subplots(2, 2, figsize=(WIDE, 4.6), gridspec_kw={"wspace": 0.42, "hspace": 0.42})
        axes = [_ax4[0, 0], _ax4[0, 1], _ax4[1, 0]]; _leg = _ax4[1, 1]; _leg.axis("off")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(7.16, 2.35), gridspec_kw={"wspace": 0.34})
    specs = [(P, "Survey share of papers (%)", "Survey share of papers", False),
             (C, "Survey share of citations (%)\n= citation drop without surveys", "Survey share of citations", False),
             (D, "Citations per survey /\ncitations per research paper", "Survey citation dividend", True)]
    for k, (ax, (col, ylab, title, logy)) in enumerate(zip(axes, specs)):
        style(ax)
        for i, (g, lab, colour) in enumerate(GROUPS):
            e = d[(d.group == g) & d[col].notna()]
            y = e[col].clip(lower=0.1) if logy else e[col]
            ax.boxplot([y], positions=[i], widths=0.5, showfliers=False, whis=1.5, patch_artist=True, zorder=2,
                       boxprops=dict(facecolor="none", edgecolor=INK_2, linewidth=0.7), whiskerprops=dict(color=INK_2, linewidth=0.7),
                       capprops=dict(color=INK_2, linewidth=0.7), medianprops=dict(color=INK, linewidth=1.3))
            plain = e[~e.is_R & ~e.top]
            ax.scatter(i + plain.x_jit, y[plain.index], color=colour, zorder=3, **POINT)
            out = plain[plain.fence_out]
            ax.scatter(i + out.x_jit, y[out.index], s=9, facecolor=colour, edgecolor=INK, linewidths=0.6, zorder=4)
            rr = e[e.is_R & ~e.top]
            ax.scatter(i + rr.x_jit, y[rr.index], **{**R_MARK, "s": 13})
            t = e[e.top]
            ax.scatter(i + t.x_jit, y[t.index], s=46, marker="D" if t.is_R.any() else "o", facecolor=colour,
                       edgecolor=INK, linewidths=1.1, zorder=7)
            if col == C and len(t):
                ax.annotate(f"{t[C].iloc[0]:.0f}%", (i + t.x_jit.iloc[0], t[C].iloc[0]), xytext=(7, -1), textcoords="offset points",
                            fontsize=6.5, color=INK, va="center", zorder=8)
        ax.set_xticks(range(len(GROUPS)), [lab for _, lab, _ in GROUPS], rotation=0)
        ax.tick_params(axis="x", length=0)
        for tick, (_, _, colour) in zip(ax.get_xticklabels(), GROUPS):
            tick.set_color(INK)
        ax.set_xlim(-0.6, len(GROUPS) - 0.4)
        ax.set_ylabel(ylab)
        if logy:
            ax.set_yscale("log"); ax.set_yticks([0.1, 0.3, 1, 3, 10, 30], ["0.1", "0.3", "1", "3", "10", "30"])
            ax.axhline(1, color=INK_2, linewidth=0.6, linestyle=":", zorder=1)
        else:
            ax.set_ylim(0, None)
        panel(ax, "abc"[k], title)
        for lab in ax.get_xticklabels():
            lab.set_fontsize(6.5)

    handles = [plt.Line2D([], [], linestyle="none", marker="o", markersize=3.5, markerfacecolor=REFERENCE, markeredgecolor=INK,
                          markeredgewidth=0.6, label="Beyond the group's Tukey fence for the survey share of citations"),
               plt.Line2D([], [], linestyle="none", marker="o", markersize=6, markerfacecolor=REFERENCE, markeredgecolor=INK,
                          markeredgewidth=1.1, label="Highest survey share of citations in the group"),
               r_handle()]
    if SPRINGER:
        import textwrap
        for _t in _leg.legend(handles=handles, loc="center left", ncol=1, fontsize=6.5).get_texts():
            _t.set_text(textwrap.fill(_t.get_text(), 34))
        fig.subplots_adjust(bottom=0.08, top=0.94, left=0.1, right=0.99)
    else:
        fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, 0.0), fontsize=6.5)
        fig.subplots_adjust(bottom=0.25, top=0.89, left=0.07, right=0.99)
    save(fig, "fig_fellows")
    s = d.groupby("group").agg(n=(C, "size"), outliers=("fence_out", "sum"), top=(C, "max"))
    print(s.to_string())


# =====================================================================================================================
# Figure: survey citation dividend
# =====================================================================================================================

def dividend_figure():
    """Figure for the Discussion section (figures.py dividend) on the incentive to write surveys (analyses.py dividend-window results).
    (a) Within-researcher survey dividend by publication cohort, citations in the first three years, medians with 95%
        bootstrap intervals, R1-R8 as diamonds. (b) Papers of 2012-2014 followed for eleven years: cumulative citations of
        surveys / research papers of the same researcher. No researcher is named.
    Output: hcr_study/figures/fig_dividend.{pdf,png}"""
    D = os.path.join(HERE, "hcr_study", "dividend_window")
    GROUPS = [("HCR-listed", "HCR-listed", LISTED), ("Reference", "Random sample", REFERENCE),
              ("ACM Fellows", "ACM Fellows", ACM), ("IEEE Fellows", "IEEE Fellows", IEEE)]
    LINE = dict(linewidth=LW, marker="o", markersize=MS)
    A = pd.read_csv(os.path.join(D, "within.csv")); A = A[A.measure == "3-year window"]
    V = pd.read_csv(os.path.join(D, "speed.csv"))
    COH = ["2012-14", "2015-17", "2018-20", "2021-23"]
    tick = lambda c: c.replace("-", "–")

    if SPRINGER:            # two panels side by side, the legend below
        fig, axes = plt.subplots(1, 2, figsize=(WIDE, 2.9), gridspec_kw={"wspace": 0.45})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(7.16, 2.45), gridspec_kw={"wspace": 0.3})
    for ax in axes:
        style(ax); ax.axhline(1, color=INK_2, linewidth=0.6, linestyle=":", zorder=1)

    ax = axes[0]
    for k, (g, _, col) in enumerate(GROUPS):
        e = A[A.group == g].set_index("cohort").reindex(COH); x = np.arange(len(COH)) + (k - 1.5) * 0.06
        ax.errorbar(x, e["median D"], yerr=[e["median D"] - e["CI low"], e["CI high"] - e["median D"]], color=col, capsize=0,
                    elinewidth=0.7, zorder=3, **LINE)
    r = A[A.group == "R1-R8"].set_index("cohort").reindex(COH)
    ax.plot(np.arange(len(COH)) + 0.12, r["median D"], color=LISTED, linewidth=0.7, linestyle="--", zorder=4)
    ax.scatter(np.arange(len(COH)) + 0.12, r["median D"], **{**R_MARK, "s": 16})
    ax.set_xticks(range(len(COH)), [tick(c) for c in COH]); ax.set_ylim(0, None)
    ax.set_xlabel("Publication years"); ax.set_ylabel("Citations per survey /\ncitations per research paper")
    panel(ax, "a", "Within researcher, first 3 years")

    ax = axes[1]
    for g, _, col in GROUPS:
        e = V[V.group == g].sort_values("years after publication")
        ax.plot(e["years after publication"], e["median ratio"], color=col, **LINE)
    ax.set_xticks(range(0, 11, 2)); ax.set_ylim(0, None)
    ax.set_xlabel("Years after publication"); ax.set_ylabel("Cumulative citations,\nsurveys / research papers")
    panel(ax, "b", "Timing, papers of 2012–2014")

    rh = r_handle(); rh.set_label("R1–R8")
    handles = [plt.Line2D([], [], color=col, label=lab, **LINE) for _, lab, col in GROUPS] + [rh]
    fig.legend(handles=handles, loc="lower center", ncol=5, bbox_to_anchor=(0.5, 0.0), fontsize=6.5, columnspacing=1.2, handlelength=2.2)
    if SPRINGER:
        fig.subplots_adjust(bottom=0.26, top=0.9, left=0.12, right=0.99)
    else:
        fig.subplots_adjust(bottom=0.3, top=0.9, left=0.075, right=0.99)
    save(fig, "fig_dividend")
    print("saved fig_dividend")


STEPS = {"hcr": hcr_figures, "fellows": fellows_figure, "dividend": dividend_figure}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv.pop(1)]()
