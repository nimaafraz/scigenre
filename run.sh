#!/usr/bin/env bash
# Reproduce the paper, from data collection to the analyses and figures: bash run.sh (stops at the first error).
#
# Layout: the Python files of this repository; third-party inputs in inputs/ and API keys in .env (README.md);
# the trained model in model/; every output in hcr_study/ (figures in hcr_study/figures/).
#
# Marks:  [OA] OpenAlex (OPENALEX_API_KEY recommended; the keyless daily budget is small)
#         [S2] Semantic Scholar (SEMANTIC_SCHOLAR_API_KEY recommended; keyless requests are throttled)
#         [EL] Scopus / Elsevier API (ELSEVIER_API_KEY required)
#         [SN] Springer Nature Meta API (SPRINGER_META_API_KEY required)
#         [in:...] reads the named folder of inputs/
#         [GPU] advisable on a GPU
#
# Most steps cache what they fetch and resume where they stopped, so a stage can be run again after an
# interruption. Steps that accept WORKER=k NWORKERS=n can be split over parallel processes (then run "merge").

set -euo pipefail
cd "$(dirname "$0")"
PY=${PYTHON:-python}

# Missing abstracts of the downloaded profiles: Semantic Scholar by DOI, then Scopus (DOI, exact title) and Springer
# Nature. Both steps are incremental and run before every classification. [S2] [EL] [SN]
fill_abstracts() {
    $PY ground_truth.py s2-abstracts
    $PY ground_truth.py publisher-abstracts
}

# ---------------------------------------------------------------------------------------------------------------
# 1. Ground truth: every work of 18 survey venues and 14 research venues, selected by their stated scope (venue =
#    label), their abstracts (Semantic Scholar, then Scopus, by DOI), and the table of usable papers
#    (hcr_study/venue_all/ground_truth.csv). [OA] [S2] [EL]
# ---------------------------------------------------------------------------------------------------------------
$PY ground_truth.py venues openalex
$PY ground_truth.py venues s2
$PY ground_truth.py venues scopus
$PY ground_truth.py venues build

# ---------------------------------------------------------------------------------------------------------------
# 2. Train SciGenre (SPECTER2, title + abstract) and install it as model/, or download the trained model into
#    model/ instead (README.md). "train.py final" writes the 80/10/10 split (hcr_study/venue_all/split.csv), which
#    profiles.py also reads. Then the leave-venue-out test (three more models) and the fine-tuned baselines. [GPU]
# ---------------------------------------------------------------------------------------------------------------
$PY train.py final --seed 42 --epochs 1
$PY train.py install
$PY train.py holdout                  # the three folds, then the summary (one fold: --fold k)
$PY train.py final --model allenai/scibert_scivocab_uncased --tag scibert
$PY train.py final --model google-bert/bert-base-uncased --tag bert
# Seeds 43-46 of the three encoders (they report the seed variation only)
for s in 43 44 45 46; do
    $PY train.py final --seed $s
    $PY train.py final --model allenai/scibert_scivocab_uncased --tag scibert_seed$s --seed $s
    $PY train.py final --model google-bert/bert-base-uncased --tag bert_seed$s --seed $s
done
# Zero-shot LLM baseline: needs vLLM 0.31 and two 48 GB GPUs, so it is left to run by hand:
#   TP=2 $PY train.py llm gpt-oss-120b

# ---------------------------------------------------------------------------------------------------------------
# 3. RQ1 on the held-out test papers, against keyword rules, database types, TF-IDF + SVM, Smyth & Cunningham's rule
#    (their code) and the zero-shot LLM (stage 2); replication of Smyth & Cunningham's published
#    labels. [S2] [OA] [in:smyth_cunningham]
# ---------------------------------------------------------------------------------------------------------------
$PY rq1_evaluation.py baselines fetch
$PY rq1_evaluation.py baselines evaluate
$PY rq1_evaluation.py label-checks
$PY rq1_evaluation.py sc-replication fetch
$PY rq1_evaluation.py sc-replication evaluate

# ---------------------------------------------------------------------------------------------------------------
# 4. Resolve and fetch the profiles of the 2025 HCR Computer Science list and match the list to the Stanford-Elsevier
#    career table. Then the HCR Computer Science lists 2014-2025, used to flag comparison researchers who were ever
#    listed. Then the classification of the HCR-listed researchers (2025 list, then the historical lists). It comes
#    before the comparison groups, which are drawn against its outputs. [OA] [S2] [EL] [SN] [in:hcr_lists] [in:stanford_elsevier]
# ---------------------------------------------------------------------------------------------------------------
$PY profiles.py fetch
$PY profiles.py els-screen
$PY profiles.py els-match
$PY profiles.py history build
$PY profiles.py history resolve        # or in parallel: WORKER=k NWORKERS=n ..., then: $PY profiles.py history merge
$PY profiles.py history els
$PY profiles.py history fetch
fill_abstracts
$PY profiles.py classify
$PY profiles.py history classify

# ---------------------------------------------------------------------------------------------------------------
# 5. Comparison groups, each classified by the same model after its abstract fill.
#    [OA] [S2] [EL] [SN] [in:hcr_lists] [in:stanford_elsevier] [in:fellows]
# ---------------------------------------------------------------------------------------------------------------
# 5a. One-to-one citation-matched peers (a sensitivity check; the random sample excludes them) and the random
#     sample.
$PY comparison_groups.py controls select
fill_abstracts
$PY comparison_groups.py controls classify
$PY comparison_groups.py reference select
fill_abstracts
$PY comparison_groups.py reference classify

# 5b. Exclusion of researchers on a Highly Cited Researchers list in any category: one-to-one peers on a 2025 list
#     are replaced (exclude, select again, classify the new matches) until none is left.
while true; do
    $PY comparison_groups.py any-list
    n=$($PY -c "import pandas as pd; d = pd.read_csv('hcr_study/hcr_any_list.csv'); print(int(((d.group == 'matched') & d['on a 2025 list'].astype(bool)).sum()))")
    [ "$n" -eq 0 ] && break
    $PY comparison_groups.py controls exclude
    $PY comparison_groups.py controls select
    fill_abstracts
    $PY comparison_groups.py controls classify --missing
done
# Random sample: drop researchers on a 2025 list and profiles whose OpenAlex topics lie mostly outside computer
# science and engineering.
$PY comparison_groups.py reference topics
$PY comparison_groups.py reference exclude

# 5c. Matched peers: up to five per HCR-listed researcher, the most similar on the Stanford-Elsevier career indicators.
$PY comparison_groups.py peers select
PROFILE_DIRS=peers/profiles $PY ground_truth.py s2-abstracts
PROFILE_DIRS=peers/profiles $PY ground_truth.py publisher-abstracts doi
PROFILE_DIRS=peers/profiles $PY ground_truth.py publisher-abstracts springer
$PY comparison_groups.py peers classify

# 5d. Boundary samples just inside the Stanford-Elsevier top-2% cutoff, in two subfields.
for SUB in "Networking & Telecommunications" "Artificial Intelligence & Image Processing"; do
    SUBFIELD_NAME="$SUB" $PY comparison_groups.py boundary select     # or in parallel: WORKER=k NWORKERS=n ...
    SUBFIELD_NAME="$SUB" $PY comparison_groups.py boundary merge
done
fill_abstracts
for SUB in "Networking & Telecommunications" "Artificial Intelligence & Image Processing"; do
    SUBFIELD_NAME="$SUB" $PY comparison_groups.py boundary classify
done

# 5e. Peer-elected groups: IEEE Fellows of the class of 2026 and ACM Fellows announced in 2025 and 2026 (needs
#     pdftotext from poppler; "fetch" fills the Fellows' abstracts itself).
for s in load match academic fetch classify report; do $PY comparison_groups.py fellows $s; done

# ---------------------------------------------------------------------------------------------------------------
# 6. Analyses (outputs under hcr_study/). "figures.py hcr" runs once here because it fixes R1-R8
#    (hcr_study/v2_top10_key_INTERNAL.csv), which later steps mark in their figures.
# ---------------------------------------------------------------------------------------------------------------
$PY comparison_groups.py peers analyze
$PY analyses.py survey-outliers
$PY analyses.py yearly-citations       # [OA]; or in parallel: WORKER=k NWORKERS=n ...
$PY figures.py hcr
$PY analyses.py outlier-model
$PY analyses.py outlier-checks
$PY analyses.py composite
for SUB in "Networking & Telecommunications" "Artificial Intelligence & Image Processing"; do
    SUBFIELD_NAME="$SUB" $PY comparison_groups.py boundary analyze
done
$PY analyses.py boundary-scenarios
$PY analyses.py primary-endpoints
$PY analyses.py s2-types hcr           # [S2]
$PY analyses.py s2-rule                # [in:smyth_cunningham]
$PY analyses.py dividend-window
$PY analyses.py keyword-rules
$PY analyses.py abstract-sources       # [S2] [EL]
$PY analyses.py paper-numbers

# ---------------------------------------------------------------------------------------------------------------
# 7. Figures (hcr_study/figures/). The analyses above already drew their own figures (fig_outlier_*, fig_model_*,
#    fig_composite); these complete the set.
# ---------------------------------------------------------------------------------------------------------------
$PY figures.py hcr                     # again, now with the composite drop (panel e of fig_top10_relative)
$PY figures.py fellows
$PY figures.py dividend

echo "Done. Outputs: hcr_study/, figures: hcr_study/figures/"
