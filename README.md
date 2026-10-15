# SciGenre: code for *The Survey Dividend*

Code for the paper

> Nighot, A., and Afraz, N. *The Survey Dividend: How Citation Metrics Reward Synthesis over Discovery.*
> Manuscript submitted to *Scientometrics*.

SciGenre is a SPECTER2 classifier that labels a paper as a survey or a research paper from its title and abstract.
The paper uses it to measure how much of the citation impact of researchers on Clarivate's Highly Cited Researchers
(HCR) list in Computer Science comes from surveys, and compares them with matched peers, a random sample of the
Stanford–Elsevier database of top-cited scientists, and IEEE and ACM Fellows. The trained model is on the Hugging Face
Hub (https://huggingface.co/nimaafraz/scigenre), with `predict.py` for classifying a CSV file of titles and abstracts.

## Files

| File | Content |
|---|---|
| `run.sh` | Every step of the paper, in order |
| `common.py` | Model input (title and abstract, cleaned), OpenAlex client, profile format, h-index, figure style |
| `ground_truth.py` | Stage 1: ground truth from 18 survey venues and 14 research venues, and missing abstracts |
| `train.py` | Stage 2: SciGenre, the fine-tuned baselines, the leave-venue-out test, the zero-shot LLM baseline |
| `rq1_evaluation.py` | Stage 3 (RQ1): SciGenre against the baselines, checks of the venue labels, Smyth and Cunningham's rule |
| `profiles.py` | Stage 4: the 2025 HCR list, OpenAlex profiles, Stanford–Elsevier records, classification of every profile |
| `comparison_groups.py` | Stage 5: matched peers, random sample, one-to-one peers, boundary samples, IEEE and ACM Fellows |
| `analyses.py` | Stage 6: the analyses behind the paper's numbers |
| `figures.py` | Stage 7: the paper's figures |

Each stage file runs its steps as `python <file> <step> [arguments]`. Run a file without a step to list its steps.
Every output is written to `hcr_study/` (figures to `hcr_study/figures/`).

## Setup

1. Python 3.11 or later: `pip install -r requirements.txt`. The Fellows step also needs `pdftotext` (poppler).
2. Copy `.env.example` to `.env` and fill in the API keys of OpenAlex, Semantic Scholar, Elsevier and Springer
   Nature. The code never prints them.
3. The third-party inputs in `inputs/` (next section).
4. The model: either train it (`train.py final`, then `train.py install`) or download it from the Hugging Face Hub
   into `model/` (`hf download nimaafraz/scigenre --local-dir model`). `profiles.py` also reads the ground truth and its
   split, so with a downloaded model run stage 1 and then `python train.py final --split-only` (seed 2026).

## Third-party inputs

None of these files is redistributed here. Download them from their publishers and place them at these paths under
`inputs/` (names are case sensitive).

| Path | Content | Source |
|---|---|---|
| `hcr_lists/2014_HCR.xlsx` ... `2025_HCR.xlsx` | Clarivate Highly Cited Researchers lists, one workbook per year (12 files) | Clarivate, [past lists](https://clarivate.com/highly-cited-researchers/past-lists/) (2025 release of the historical lists) |
| `stanford_elsevier/els.csv` | Career table, August 2025 update: the `Data` sheet of `Table_1_Authors_career_2024_pubs_since_1788_wopp_extracted_202508.xlsx`, saved as CSV UTF-8 | Elsevier Data Repository, doi:[10.17632/btchxktzyw.8](https://doi.org/10.17632/btchxktzyw.8) |
| `stanford_elsevier/Table_3_maxlog_career_2024_pubs_since_1788_wopp_extracted_202508.xlsx` | maxlog normalizers of the composite indicator | same release |
| `stanford_elsevier/Table_1_Authors_singleyr_2024_pubs_since_1788_wopp_extracted_202508.xlsx` | single-year 2024 table (optional) | same release |
| `fellows/IEEE-fellow-2026.pdf` | IEEE Fellow class of 2026, as published with the announcement | IEEE Fellow Program, ieee.org |
| `fellows/ACM_Fellows_2024_announcement.txt`, `fellows/ACM_Fellows_2025_announcement.txt` | text of ACM's announcements of January 2025 and January 2026 | ACM press releases, acm.org |
| `fellows/ACM Fellows.csv` | ACM Fellows listing with world region | ACM awards site, awards.acm.org |
| `smyth_cunningham/sc_labels_cs.csv` | Smyth and Cunningham's released labels, Computer Science rows | figshare, doi:[10.6084/m9.figshare.32077821](https://doi.org/10.6084/m9.figshare.32077821) |
| `smyth_cunningham/review_paper_rule.py` | Smyth and Cunningham's code of their review rule, unchanged | same figshare record |

Notes on the inputs:

* **HCR lists.** The code finds the columns by name (first name, last or family name, category, affiliations), so
  the headings may differ between years. The 2025 workbook needs `First Name`, `Last Name`, `Category`,
  `Primary Affiliation` and `Secondary Affiliations`. Its 119 rows in `Computer Science` are the list studied.
* **els.csv.** The code reads numbers with thousands separators and an optional byte-order mark, so any CSV export of
  the sheet works. The paper's numbers come from the Excel export, which stores values as displayed (four decimals for
  the composite indicator). An export at full precision may change results slightly near the top-2% cutoff. The
  single-year table only adds a descriptive column that no analysis uses.
* **Fellows.** `IEEE-fellow-2026.pdf` is read with `pdftotext -layout`. Each Fellow is a block of name, affiliation,
  location and a citation that starts with "for". The ACM announcement files hold one Fellow per block of three lines
  (name, affiliation, citation), blocks separated by a blank line, with `NA` for an unknown affiliation.
  `ACM Fellows.csv` has the columns `Name` ("Last, First"), `Award`, `Year` (class) and `Region`. Only the classes
  2024 and 2025 are used.
* **Smyth and Cunningham.** Smyth, B., and Cunningham, P. (2026), "Have we reached the beginning of the end for
  review papers?", *Quantitative Science Studies* 7, 1044–1061, doi:[10.1162/qss.a.488](https://doi.org/10.1162/qss.a.488).
  `sc_labels_cs.csv` holds the rows of their dataset with `primary_fos == "computer_science"`, with all columns:

  ```python
  import pandas as pd
  d = pd.read_csv("<their dataset>.csv", dtype={"corpusid": str})
  d[d.primary_fos == "computer_science"].to_csv("inputs/smyth_cunningham/sc_labels_cs.csv", index=False)
  ```

  `review_paper_rule.py` is the authors' code of the rule that labels their 17M data set, released with their
  dataset. The code loads it on first use and stops with a message if it is missing. With our Semantic Scholar
  records it reproduces their published labels for 99.95% of 26,589 re-fetched Computer Science papers
  (`rq1_evaluation.py sc-replication`).

## Running

`bash run.sh` runs every step in order and stops at the first error. Its comments mark the steps that call an API or
read an input folder. Most steps cache what they fetch and resume where they stopped, and the slow retrieval steps can
be split over parallel processes (`WORKER=k NWORKERS=n`, then `merge`). Training and scoring are advisable on a GPU.

## Reproducibility

Run on the data retrieved for the paper, stages 6 and 7 reproduce the reports from which the paper's numbers were
taken. A new run depends on the current state of OpenAlex and of the other APIs, so profiles retrieved today can
differ from those of October 2026.

## License

MIT (`LICENSE`). Anyone may use, copy, modify and redistribute the code, also commercially, provided the copyright
and license notice are kept.
