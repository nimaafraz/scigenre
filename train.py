"""Stage 2: training of SciGenre and of the fine-tuned baselines, the leave-venue-out test, the zero-shot LLM
baseline, and the scoring of the analyses' texts on a separate GPU machine (SE_COLLECT, see profiles.py).

    python train.py final [--seed 42] [--epochs 1] [--model M --tag T]   SciGenre (or a fine-tuned baseline)
    python train.py install                                              install the trained model as model/
    python train.py holdout --fold 1|2|3 (then --summary)                leave-venue-out test
    python train.py score                                                score the texts recorded with SE_COLLECT=1 (GPU)
    python train.py llm gpt-oss-120b                                     zero-shot LLM baseline (vLLM, GPU)
"""
import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import time
import warnings

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.utils.class_weight import compute_class_weight
from transformers import (
    AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding, EarlyStoppingCallback, Trainer,
    TrainingArguments,
)

from common import clean_text, MAX_LENGTH, model_text, strip_boilerplate

HERE = os.path.dirname(os.path.abspath(__file__))
STUDY = os.path.join(HERE, "hcr_study")


# =====================================================================================================================
# Training helpers
# =====================================================================================================================
#
# Training helpers for train.py final: seeding, and a Hugging Face Trainer with class-weighted cross-entropy that is
# evaluated on the validation split after every epoch (survey = label 0 = positive class for all metrics), keeps the
# best epoch by survey-class F1 and stops early after two epochs without improvement.

warnings.simplefilter("ignore", category=FutureWarning)

SURVEY_LABEL = 0  # label ids: 0 = survey, 1 = non-survey


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class SurveyDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {k: v[idx] for k, v in self.encodings.items()}
        item["labels"] = int(self.labels[idx])
        return item


# -------------------- Metrics (survey = positive) -------------------- #
def survey_metrics(is_survey_true: np.ndarray, is_survey_pred: np.ndarray) -> dict:
    precision, recall, f1, _ = precision_recall_fscore_support(
        is_survey_true, is_survey_pred, average="binary", pos_label=1, zero_division=0
    )
    return {"accuracy": accuracy_score(is_survey_true, is_survey_pred),
            "precision": precision, "recall": recall, "f1": f1}


def compute_metrics(pred):
    logits, labels = pred.predictions, pred.label_ids
    is_survey_pred = (np.argmax(logits, axis=1) == SURVEY_LABEL).astype(int)
    return survey_metrics((labels == SURVEY_LABEL).astype(int), is_survey_pred)


class WeightedLossTrainer(Trainer):
    """Cross-entropy with class weights (the Trainer ignores model.loss_fct)."""

    def __init__(self, *args, class_weights=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = class_weights

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        weight = self.class_weights.to(outputs.logits.device) if self.class_weights is not None else None
        loss = torch.nn.functional.cross_entropy(outputs.logits.float(), labels, weight=weight)
        return (loss, outputs) if return_outputs else loss


def make_trainer(model, tokenizer, train_df, val_df, output_dir, epochs, batch_size, lr,
                 weight_decay=0.01, seed=42, save_best=True, fp16=None) -> Trainer:
    def encode(frame):
        enc = tokenizer(frame["Text"].tolist(), truncation=True, max_length=MAX_LENGTH)
        return SurveyDataset(enc, frame["Label"].values)

    labels = train_df["Label"].values
    weights = compute_class_weight("balanced", classes=np.array([0, 1]), y=labels)
    total_steps = epochs * int(np.ceil(len(train_df) / batch_size))

    args = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=64,
        learning_rate=lr,
        weight_decay=weight_decay,
        warmup_steps=int(0.1 * total_steps),
        lr_scheduler_type="linear",
        eval_strategy="epoch",
        save_strategy="epoch" if save_best else "no",
        load_best_model_at_end=save_best,
        metric_for_best_model="f1",
        greater_is_better=True,
        save_total_limit=1,
        logging_steps=50,
        fp16=torch.cuda.is_available() if fp16 is None else fp16,   # None: half precision on CUDA
        seed=seed,
        report_to="none",
        dataloader_num_workers=0,
    )
    return WeightedLossTrainer(
        model=model,
        args=args,
        train_dataset=encode(train_df),
        eval_dataset=encode(val_df),
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=2)] if save_best else None,
        class_weights=torch.tensor(weights, dtype=torch.float),
    )


# =====================================================================================================================
# Training and test of SciGenre and of the fine-tuned baselines
# =====================================================================================================================

def train_final():
    """Train and test the SciGenre on the one venue-labeled ground truth (hcr_study/venue_all/ground_truth.csv).
    Data: every usable paper (abstract present, not a non-paper) of 18 survey venues and 14 research venues, selected by
    their stated scope (ground_truth.py venues); the venue is the label. Split: random by paper, stratified by venue, 80% train / 10% validation / 10% test, made once
    (seed 2026) and saved to hcr_study/venue_all/split.csv.
    Model: SPECTER2 (allenai/specter2_base), input title + abstract only (common.model_text; never the venue),
    class-weighted cross-entropy, lr 2e-5, batch 16, 10% warm-up, 384 tokens; one pass over the training papers
    (76,396), decision threshold 0.5.
    Training runs in full precision on every device.
    Usage: train.py final [--seed 42] [--epochs 1]   Output: hcr_study/models_final/seed<seed>/ (model, preds.csv, metrics.json)
    Baselines with the same data, split and recipe: train.py final --model allenai/scibert_scivocab_uncased --tag scibert
      Output: hcr_study/models_baselines/<tag>/"""
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(); ap.add_argument("--seed", type=int, default=42); ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--split-only", action="store_true", help="draw the split (needed with a downloaded model) and stop")
    ap.add_argument("--model", default="allenai/specter2_base"); ap.add_argument("--tag", help="baseline name: write to hcr_study/models_baselines/<tag>")
    args = ap.parse_args()
    GT = os.path.join(HERE, "hcr_study", "venue_all"); OUT = os.path.join(HERE, "hcr_study", "models_final", f"seed{args.seed}") if args.tag is None else os.path.join(HERE, "hcr_study", "models_baselines", args.tag)
    os.makedirs(OUT, exist_ok=True)
    PI = 0.08

    # ---- data and split ----
    G = pd.read_csv(os.path.join(GT, "ground_truth.csv"), dtype={"OpenAlexId": str, "doi": str})
    G = G[G.usable].reset_index(drop=True)
    SPLIT = os.path.join(GT, "split.csv")
    if not os.path.exists(SPLIT):
        rng = np.random.default_rng(2026); parts = []
        for venue, d in G.groupby("venue"):
            r = rng.random(len(d)); parts.append(pd.DataFrame({"OpenAlexId": d.OpenAlexId.values,
                                                               "split": np.where(r < 0.8, "train", np.where(r < 0.9, "val", "test"))}))
        pd.concat(parts).to_csv(SPLIT, index=False)
    if args.split_only:
        print(f"split: {SPLIT}"); sys.exit(0)
    G = G.merge(pd.read_csv(SPLIT, dtype={"OpenAlexId": str}), on="OpenAlexId")
    G["Label"] = np.where(G.label == "survey", 0, 1)            # 0 = survey, 1 = non-survey
    G["Text"] = [model_text(t, a) for t, a in zip(G.title, G.abstract)]
    tr, va, te = G[G.split == "train"], G[G.split == "val"], G[G.split == "test"]
    print(f"train {len(tr)} ({(tr.Label == 0).sum()} surveys), val {len(va)}, test {len(te)}", flush=True)

    # ---- model ----
    set_seed(args.seed)
    dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=2, id2label={0: "survey", 1: "non-survey"},
                                                               label2id={"survey": 0, "non-survey": 1}, dtype=torch.float32)
    t0 = time.time()
    trainer = make_trainer(model, tok, tr, va, output_dir=os.path.join(OUT, "ckpt"), epochs=args.epochs, batch_size=16, lr=2e-5, seed=args.seed, fp16=False)
    trainer.data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=64)   # fixed shapes keep Apple MPS fast
    trainer.train()
    minutes = (time.time() - t0) / 60
    model = trainer.model.eval().to(dev)
    trainer.save_model(os.path.join(OUT, "model")); tok.save_pretrained(os.path.join(OUT, "model"))

    @torch.no_grad()
    def proba(texts, bs=64):
        order = np.argsort([len(t) for t in texts]); out = np.empty(len(texts))
        for i in range(0, len(texts), bs):
            idx = order[i:i + bs]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LENGTH, padding=True, pad_to_multiple_of=64, return_tensors="pt").to(dev)
            out[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, 0].cpu().numpy()
        return out

    def metrics(d):
        y, f = (d.Label == 0).values, (d.p >= 0.5).values
        fpr = f[~y].mean(); rec = f[y].mean(); prec = (f & y).sum() / max(f.sum(), 1)
        p8 = rec * PI / (rec * PI + fpr * (1 - PI)) if rec > 0 else 0.0
        return {"papers": len(d), "surveys": int(y.sum()), "FPR": 100 * fpr, "recall": 100 * rec, "precision": 100 * prec,
                "F1": 100 * 2 * prec * rec / max(prec + rec, 1e-9), "precision_8": 100 * p8, "F1_8": 100 * 2 * p8 * rec / max(p8 + rec, 1e-9)}

    rows = []
    for name, d in (("val", va), ("test", te)):
        d = d.copy(); d["p"] = proba(d.Text.tolist()); rows.append(d[["OpenAlexId", "venue", "label", "split", "p"]])
    P = pd.concat(rows); P.to_csv(os.path.join(OUT, "preds.csv"), index=False)
    P["Label"] = np.where(P.label == "survey", 0, 1)
    res = {"model": args.model, "device": torch.cuda.get_device_name(0) if dev.type == "cuda" else dev.type, "seed": args.seed, "epochs": args.epochs, "train_minutes": round(minutes, 1),
           "val": metrics(P[P.split == "val"]), "test": metrics(P[P.split == "test"]),
           "test_by_venue": {v: metrics(d) for v, d in P[P.split == "test"].groupby("venue")}}
    json.dump(res, open(os.path.join(OUT, "metrics.json"), "w"), indent=1, default=float)
    print(json.dumps({k: res[k] for k in ("val", "test")}, indent=1, default=lambda x: round(float(x), 2)))


# =====================================================================================================================
# Installation of the final model
# =====================================================================================================================

def install_model():
    """Install the final SciGenre (train.py final, hcr_study/models_final/seed42) as the model every HCR analysis uses
    (model/, read by profiles.py) and record its configuration and its error rates on the held-out test split,
    which the error correction and the outlier simulation use. An existing model/ is replaced."""
    SRC = os.path.join(HERE, "hcr_study", "models_final", "seed42"); DST = os.path.join(HERE, "model")
    M = json.load(open(os.path.join(SRC, "metrics.json")))
    if os.path.exists(DST):
        shutil.rmtree(DST)
    shutil.copytree(os.path.join(SRC, "model"), DST)
    cfg = {"base_model": "allenai/specter2_base",
           "input": "model_text(title, abstract) of common.py and predict.py; never the venue",
           "training_data": "hcr_study/venue_all/ground_truth.csv (18 survey venues and 14 research venues, selected by their "
                            "officially stated scope; venue as label, no hand checking), train split of split.csv (80/10/10 by paper, "
                            "stratified by venue, seed 2026)",
           "training": {"epochs": M["epochs"], "lr": 2e-5, "batch_size": 16, "loss": "class-weighted cross-entropy", "seed": M["seed"],
                        "minutes": M["train_minutes"], "device": M.get("device", "mps")},
           "threshold": 0.5, "min_abstract_words": 10, "max_length": 384, "survey_label_index": 0,
           "test_recall": M["test"]["recall"], "test_fpr": M["test"]["FPR"], "test_papers": M["test"]["papers"],
           "test_surveys": M["test"]["surveys"], "val": M["val"], "test": M["test"]}
    json.dump(cfg, open(os.path.join(DST, "scigenre.json"), "w"), indent=1)
    print(f"installed {SRC} -> {DST}; test recall {cfg['test_recall']:.2f}%, FPR {cfg['test_fpr']:.3f}%")


# =====================================================================================================================
# Leave-venue-out test
# =====================================================================================================================

def venue_holdout():
    """Leave-venue-out test of the SciGenre: does it work on venues it never saw in training?
    The ground-truth venues are split into 3 folds. For each fold, a model is trained exactly as the released one
    (train.py final: SPECTER2, title + abstract, class-weighted cross-entropy, lr 2e-5, batch 16, one epoch, seed 42) on
    the training papers of the other two folds' venues, and it scores every paper of the held-out venues. Every paper
    of every venue is therefore scored once by a model that never saw its venue. Each fold model also scores the test
    papers of its own training venues, so seen and unseen venues are compared within the same model.
    Venues of one series share a title and house format and are held out together (all 14 Foundations and Trends
    journals; the Proceedings of the ACM journals). Survey venues: ACM Computing Surveys alone (half of all
    surveys), IEEE Communications Surveys & Tutorials with the Annual Review, Computer Science Review with Foundations
    and Trends. Research venues are spread over the folds to balance paper counts (largest first into the smallest fold).
    Usage: train.py holdout [--fold 1|2|3] (default: all folds, skipping finished ones); train.py holdout --summary
    Output: hcr_study/models_venue_holdout/folds.csv, fold<k>/ (model, preds.csv, metrics.json), summary.{json,md}"""
    warnings.simplefilter("ignore")
    ap = argparse.ArgumentParser(); ap.add_argument("--fold", type=int); ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    GT = os.path.join(HERE, "hcr_study", "venue_all"); OUT = os.path.join(HERE, "hcr_study", "models_venue_holdout"); os.makedirs(OUT, exist_ok=True)
    PI, SEED = 0.08, 42

    G = pd.read_csv(os.path.join(GT, "ground_truth.csv"), dtype={"OpenAlexId": str, "doi": str})
    G = G[G.usable].reset_index(drop=True)
    G = G.merge(pd.read_csv(os.path.join(GT, "split.csv"), dtype={"OpenAlexId": str}), on="OpenAlexId")


    def group(venue):
        if venue.startswith("F&T "): return "Foundations and Trends (14 journals)"
        if venue.startswith("Proceedings of the ACM on "): return "Proceedings of the ACM journals"
        return venue


    # ---- fold assignment (deterministic) ----
    FOLDS = os.path.join(OUT, "folds.csv")
    if not os.path.exists(FOLDS):
        survey_fold = {"ACM Computing Surveys": 1, "IEEE Communications Surveys & Tutorials": 2,
                       "Annual Review of Control, Robotics, and Autonomous Systems": 2, "Computer Science Review": 3,
                       "Foundations and Trends (14 journals)": 3}
        G["group"] = G.venue.map(group)
        size = G[G.label == "research"].groupby("group").size().sort_values(ascending=False)
        load, research_fold = {1: 0, 2: 0, 3: 0}, {}
        for g, n in size.items():
            k = min(load, key=lambda f: (load[f], f)); research_fold[g] = k; load[k] += n
        gf = {**survey_fold, **research_fold}
        V = G.groupby(["venue", "label"]).size().rename("papers").reset_index()
        V["group"] = V.venue.map(group); V["fold"] = V.group.map(gf)
        assert V.fold.notna().all(), V[V.fold.isna()]
        V.sort_values(["fold", "label", "papers"], ascending=[True, False, False]).to_csv(FOLDS, index=False)
    F = pd.read_csv(FOLDS)
    G = G.merge(F[["venue", "fold"]], on="venue")


    def metrics(d):
        y, f = (d.label == "survey").values, (d.p >= 0.5).values
        fpr = f[~y].mean() if (~y).any() else np.nan; rec = f[y].mean() if y.any() else np.nan
        prec = (f & y).sum() / max(f.sum(), 1)
        p8 = rec * PI / (rec * PI + fpr * (1 - PI)) if y.any() and (~y).any() and rec > 0 else np.nan
        return {"papers": len(d), "surveys": int(y.sum()), "FPR": 100 * fpr, "recall": 100 * rec, "precision": 100 * prec,
                "F1": 100 * 2 * prec * rec / max(prec + rec, 1e-9), "precision_8": 100 * p8,
                "F1_8": 100 * 2 * p8 * rec / (p8 + rec) if p8 == p8 else np.nan}


    def run_fold(k):
        import torch
        from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding
        out = os.path.join(OUT, f"fold{k}"); os.makedirs(out, exist_ok=True)
        D = G.copy()
        D["Label"] = np.where(D.label == "survey", 0, 1)
        D["Text"] = [model_text(t, a) for t, a in zip(D.title, D.abstract)]
        seen = D.fold != k
        tr, va = D[seen & (D.split == "train")], D[seen & (D.split == "val")]
        score = D[~seen | (D.split == "test")].copy()          # all papers of held-out venues + test papers of seen venues
        score["venue_seen"] = score.fold != k
        print(f"fold {k}: train {len(tr)} ({(tr.Label == 0).sum()} surveys), val {len(va)}, "
              f"score {len(score)} ({(~score.venue_seen).sum()} from held-out venues)", flush=True)
        set_seed(SEED)
        dev = torch.device("mps" if torch.backends.mps.is_available() else ("cuda" if torch.cuda.is_available() else "cpu"))
        tok = AutoTokenizer.from_pretrained("allenai/specter2_base")
        model = AutoModelForSequenceClassification.from_pretrained("allenai/specter2_base", num_labels=2, id2label={0: "survey", 1: "non-survey"},
                                                                   label2id={"survey": 0, "non-survey": 1}, dtype=torch.float32)
        t0 = time.time()
        trainer = make_trainer(model, tok, tr, va, output_dir=os.path.join(out, "ckpt"), epochs=1, batch_size=16, lr=2e-5, seed=SEED, fp16=False)
        trainer.data_collator = DataCollatorWithPadding(tok, pad_to_multiple_of=64)
        trainer.train()
        minutes = (time.time() - t0) / 60
        model = trainer.model.eval().to(dev)
        trainer.save_model(os.path.join(out, "model")); tok.save_pretrained(os.path.join(out, "model"))
        shutil.rmtree(os.path.join(out, "ckpt"), ignore_errors=True)   # optimizer state, not needed

        @torch.no_grad()
        def proba(texts, bs=64):
            order = np.argsort([len(t) for t in texts]); res = np.empty(len(texts))
            for i in range(0, len(texts), bs):
                idx = order[i:i + bs]
                enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LENGTH, padding=True, pad_to_multiple_of=64, return_tensors="pt").to(dev)
                res[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, 0].cpu().numpy()
            return res

        score["p"] = proba(score.Text.tolist())
        score[["OpenAlexId", "venue", "label", "split", "fold", "venue_seen", "p"]].to_csv(os.path.join(out, "preds.csv"), index=False)
        res = {"fold": k, "seed": SEED, "train_papers": len(tr), "train_surveys": int((tr.Label == 0).sum()), "train_minutes": round(minutes, 1),
               "held_out_venues": metrics(score[~score.venue_seen]), "seen_venues_test": metrics(score[score.venue_seen])}
        json.dump(res, open(os.path.join(out, "metrics.json"), "w"), indent=1, default=float)
        print(json.dumps(res, indent=1, default=lambda x: round(float(x), 2)), flush=True)


    def summary():
        P = pd.concat([pd.read_csv(os.path.join(OUT, f"fold{k}", "preds.csv"), dtype={"OpenAlexId": str}) for k in (1, 2, 3)])
        U = P[~P.venue_seen]
        assert U.OpenAlexId.is_unique and len(U) == len(G), (len(U), len(G))
        final = pd.read_csv(os.path.join(HERE, "hcr_study", "models_final", "seed42", "preds.csv"), dtype={"OpenAlexId": str})
        test_ids = set(G.loc[G.split == "test", "OpenAlexId"])
        res = {"unseen_all_papers": metrics(U),
               "unseen_test_papers": metrics(U[U.OpenAlexId.isin(test_ids)]),          # the test papers of Table RQ1
               "released_model_test_papers": metrics(final[final.split == "test"]),
               "seen_test_papers_fold_models": metrics(P[P.venue_seen]),
               "by_fold": {int(k): {"unseen": metrics(d[~d.venue_seen]), "seen_test": metrics(d[d.venue_seen])} for k, d in P.groupby("fold")},
               "by_venue_unseen": {v: metrics(d) for v, d in U.groupby("venue")}}
        from statsmodels.stats.proportion import proportion_confint
        for key in ("unseen_all_papers", "unseen_test_papers"):
            d = U if key == "unseen_all_papers" else U[U.OpenAlexId.isin(test_ids)]
            y, f = (d.label == "survey").values, (d.p >= 0.5).values
            res[key]["recall CI"] = [100 * x for x in proportion_confint(int((f & y).sum()), int(y.sum()), method="wilson")]
            res[key]["FPR CI"] = [100 * x for x in proportion_confint(int((f & ~y).sum()), int((~y).sum()), method="wilson")]
            res[key]["research flagged"] = int((f & ~y).sum()); res[key]["surveys found"] = int((f & y).sum())
        json.dump(res, open(os.path.join(OUT, "summary.json"), "w"), indent=1, default=float)
        labels = {"unseen_all_papers": "Held-out venues, all papers (fold models)", "unseen_test_papers": "Held-out venues, test papers only (fold models)",
                  "released_model_test_papers": "Test papers, released model (venues seen)", "seen_test_papers_fold_models": "Test papers of training venues (fold models)"}
        md = ["# Leave-venue-out test (3 folds; series held out together)", "",
              "| Papers | n | Surveys | Recall | FPR | F1 at 8% |", "|---|---|---|---|---|---|"]
        for key, lab in labels.items():
            m = res[key]; md.append(f"| {lab} | {m['papers']} | {m['surveys']} | {m['recall']:.1f}% | {m['FPR']:.2f}% | {m['F1_8']:.1f}% |")
        for key in ("unseen_all_papers", "unseen_test_papers"):
            m = res[key]
            md.append(f"\n{labels[key]}: recall 95% CI {m['recall CI'][0]:.1f} to {m['recall CI'][1]:.1f}; FPR 95% CI {m['FPR CI'][0]:.2f} to {m['FPR CI'][1]:.2f} "
                      f"({m['surveys found']} of {m['surveys']} surveys, {m['research flagged']} research papers labeled survey)")
        md += ["", "## By fold", "", "| Fold | Held-out surveys | Recall | Held-out research | FPR | Seen-venue test recall | Seen-venue test FPR |", "|---|---|---|---|---|---|---|"]
        for k, v in res["by_fold"].items():
            u, sn = v["unseen"], v["seen_test"]
            md.append(f"| {k} | {u['surveys']} | {u['recall']:.1f}% | {u['papers'] - u['surveys']} | {u['FPR']:.2f}% | {sn['recall']:.1f}% | {sn['FPR']:.2f}% |")
        md += ["", "## By held-out venue", "", "| Venue | Label | Papers | Labeled survey |", "|---|---|---|---|"]
        lab_of = G.drop_duplicates("venue").set_index("venue").label
        for v, m in sorted(res["by_venue_unseen"].items(), key=lambda kv: (lab_of[kv[0]] != "survey", -kv[1]["papers"])):
            share = m["recall"] if lab_of[v] == "survey" else m["FPR"]
            md.append(f"| {v} | {lab_of[v]} | {m['papers']} | {share:.2f}% |")
        open(os.path.join(OUT, "summary.md"), "w").write("\n".join(md) + "\n")
        print("\n".join(md[:8]))


    if args.summary:
        summary()
    else:
        for k in ([args.fold] if args.fold else [1, 2, 3]):
            if os.path.exists(os.path.join(OUT, f"fold{k}", "metrics.json")):
                print(f"fold {k} done, skipping", flush=True); continue
            run_fold(k)
        if not args.fold:
            summary()


# =====================================================================================================================
# Scoring of the analyses' texts on a GPU
# =====================================================================================================================

def score_texts():
    """Score on a GPU every text the analyses classify (hcr_study/scores/texts.jsonl, recorded with
    SE_COLLECT=1; see profiles.py) with the paper's model, hcr_study/models_final/seed42/model, exactly as profiles.score_texts
    does: length-sorted batches of 64, 384 tokens, padding to a multiple of 64, P(survey) from the softmax.
    Output: hcr_study/scores/scores.csv (h = SHA-1 of the text, p) and scores.json (SHA-256 of the model weights)."""
    MODEL_DIR = os.path.join(HERE, "hcr_study", "models_final", "seed42", "model"); D = os.environ.get("SE_SCORES_DIR") or os.path.join(HERE, "hcr_study", "scores")
    MAX_LEN, BS = 384, 64

    h = hashlib.sha256()
    with open(os.path.join(MODEL_DIR, "model.safetensors"), "rb") as f:
        for block in iter(lambda: f.read(1 << 24), b""): h.update(block)
    T = pd.read_json(os.path.join(D, "texts.jsonl"), lines=True, dtype={"h": str}).drop_duplicates("h").reset_index(drop=True)
    assert all(hashlib.sha1(t.encode("utf-8")).hexdigest() == k for k, t in zip(T.h, T.t)), "text hashes do not match"
    dev = torch.device("cuda")
    tok = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR, dtype=torch.float32).to(dev).eval()
    survey = {v: int(k) for k, v in model.config.id2label.items()}["survey"]
    texts = T.t.tolist(); order = np.argsort([len(t) for t in texts]); p = np.empty(len(texts)); t0 = time.time()
    with torch.no_grad():
        for i in range(0, len(texts), BS):
            idx = order[i:i + BS]
            enc = tok([texts[j] for j in idx], truncation=True, max_length=MAX_LEN, padding=True, pad_to_multiple_of=64, return_tensors="pt").to(dev)
            p[idx] = torch.softmax(model(**enc).logits.float(), -1)[:, survey].cpu().numpy()
    pd.DataFrame({"h": T.h, "p": p}).to_csv(os.path.join(D, "scores.csv"), index=False, float_format="%.10g")
    json.dump({"model_sha256": h.hexdigest(), "texts": len(T), "device": torch.cuda.get_device_name(0), "minutes": round((time.time() - t0) / 60, 1),
               "date": time.strftime("%Y-%m-%d %H:%M")}, open(os.path.join(D, "scores.json"), "w"), indent=1)
    print(f"scored {len(T)} texts in {(time.time() - t0) / 60:.1f} min on {torch.cuda.get_device_name(0)}; {int((p >= 0.5).sum())} at P >= 0.5")


# =====================================================================================================================
# Zero-shot LLM baseline
# =====================================================================================================================

def llm_zero_shot():
    """Zero-shot LLM baseline for RQ1. Same 9,555 test papers as every other method; title and
    abstract cleaned as for the SciGenre (common.model_text). Open models only, run with vLLM 0.31 on two 48 GB GPUs (TP=2).
    Usage: train.py llm gpt-oss-120b
    Output: hcr_study/models_baselines/llm_<name>/preds.csv (p = 1 for an answer of survey), outputs.jsonl (every raw answer), info.json"""
    from vllm import LLM, SamplingParams
    import vllm, torch

    MODELS = {  # model card settings; revisions pinned on 9 Oct 2026
        "gpt-oss-120b": dict(id="openai/gpt-oss-120b", revision="b5c939de8f754692c1647ca79fbf85e8c1e70f8a",
                             sampling=dict(temperature=1.0, top_p=1.0), template=dict(reasoning_effort="medium")),
    }
    PROMPT = ("Below are the title and abstract of a scientific paper. Decide whether the paper is a survey or a research paper. "
              "A survey reviews, summarises or organises existing work (this includes reviews, overviews and tutorials). "
              "A research paper reports original work, such as new methods, results, systems, proofs or measurements.\n\n"
              "Title: {title}\n\nAbstract: {abstract}\n\nAnswer with exactly one word: survey or research.")
    def answer(raw):
        """The text after the reasoning; None when the reasoning never finished."""
        for marker in ("<|channel|>final<|message|>", "</think>"):
            if marker in raw: return raw.rsplit(marker, 1)[1]
        return None if ("<think>" in raw or "<|channel|>analysis" in raw) else raw

    def label(raw):
        a = answer(raw)
        if a is None: return "unparsed"
        a = re.sub(r"<\|[^|>]*\|>", " ", a).strip().lower()
        s, r = "survey" in a, "research" in a
        return "survey" if s and not r else ("research" if r and not s else "unparsed")


    def main():
        name = sys.argv[1]; M = MODELS[name]
        OUT = os.path.join(HERE, "hcr_study", "models_baselines", f"llm_{name}"); os.makedirs(OUT, exist_ok=True)

        GT = os.path.join(HERE, "hcr_study", "venue_all")
        G = pd.read_csv(os.path.join(GT, "ground_truth.csv"), dtype={"OpenAlexId": str, "doi": str})
        G = G[G.usable].merge(pd.read_csv(os.path.join(GT, "split.csv"), dtype={"OpenAlexId": str}), on="OpenAlexId")
        te = G[G.split == "test"].reset_index(drop=True)
        print(f"{name}: {len(te)} test papers ({(te.label == 'survey').sum()} surveys)", flush=True)
        msgs = [[{"role": "user", "content": PROMPT.format(title=clean_text(t), abstract=strip_boilerplate(clean_text(a)))}]
                for t, a in zip(te.title, te.abstract)]

        MAX_TOKENS = 8192
        TP = int(os.environ.get("TP", "1"))                     # GPUs per model (tensor parallel)
        llm = LLM(model=M["id"], revision=M["revision"], seed=42, max_model_len=16384, gpu_memory_utilization=0.9, tensor_parallel_size=TP, **M.get("engine", {}))
        sp = SamplingParams(**M["sampling"], max_tokens=MAX_TOKENS, seed=42, skip_special_tokens=False)
        t0 = time.time()
        outs = llm.chat(msgs, sp, chat_template_kwargs=M["template"])
        minutes = (time.time() - t0) / 60

        raw = [o.outputs[0].text for o in outs]
        te["answer"] = [label(x) for x in raw]
        te["p"] = (te.answer == "survey").astype(float)          # unparsed answers count as research (not flagged)
        te[["OpenAlexId", "venue", "label", "split", "answer", "p"]].to_csv(os.path.join(OUT, "preds.csv"), index=False)
        with open(os.path.join(OUT, "outputs.jsonl"), "w") as f:
            for i, x, o in zip(te.OpenAlexId, raw, outs):
                f.write(json.dumps({"OpenAlexId": i, "tokens": len(o.outputs[0].token_ids), "finish": o.outputs[0].finish_reason, "raw": x}) + "\n")
        y, f = (te.label == "survey").values, (te.answer == "survey").values
        rec, fpr = f[y].mean(), f[~y].mean(); prec = (f & y).sum() / max(f.sum(), 1)
        info = {"model": M["id"], "revision": M["revision"], "sampling": {**M["sampling"], "max_tokens": MAX_TOKENS, "seed": 42},
                "chat_template_kwargs": M["template"], "vllm": vllm.__version__, "torch": torch.__version__, "gpu": f"{TP} x {torch.cuda.get_device_name(0)}",
                "minutes": round(minutes, 1), "papers": len(te), "answers": te.answer.value_counts().to_dict(),
                "truncated": int(sum(o.outputs[0].finish_reason == "length" for o in outs)),
                "recall": 100 * rec, "FPR": 100 * fpr, "precision": 100 * prec, "F1": 100 * 2 * prec * rec / max(prec + rec, 1e-9),
                "example_prompt": llm.get_tokenizer().apply_chat_template(msgs[0], tokenize=False, add_generation_prompt=True, **M["template"])}
        json.dump(info, open(os.path.join(OUT, "info.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in info.items() if k != "example_prompt"}, indent=1, default=str))


    main()


STEPS = {"final": train_final, "install": install_model, "holdout": venue_holdout, "score": score_texts, "llm": llm_zero_shot}

if __name__ == "__main__":
    if len(sys.argv) < 2 or sys.argv[1] not in STEPS:
        sys.exit(__doc__)
    STEPS[sys.argv.pop(1)]()
