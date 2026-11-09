"""Criterion-level metrics (assignment Task 2 / rubric C1-C3).

Implemented in plain NumPy so results are auditable line by line and the
module runs anywhere (no sklearn dependency on Colab).

Classification: accuracy, macro-F1, per-class precision/recall/F1 and the
full confusion matrix over the *preserved* label set (the six expert
categories, never collapsed to yes/no).

Evidence retrieval: sentence-level precision/recall/F1.
Convention (stated here and in the report):
  * micro  = pooled tp/fp/fn over all examples;
  * macro  = mean of per-example F1;
  * examples with an EMPTY gold set are excluded from the macro mean and
    reported separately (an annotator found nothing to cite is a property of
    the record, not a retrieval mistake).  We additionally report the
    both-empty agreement rate.  An empty *prediction* on a non-empty gold
    set scores 0 recall / 0 precision by construction.
"""
from __future__ import annotations

import json
from collections import Counter

import numpy as np


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------
def classification_metrics(gold: list[str], pred: list[str], labels: list[str]):
    idx = {lab: i for i, lab in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)
    for g, p in zip(gold, pred):
        cm[idx[g], idx[p]] += 1

    correct = int(np.trace(cm))
    total = int(cm.sum())
    per_class: dict[str, dict] = {}
    f1s = []
    for i, lab in enumerate(labels):
        tp = cm[i, i]
        fp = cm[:, i].sum() - tp
        fn = cm[i, :].sum() - tp
        prec = tp / (tp + fp) if tp + fp else 0.0
        rec = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        per_class[lab] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": int(cm[i, :].sum()),
        }
        if cm[i, :].sum() > 0:          # only classes present in the gold data
            f1s.append(f1)

    macro_f1 = float(np.mean(f1s)) if f1s else 0.0
    return {
        "labels": labels,
        "accuracy": round(correct / total, 4) if total else 0.0,
        "macro_f1": round(macro_f1, 4),
        "correct": correct,
        "total": total,
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),   # rows = gold, cols = predicted
    }


# ---------------------------------------------------------------------------
# Evidence sentence retrieval
# ---------------------------------------------------------------------------
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return prec, rec, f1


def evidence_metrics(gold_sets: list[list[int]], pred_sets: list[list[int]]):
    """Micro/macro evidence P/R/F1 with the empty-gold convention above."""
    tp = fp = fn = 0
    per_ex_f1, per_ex_p, per_ex_r = [], [], []
    n_empty_gold = n_empty_gold_empty_pred = 0
    both_empty_correct = 0

    for gold, pred in zip(gold_sets, pred_sets):
        g, p = set(gold), set(pred)
        if not g:
            n_empty_gold += 1
            if not p:
                n_empty_gold_empty_pred += 1
                both_empty_correct += 1
            else:
                fp += len(p)          # predicted evidence on an unannotatable record
            continue
        tp += len(g & p)
        fp += len(p - g)
        fn += len(g - p)
        ep, er, ef = _prf(len(g & p), len(p - g), len(g - p))
        per_ex_p.append(ep)
        per_ex_r.append(er)
        per_ex_f1.append(ef)

    mp, mr, mf = _prf(tp, fp, fn)
    return {
        "micro": {"precision": round(mp, 4), "recall": round(mr, 4), "f1": round(mf, 4)},
        "macro": {
            "precision": round(float(np.mean(per_ex_p)), 4) if per_ex_p else 0.0,
            "recall": round(float(np.mean(per_ex_r)), 4) if per_ex_r else 0.0,
            "f1": round(float(np.mean(per_ex_f1)), 4) if per_ex_f1 else 0.0,
            "n_examples": len(per_ex_f1),
        },
        "counts": {
            "tp": tp, "fp": fp, "fn": fn,
            "empty_gold": n_empty_gold,
            "empty_gold_empty_pred": n_empty_gold_empty_pred,
            "empty_gold_nonempty_pred": n_empty_gold - n_empty_gold_empty_pred,
            "nonempty_gold_empty_pred": sum(
                1 for g, p in zip(gold_sets, pred_sets) if g and not p
            ),
        },
        "convention": (
            "micro = pooled counts over all examples with non-empty gold; "
            "macro = mean per-example over non-empty gold; empty-gold records "
            "excluded from macro and reported separately"
        ),
    }


# ---------------------------------------------------------------------------
# Helpers used by the report / error analysis
# ---------------------------------------------------------------------------
def per_type_metrics(gold, pred, types, labels_for_type):
    """Split classification metrics into inclusion vs exclusion criteria."""
    out = {}
    for ctype, labels in labels_for_type.items():
        g = [x for x, t in zip(gold, types) if t == ctype]
        p = [x for x, t in zip(pred, types) if t == ctype]
        out[ctype] = classification_metrics(g, p, list(labels)) if g else None
    return out


def selftest():
    """Tiny sanity checks with hand-computed answers."""
    gold = ["included", "not included", "not enough information", "excluded"]
    pred = ["included", "included", "not enough information", "excluded"]
    labels = ["included", "not included", "not enough information",
              "excluded", "not excluded", "not applicable"]
    m = classification_metrics(gold, pred, labels)
    assert m["accuracy"] == 0.75 and m["correct"] == 3
    assert abs(m["per_class"]["included"]["f1"] - 2 / 3) < 1e-3

    g = [[1, 2], [3], []]
    p = [[1], [3, 4], []]
    em = evidence_metrics(g, p)
    # pooled: tp=2, fp=1, fn=1 (example 3 excluded: empty gold, empty pred)
    assert em["counts"]["tp"] == 2 and em["counts"]["fp"] == 1 and em["counts"]["fn"] == 1
    assert em["counts"]["empty_gold"] == 1 and em["counts"]["empty_gold_empty_pred"] == 1
    # per-example F1: ex1 p=1/1? tp=1,fp=0,fn=1 -> f1=2/3 ; ex2 tp=1,fp=1,fn=0 -> 2/3
    assert abs(em["macro"]["f1"] - 2 / 3) < 1e-3
    print("metrics selftest OK")


if __name__ == "__main__":
    selftest()
