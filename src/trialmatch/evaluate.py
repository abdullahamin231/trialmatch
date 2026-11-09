"""Evaluation runner (assignment Tasks 1-2, rubric B4, C1-C4).

Runs a checkpoint variant over a processed split with the production
decoding path, computes all common metrics, renders confusion matrices and
selects a reproducible set of representative errors.

    python -m trialmatch.evaluate --variant base --splits validation,test
    python -m trialmatch.evaluate --variant tuned --splits validation,test

Outputs (all under results/):
    eval_<variant>_<split>.json      metrics + decoding settings + config
    preds_<variant>_<split>.jsonl    one record per annotation
    errors_<variant>_<split>.json    categorised representative failures
    figures/cm_<variant>_<split>_<scope>.png
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path

import torch

from .config import (
    ADAPTER_DIR,
    LABELS,
    LABELS_FOR_TYPE,
    TARGET_MODEL,
    RESULTS_DIR,
)
from .data import Example, read_split, parse_note
from .metrics import classification_metrics, evidence_metrics, per_type_metrics
from .pipeline import Predictor

FIG_DIR = RESULTS_DIR / "figures"


# ---------------------------------------------------------------------------
def _sentence_text(ex: Example, sid: int) -> str:
    for s, text in ex.sentences:
        if s == sid:
            return text
    return ""


def _prediction_record(ex: Example, pred) -> dict:
    return {
        "annotation_id": ex.annotation_id,
        "patient_id": ex.patient_id,
        "trial_id": ex.trial_id,
        "criterion_type": ex.criterion_type,
        "criterion_text": ex.criterion_text,
        "gold_label": ex.label,
        "gold_evidence_ids": ex.evidence_ids,
        "pred_label": pred.label,
        "pred_label_valid": pred.label_valid,
        "pred_evidence_ids": pred.evidence_ids,
        "pred_evidence_texts": {str(i): _sentence_text(ex, i) for i in pred.evidence_ids},
        "raw_output": pred.raw,
        "parse_notes": pred.notes,
        "latency_s": round(pred.latency_s, 4),
        "n_prompt_tokens": pred.n_prompt_tokens,
        "n_generated": pred.n_generated,
    }


def _confusion_png(cm, labels, path: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(1.1 * len(labels) + 2.2, 1.0 * len(labels) + 2.0))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels=labels, rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels=labels, fontsize=8)
    for i in range(len(labels)):
        for j in range(len(labels)):
            v = cm[i][j]
            if v:
                ax.text(j, i, str(v), ha="center", va="center", fontsize=8,
                        color="white" if v > cm.max() / 2 else "black")
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("gold", fontsize=9)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, shrink=0.8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Error taxonomy (rubric C4: at least 8 distinct errors, incl. unsupported /
# missing-information cases)
# ---------------------------------------------------------------------------
def categorise_errors(records: list[dict]) -> dict[str, list[dict]]:
    buckets: dict[str, list[dict]] = {
        "wrong_label_evidence_ok": [],
        "wrong_label_wrong_evidence": [],
        "missing_evidence": [],            # gold nonempty, pred empty
        "hallucinated_evidence": [],       # gold empty, pred nonempty
        "insufficient_info_failure": [],   # gold NEI, pred positive/negative
        "cross_type_label": [],            # e.g. 'excluded' on an inclusion criterion
        "contradictory_verdict": [],       # direct positive/negative flip
        "parse_failure": [],
        "latency_outlier": [],
    }
    lat = sorted(records, key=lambda r: r["latency_s"])
    outliers = {r["annotation_id"] for r in lat[-3:]} if len(lat) > 6 else set()

    for r in records:
        gold_ev, pred_ev = set(r["gold_evidence_ids"]), set(r["pred_evidence_ids"])
        wrong_label = r["pred_label"] != r["gold_label"]
        ev_ok = pred_ev == gold_ev

        if wrong_label and ev_ok:
            buckets["wrong_label_evidence_ok"].append(r)
        if wrong_label and not ev_ok:
            buckets["wrong_label_wrong_evidence"].append(r)
        if gold_ev and not pred_ev:
            buckets["missing_evidence"].append(r)
        if not gold_ev and pred_ev:
            buckets["hallucinated_evidence"].append(r)
        legal = LABELS_FOR_TYPE.get(r["criterion_type"], ())
        if r["pred_label"] not in legal:
            buckets["cross_type_label"].append(r)
        if r["gold_label"] == "not enough information" and r["pred_label"] in (
            "included", "not included", "excluded", "not excluded"
        ):
            buckets["insufficient_info_failure"].append(r)
        opposite = {
            "included": "not included", "not included": "included",
            "excluded": "not excluded", "not excluded": "excluded",
        }.get(r["gold_label"])
        if r["pred_label"] == opposite:
            buckets["contradictory_verdict"].append(r)
        parse_failed = (not r["pred_label_valid"]) or any(
            "no label produced" in n or "no EVIDENCE" in n for n in r["parse_notes"]
        )
        if parse_failed:
            buckets["parse_failure"].append(r)
        if r["annotation_id"] in outliers:
            buckets["latency_outlier"].append(r)

    return {k: v for k, v in buckets.items() if v}


def run_variant_split(predictor: Predictor, variant: str, split: str,
                      limit: int = 0) -> dict:
    examples = read_split(split)
    if limit:
        examples = examples[:limit]
    t0 = time.time()
    records = [_prediction_record(ex, predictor.predict(ex)) for ex in examples]
    wall = time.time() - t0

    gold = [r["gold_label"] for r in records]
    pred = [r["pred_label"] for r in records]
    types = [r["criterion_type"] for r in records]
    labels_all = list(LABELS)

    overall = classification_metrics(gold, pred, labels_all)
    by_type = per_type_metrics(gold, pred, types, LABELS_FOR_TYPE)
    ev_all = evidence_metrics(
        [r["gold_evidence_ids"] for r in records],
        [r["pred_evidence_ids"] for r in records],
    )
    ev_by_type = {
        ctype: evidence_metrics(
            [r["gold_evidence_ids"] for r in records if r["criterion_type"] == ctype],
            [r["pred_evidence_ids"] for r in records if r["criterion_type"] == ctype],
        )
        for ctype in ("inclusion", "exclusion")
    }
    invalid_rate = sum(1 for r in records if not r["pred_label_valid"]) / len(records)

    # ---- persist --------------------------------------------------------
    stem = f"{variant}_{split}"
    RESULTS_DIR.mkdir(exist_ok=True)
    with (RESULTS_DIR / f"preds_{stem}.jsonl").open("w") as fh:
        for r in records:
            fh.write(json.dumps(r) + "\n")

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    _confusion_png(overall["confusion_matrix"], overall["labels"],
                   FIG_DIR / f"cm_{stem}_overall.png", f"{variant}/{split} overall")
    for ctype, m in by_type.items():
        if m:
            _confusion_png(m["confusion_matrix"], m["labels"],
                           FIG_DIR / f"cm_{stem}_{ctype}.png",
                           f"{variant}/{split} {ctype}")

    errors = categorise_errors(records)
    err_path = RESULTS_DIR / f"errors_{stem}.json"
    err_path.write_text(json.dumps(_slim_errors(errors), indent=2))

    summary = {
        "variant": variant,
        "split": split,
        "model": TARGET_MODEL,
        "adapter": str(ADAPTER_DIR) if variant == "tuned" else None,
        "adapter_loaded": predictor.adapter_loaded,
        "decoding": {
            "strategy": "greedy, cache-aware, trie-constrained label slot",
            "assistant_prefix": "LABEL:",
            "max_new_tokens": predictor.max_new_tokens,
            "deterministic": True,
        },
        "n": len(records),
        "wall_clock_s": round(wall, 1),
        "invalid_label_rate": round(invalid_rate, 4),
        "classification_overall": overall,
        "classification_by_type": by_type,
        "evidence_overall": ev_all,
        "evidence_by_type": ev_by_type,
        "latency_s": {
            "mean": round(sum(r["latency_s"] for r in records) / len(records), 3),
            "max": round(max(r["latency_s"] for r in records), 3),
            "p95": round(sorted(r["latency_s"] for r in records)[int(0.95 * len(records)) - 1], 3),
        },
        "environment": {
            "device": str(getattr(predictor, "device", "cpu")),
            "torch": torch.__version__,
            "python": platform.python_version(),
        },
        "errors_file": err_path.name,
        "error_category_counts": {k: len(v) for k, v in errors.items()},
    }
    out = RESULTS_DIR / f"eval_{stem}.json"
    out.write_text(json.dumps(summary, indent=2))
    print(f"[{stem}] acc={summary['classification_overall']['accuracy']} "
          f"macroF1={summary['classification_overall']['macro_f1']} "
          f"evF1(micro)={summary['evidence_overall']['micro']['f1']} -> {out.name}")
    return summary


def _slim_errors(buckets: dict[str, list[dict]], per_bucket: int = 4) -> dict:
    slim: dict[str, list] = {}
    for cat, items in buckets.items():
        slim[cat] = [
            {
                k: r[k] for k in (
                    "annotation_id", "patient_id", "criterion_type", "criterion_text",
                    "gold_label", "pred_label", "gold_evidence_ids", "pred_evidence_ids",
                    "raw_output", "parse_notes",
                )
            }
            for r in items[:per_bucket]
        ]
    return slim


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=["base", "tuned", "both"], default="both")
    ap.add_argument("--splits", default="validation,test")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--max-new-tokens", type=int, default=None)
    args = ap.parse_args()

    variants = ["base", "tuned"] if args.variant == "both" else [args.variant]
    dtype = getattr(torch, args.dtype)
    for variant in variants:
        try:
            predictor = Predictor(variant, dtype=dtype, device=args.device)
        except FileNotFoundError as e:
            print(f"skipping {variant}: {e}")
            continue
        if args.max_new_tokens:
            predictor.max_new_tokens = args.max_new_tokens
        for split in args.splits.split(","):
            run_variant_split(predictor, variant, split.strip(), args.limit)


if __name__ == "__main__":
    main()
