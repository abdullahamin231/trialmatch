"""Dataset loading, note parsing and the mandated patient-grouped split.

Run as a script to (re)build the split manifest and the processed JSONL files:

    python -m trialmatch.data --prepare
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from dataclasses import dataclass, asdict
from typing import Iterable

import numpy as np

from .config import (
    HF_DATASET,
    N_TEST_PATIENTS,
    N_TRAIN_PATIENTS,
    N_VAL_PATIENTS,
    PROCESSED_DIR,
    RAW_DIR,
    SPLIT_DIR,
    SPLIT_SEED,
)

# Columns that contain GPT-4 predictions/explanations.  They must never reach a
# model input or a metric; they are dropped as soon as the parquet is read.
LEAKY_COLUMNS = ("gpt4_explanation", "gpt4_sentences", "gpt4_eligibility")

SENT_RE = re.compile(r"^\s*(\d+)\.\s*(.*)$")


@dataclass
class Example:
    annotation_id: int
    patient_id: str
    trial_id: str
    trial_title: str
    criterion_type: str
    criterion_text: str
    label: str
    evidence_ids: list[int]
    note: str
    split: str = ""

    @property
    def sentences(self) -> list[tuple[int, str]]:
        return parse_note(self.note)

    def to_json(self) -> dict:
        return asdict(self)


def parse_note(note: str) -> list[tuple[int, str]]:
    """Split a TrialGPT note into its native, pre-numbered sentences.

    The dataset already carries sentence indices in the note text ("0. ..."),
    and the expert evidence annotations refer to exactly those indices, so we
    preserve them instead of re-segmenting (re-segmentation would silently
    invalidate every gold sentence id).
    """
    out: list[tuple[int, str]] = []
    for line in note.splitlines():
        m = SENT_RE.match(line)
        if m:
            out.append((int(m.group(1)), m.group(2).strip()))
    return out


def parse_sentence_ids(raw: str) -> list[int]:
    if raw is None:
        return []
    raw = str(raw).strip()
    if not raw or raw in {"[]", "nan", "None"}:
        return []
    try:
        val = json.loads(raw)
    except json.JSONDecodeError:
        return []
    if isinstance(val, int):
        return [val]
    return sorted({int(v) for v in val})


def load_annotations():
    import pandas as pd

    parquet = RAW_DIR / "criterion_annotations.parquet"
    if not parquet.exists():
        from datasets import load_dataset

        ds = load_dataset(HF_DATASET, split="train")
        df = ds.to_pandas()
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        df.to_parquet(parquet)
    else:
        df = pd.read_parquet(parquet)
    # Hard guarantee: model-generated fields never enter the pipeline.
    df = df.drop(columns=[c for c in LEAKY_COLUMNS if c in df.columns])
    return df


def make_split(patient_ids: Iterable[str]) -> dict[str, list[str]]:
    """37/8/8 patient-grouped split with NumPy default_rng(42)."""
    ids = np.array(sorted(set(patient_ids)))
    rng = np.random.default_rng(SPLIT_SEED)
    rng.shuffle(ids)
    tr = N_TRAIN_PATIENTS
    va = tr + N_VAL_PATIENTS
    return {
        "train": [str(x) for x in ids[:tr]],
        "validation": [str(x) for x in ids[tr:va]],
        "test": [str(x) for x in ids[va:]],
    }


def build_examples(df) -> list[Example]:
    examples: list[Example] = []
    for row in df.itertuples(index=False):
        if not isinstance(row.criterion_text, str) or not row.criterion_text.strip():
            continue
        label = str(row.expert_eligibility).strip()
        examples.append(
            Example(
                annotation_id=int(row.annotation_id),
                patient_id=str(row.patient_id),
                trial_id=str(row.trial_id),
                trial_title=str(row.trial_title),
                criterion_type=str(row.criterion_type).strip().lower(),
                criterion_text=" ".join(row.criterion_text.split()),
                label=label,
                evidence_ids=parse_sentence_ids(row.expert_sentences),
                note=row.note,
            )
        )
    return examples


def prepare() -> dict:
    df = load_annotations()
    examples = build_examples(df)
    split = make_split(df.patient_id.unique())

    patient_to_split = {p: s for s, ps in split.items() for p in ps}
    for ex in examples:
        ex.split = patient_to_split[ex.patient_id]

    # ---- leakage checks -------------------------------------------------
    seen: dict[str, str] = {}
    for ex in examples:
        prev = seen.setdefault(ex.patient_id, ex.split)
        assert prev == ex.split, f"patient {ex.patient_id} spans two splits"
    assert set(seen.values()) <= {"train", "validation", "test"}

    manifest = {
        "dataset": HF_DATASET,
        "seed": SPLIT_SEED,
        "numpy_generator": "numpy.random.default_rng(42)",
        "rule": (
            "sort unique patient_id -> default_rng(42).shuffle -> "
            f"first {N_TRAIN_PATIENTS} train, next {N_VAL_PATIENTS} validation, "
            f"remaining {N_TEST_PATIENTS} test"
        ),
        "patients": split,
        "counts": {
            s: {
                "patients": len(split[s]),
                "annotations": sum(1 for e in examples if e.split == s),
                "inclusion": sum(1 for e in examples if e.split == s and e.criterion_type == "inclusion"),
                "exclusion": sum(1 for e in examples if e.split == s and e.criterion_type == "exclusion"),
            }
            for s in ("train", "validation", "test")
        },
        "annotation_id_sha256": {
            s: hashlib.sha256(
                ",".join(str(e.annotation_id) for e in sorted(
                    (x for x in examples if x.split == s), key=lambda x: x.annotation_id
                )).encode()
            ).hexdigest()
            for s in ("train", "validation", "test")
        },
        "leakage_check": "passed (no patient_id appears in more than one split)",
    }

    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    (SPLIT_DIR / "split_manifest.json").write_text(json.dumps(manifest, indent=2))
    for s in ("train", "validation", "test"):
        path = PROCESSED_DIR / f"{s}.jsonl"
        with path.open("w") as fh:
            for ex in examples:
                if ex.split == s:
                    fh.write(json.dumps(ex.to_json()) + "\n")
    print(json.dumps(manifest["counts"], indent=2))
    return manifest


def read_split(split: str) -> list[Example]:
    path = PROCESSED_DIR / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `python -m trialmatch.data --prepare`")
    with path.open() as fh:
        return [Example(**json.loads(line)) for line in fh]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    if args.prepare:
        prepare()


if __name__ == "__main__":
    main()
