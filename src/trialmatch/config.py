"""Central configuration for the TrialMatch project.

All paths are resolved relative to the repository root so that every script can
be launched from anywhere.  Values that materially affect results (split seed,
model ids, decoding budgets) live here and are imported everywhere else, so a
single change is reflected in training, evaluation and the web application.
"""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
SPLIT_DIR = DATA_DIR / "splits"
RESULTS_DIR = ROOT / "results"
ARTIFACT_DIR = ROOT / "artifacts"
ADAPTER_DIR = ARTIFACT_DIR / "lora_smollm2-360m"
LOG_DIR = RESULTS_DIR / "logs"

for _d in (RAW_DIR, PROCESSED_DIR, SPLIT_DIR, RESULTS_DIR, ARTIFACT_DIR, LOG_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------------------------
# Dataset / split (mandated by the assignment)
# --------------------------------------------------------------------------
HF_DATASET = "ncbi/TrialGPT-Criterion-Annotations"
SPLIT_SEED = 42
N_TRAIN_PATIENTS = 37
N_VAL_PATIENTS = 8
N_TEST_PATIENTS = 8

# --------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------
# Target model: open-weight, locally executable decoder-only Transformer.
# 360M parameters, Apache-2.0, 2048-token context, fp32 fits in ~1.5 GB RAM,
# which is what makes the full pipeline (training + evaluation + live app)
# executable on the 8 GB / CPU-only machine this work was developed on.
TARGET_MODEL = os.environ.get("TRIALMATCH_TARGET", "HuggingFaceTB/SmolLM2-360M-Instruct")
# Draft model: a genuinely smaller checkpoint from the same family. Both
# checkpoints ship the identical 49,152-token tokenizer, so draft tokens are
# valid target tokens without any re-encoding (assignment Task 5 requirement).
DRAFT_MODEL = os.environ.get("TRIALMATCH_DRAFT", "HuggingFaceTB/SmolLM2-135M-Instruct")
# Optional: self-speculative draft (first N layers of the target). Set
# TRIALMATCH_DRAFT_LAYERS>0 to use it instead of DRAFT_MODEL.
DRAFT_LAYERS = int(os.environ.get("TRIALMATCH_DRAFT_LAYERS", "0"))

# --------------------------------------------------------------------------
# Label space.  The six expert categories are preserved verbatim; inclusion and
# exclusion criteria use disjoint "positive" labels, which is why the decoder
# restricts the label vocabulary per criterion type.
# --------------------------------------------------------------------------
LABELS = (
    "included",
    "not included",
    "excluded",
    "not excluded",
    "not enough information",
    "not applicable",
)
LABELS_FOR_TYPE = {
    "inclusion": ("included", "not included", "not enough information", "not applicable"),
    "exclusion": ("excluded", "not excluded", "not enough information", "not applicable"),
}
# Aggregation rules used for the *derived* patient/trial verdict (never trained,
# never scored against a gold patient-level label - see report Section 5.4).
NEGATIVE_LABEL = {
    "included": "not included",
    "not included": "included",
    "excluded": "not excluded",
    "not excluded": "excluded",
    "not enough information": "not enough information",
    "not applicable": "not applicable",
}

# --------------------------------------------------------------------------
# Prompting / training budget
# --------------------------------------------------------------------------
MAX_SEQ_LEN = 1024          # prompt + target; covers every note in the dataset
MAX_NEW_TOKENS = 24         # "LABEL: ...\nEVIDENCE: [..]" is <= 20 tokens
PROMPT_VERSION = "v3-numbered-note"

SYSTEM_PROMPT = (
    "You screen clinical trial eligibility criteria against a patient note and "
    "cite the note sentences that support your decision."
)
