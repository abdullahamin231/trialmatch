"""Production prediction pipeline (used by evaluation and the web app).

Decoding is cache-aware and constrained:
  * the assistant reply is prefilled with "LABEL:" so the label slot is a
    single controlled decision (same for base and fine-tuned checkpoints);
  * the label slot is restricted by a character trie over the labels that are
    legal for the criterion type (no cross-type labels, no invented ones);
  * the evidence slot is free-form and parsed back to sentence ids.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import torch

from .config import (
    ADAPTER_DIR,
    LABELS_FOR_TYPE,
    MAX_NEW_TOKENS,
    TARGET_MODEL,
)
from .data import Example, parse_note
from .kv_cache import CachedDecoder, KVCache
from .prompts import build_messages
from .schema import ConstrainedLabelDecoder, parse_prediction

ASSISTANT_PREFIX = "LABEL:"


@dataclass
class Prediction:
    annotation_id: int
    patient_id: str
    criterion_type: str
    label: str
    label_valid: bool
    evidence_ids: list[int]
    raw: str
    notes: list[str]
    latency_s: float
    n_prompt_tokens: int
    n_generated: int


class Predictor:
    """Loads one checkpoint variant and answers criterion-level questions."""

    def __init__(self, variant: str = "tuned", dtype=torch.float32, device: str = "cpu"):
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.variant = variant
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(TARGET_MODEL)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=dtype,
                                                     attn_implementation="eager")
        self.adapter_loaded = False
        if variant == "tuned":
            if not (ADAPTER_DIR / "adapter_config.json").exists():
                raise FileNotFoundError(
                    f"No adapter at {ADAPTER_DIR}. Train it first: "
                    "python -m trialmatch.train_lora"
                )
            from peft import PeftModel

            model = PeftModel.from_pretrained(model, str(ADAPTER_DIR))
            model = model.merge_and_unload()   # single fused checkpoint at inference
            self.adapter_loaded = True
        model.eval()
        self.model = model.to(device)
        self.device = device
        self.decoder = CachedDecoder(model, dtype=dtype)
        self.label_decoders: dict[str, ConstrainedLabelDecoder] = {}
        self.model_params = sum(p.numel() for p in model.parameters())
        self.max_new_tokens = MAX_NEW_TOKENS

    # ------------------------------------------------------------------
    def _label_decoder(self, criterion_type: str) -> ConstrainedLabelDecoder:
        labels = LABELS_FOR_TYPE.get(criterion_type, LABELS_FOR_TYPE["inclusion"])
        key = "|".join(labels)
        if key not in self.label_decoders:
            self.label_decoders[key] = ConstrainedLabelDecoder(
                self.tokenizer, labels, model_id=TARGET_MODEL
            )
        return self.label_decoders[key]

    def _prompt_ids(self, ex: Example) -> torch.Tensor:
        msgs = build_messages(ex)
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ) + ASSISTANT_PREFIX
        return torch.tensor(
            [self.tokenizer(text, add_special_tokens=False).input_ids], device=self.device
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, ex: Example, max_new_tokens: int | None = None) -> Prediction:
        max_new_tokens = max_new_tokens or self.max_new_tokens
        t0 = time.perf_counter()
        ids = self._prompt_ids(ex)
        cache = KVCache(self.decoder.n_layers)
        logits = self.decoder.forward(ids, cache)[0, -1]

        cd = self._label_decoder(ex.criterion_type)
        cd.reset()
        cd.begin()
        out_ids: list[int] = []
        phase_label = True
        for _ in range(max_new_tokens):
            step_logits = cd.mask_scores(logits.unsqueeze(0))[0] if phase_label else logits
            tok = int(step_logits.argmax())
            out_ids.append(tok)
            cd.advance(tok)
            if phase_label and not cd.active:
                phase_label = False
            text = self.tokenizer.decode(out_ids, skip_special_tokens=True)
            if tok == self.tokenizer.eos_token_id or (not phase_label and text.rstrip().endswith("]")):
                break
            logits = self.decoder.forward(
                torch.tensor([[tok]], device=self.device), cache
            )[0, -1]

        raw = ASSISTANT_PREFIX + self.tokenizer.decode(out_ids, skip_special_tokens=True)
        n_sent = len(parse_note(ex.note))
        parsed = parse_prediction(raw, ex.criterion_type, max_sentence_id=n_sent - 1)
        return Prediction(
            annotation_id=ex.annotation_id,
            patient_id=ex.patient_id,
            criterion_type=ex.criterion_type,
            label=parsed.label,
            label_valid=parsed.label_valid,
            evidence_ids=parsed.evidence_ids,
            raw=raw,
            notes=parsed.notes,
            latency_s=time.perf_counter() - t0,
            n_prompt_tokens=int(ids.shape[1]),
            n_generated=len(out_ids),
        )

    def predict_many(self, examples: list[Example]) -> list[Prediction]:
        return [self.predict(ex) for ex in examples]
