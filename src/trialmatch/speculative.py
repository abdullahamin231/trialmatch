"""Greedy speculative decoding (assignment Task 5, rubric F1-F3).

Implements the Leviathan et al. (2023) greedy variant from scratch:

  1. the draft model proposes ``gamma`` candidate tokens autoregressively
     from its own KV cache;
  2. the target model verifies all ``gamma`` candidates in a single forward
     pass over its own KV cache (one position per candidate);
  3. left to right, a candidate is accepted iff it equals the target's
     argmax at that position; the first mismatch rejects the remaining
     candidates and the target's argmax is emitted instead (the
     *correction* token);
  4. when every candidate is accepted, the target's argmax after the last
     candidate is emitted as a free *bonus* token.

Because acceptance compares against the target's greedy argmax and every
correction IS the target's argmax, the emitted sequence is identical to
plain greedy decoding of the target (verified in ``trialmatch.correctness``).

Cache synchronisation (the part most implementations get wrong).

Round-start invariant::

    both caches hold exactly  prompt + emitted
    pending  = target argmax for the token after emitted
    d_seed   = draft  argmax for the token after emitted

  * verification leaves the target cache holding the drafted continuation;
    after a rejection it is truncated back to the last accepted token;
  * the draft cache also holds the (possibly rejected) draft suffix and is
    truncated to the same length;
  * the final emitted token of the round (correction or bonus) is fed once
    to each model: for the target this single step produces the next
    ``pending``, for the draft the next ``d_seed`` - no work is wasted;
  * at every round boundary the invariant is restored.

The label trie constraint is applied to the *target's* argmax during
verification, so speculative screening emits the same constrained labels
as the sequential pipeline.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import torch

from .kv_cache import CachedDecoder, KVCache
from .schema import ConstrainedLabelDecoder


@dataclass
class SpecStats:
    rounds: int = 0
    drafted: int = 0
    accepted: int = 0
    bonus_tokens: int = 0
    corrections: int = 0
    draft_positions: int = 0
    target_positions: int = 0
    prefill_s: float = 0.0
    decode_s: float = 0.0
    round_log: list = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted / self.drafted if self.drafted else 0.0

    def as_dict(self) -> dict:
        return {
            "rounds": self.rounds,
            "drafted": self.drafted,
            "accepted": self.accepted,
            "acceptance_rate": round(self.acceptance_rate, 4),
            "bonus_tokens": self.bonus_tokens,
            "corrections": self.corrections,
            "draft_positions": self.draft_positions,
            "target_positions": self.target_positions,
            "prefill_s": round(self.prefill_s, 4),
            "decode_s": round(self.decode_s, 4),
            "round_log": self.round_log,
        }


class SpeculativeDecoder:
    """Greedy speculative decoding over two ``CachedDecoder``s."""

    def __init__(self, target: CachedDecoder, draft: CachedDecoder,
                 gamma: int = 4, device: str = "cpu"):
        assert target.lm_head.weight.shape[0] == draft.lm_head.weight.shape[0], (
            "draft/target vocab mismatch - tokenizers must be identical"
        )
        self.target = target
        self.draft = draft
        self.gamma = gamma
        self.device = device
        self.stats = SpecStats()

    # ------------------------------------------------------------------
    def _draft_gamma(self, seed_logits: torch.Tensor, dcache: KVCache,
                     gamma: int) -> list[int]:
        """Draft ``gamma`` tokens greedily.

        ``seed_logits`` is the draft distribution at the cache end (saved
        during the previous round's sync).  Every drafted token is fed back
        once, so afterwards the draft cache ends at d_gamma.
        """
        toks: list[int] = []
        logits = seed_logits
        for _ in range(gamma):
            tok = int(logits.argmax())
            toks.append(tok)
            self.stats.draft_positions += 1
            logits = self.draft.forward(
                torch.tensor([[tok]], device=self.device), dcache
            )[0, -1]
        return toks

    # ------------------------------------------------------------------
    @staticmethod
    def _masked_argmax(logits, cd: ConstrainedLabelDecoder | None) -> int:
        if cd is not None and cd.active:
            logits = cd.mask_scores(logits.unsqueeze(0))[0]
        return int(logits.argmax())

    # ------------------------------------------------------------------
    @torch.no_grad()
    def greedy(self, prompt_ids: torch.Tensor, max_new_tokens: int,
               eos_id: int | None = None,
               label_decoder: ConstrainedLabelDecoder | None = None,
               stop_text_fn=None) -> tuple[list[int], SpecStats]:
        self.stats = SpecStats()
        cd = label_decoder
        if cd is not None:
            cd.reset()
            cd.begin()

        tcache = KVCache(self.target.n_layers)
        dcache = KVCache(self.draft.n_layers)
        one = lambda tok: torch.tensor([[tok]], device=self.device)

        # ---- prefill both models on the prompt ---------------------------
        t0 = time.perf_counter()
        t_logits = self.target.forward(prompt_ids, tcache)[0, -1]
        d_logits = self.draft.forward(prompt_ids, dcache)[0, -1]
        self.stats.target_positions += int(prompt_ids.shape[1])
        self.stats.draft_positions += int(prompt_ids.shape[1])
        # pending = the *constrained* target argmax, exactly like the
        # sequential pipeline's first decoding step
        pending = self._masked_argmax(t_logits, cd)
        self.stats.prefill_s = time.perf_counter() - t0

        emitted: list[int] = []
        t1 = time.perf_counter()

        while len(emitted) < max_new_tokens:
            self.stats.rounds += 1
            gamma = min(self.gamma, max_new_tokens - len(emitted))

            # --- 1) draft gamma candidates --------------------------------
            d_toks = self._draft_gamma(d_logits, dcache, gamma)
            self.stats.drafted += gamma

            # --- 2) verify all candidates with one target pass ------------
            v_logits = self.target.forward(
                torch.tensor([d_toks], device=self.device), tcache
            )[0]                                        # (gamma, V)
            self.stats.target_positions += gamma

            # Masked argmax per position.  The trie state that applies at
            # candidate i is simulated over the accepted prefix d_1..d_{i-1}
            # and restored afterwards; the real trie is advanced with the
            # finally emitted tokens in step 4.
            saved = (cd.node, cd.active, cd.text) if cd is not None else None
            us: list[int] = []
            for i in range(gamma):
                if cd is not None and cd.active:
                    masked = cd.mask_scores(v_logits[i].unsqueeze(0))[0]
                    us.append(int(masked.argmax()))
                    cd.advance(d_toks[i])
                else:
                    us.append(int(v_logits[i].argmax()))
            if saved is not None:
                cd.node, cd.active, cd.text = saved

            # --- 3) accept / reject ---------------------------------------
            t = pending
            k = 0
            for i in range(gamma):
                if d_toks[i] == t:
                    k += 1
                    t = us[i]
                else:
                    break
            full = k == gamma
            emitted.extend(d_toks[:k])
            emitted.append(t)          # bonus token if full, else the correction
            if full:
                self.stats.bonus_tokens += 1
            else:
                self.stats.corrections += 1
            self.stats.accepted += k
            self.stats.round_log.append({"gamma": gamma, "accepted": k, "full": full})

            # --- 4) resynchronise both caches to prompt+emitted ------------
            # Both caches hold prompt+old_emitted+gamma drafted positions.
            # Only the first k of those are emitted; roll both back to the
            # common prefix, then feed the final emitted token once.
            cut = tcache.seq_len - (gamma - k)
            tcache.truncate(cut)
            dcache.truncate(cut)
            last = emitted[-1]
            t_logits = self.target.forward(one(last), tcache)[0, -1]
            pending = self._masked_argmax(t_logits, cd)
            d_logits = self.draft.forward(one(last), dcache)[0, -1]
            self.stats.target_positions += 1
            self.stats.draft_positions += 1

            # advance the real trie over this round's emitted tokens
            if cd is not None:
                for tok in d_toks[:k] + [last]:
                    cd.advance(tok)

            # round-start invariant for the next iteration
            assert tcache.seq_len == prompt_ids.shape[1] + len(emitted), (
                "cache out of sync: target cache does not hold prompt+emitted"
            )
            assert dcache.seq_len == tcache.seq_len

            if eos_id is not None and emitted[-1] == eos_id:
                break
            if stop_text_fn is not None and stop_text_fn(emitted):
                break

        emitted = emitted[:max_new_tokens]   # a bonus token may exceed the budget
        self.stats.decode_s = time.perf_counter() - t1
        return emitted, self.stats


# ---------------------------------------------------------------------------
# Predictor-compatible wrapper so the app can swap decoding strategies
# ---------------------------------------------------------------------------
class SpeculativePredictor:
    """Same contract as ``Predictor.predict`` but decoding speculatively."""

    def __init__(self, spec: SpeculativeDecoder, tokenizer, variant: str = "tuned",
                 max_new_tokens: int = 24, device: str = "cpu"):
        from .config import TARGET_MODEL

        self._model_id = TARGET_MODEL
        self.spec = spec
        self.tokenizer = tokenizer
        self.variant = variant
        self.max_new_tokens = max_new_tokens
        self.device = device
        self.adapter_loaded = variant == "tuned"

    def _prompt_ids(self, ex):
        from .pipeline import ASSISTANT_PREFIX
        from .prompts import build_messages

        msgs = build_messages(ex)
        text = self.tokenizer.apply_chat_template(
            msgs, tokenize=False, add_generation_prompt=True
        ) + ASSISTANT_PREFIX
        return torch.tensor(
            [self.tokenizer(text, add_special_tokens=False).input_ids],
            device=self.device,
        )

    @torch.no_grad()
    def predict(self, ex, max_new_tokens: int | None = None):
        import time as _time

        from .config import LABELS_FOR_TYPE
        from .data import parse_note
        from .pipeline import ASSISTANT_PREFIX, Prediction
        from .schema import parse_prediction

        max_new_tokens = max_new_tokens or self.max_new_tokens
        t0 = _time.perf_counter()
        ids = self._prompt_ids(ex)
        cd = ConstrainedLabelDecoder(
            self.tokenizer,
            LABELS_FOR_TYPE.get(ex.criterion_type, LABELS_FOR_TYPE["inclusion"]),
            model_id=self._model_id,
        )
        out_ids, stats = self.spec.greedy(
            ids, max_new_tokens, eos_id=self.tokenizer.eos_token_id, label_decoder=cd
        )
        raw = ASSISTANT_PREFIX + self.tokenizer.decode(out_ids, skip_special_tokens=True)
        parsed = parse_prediction(raw, ex.criterion_type,
                                  max_sentence_id=len(parse_note(ex.note)) - 1)
        return Prediction(
            annotation_id=ex.annotation_id,
            patient_id=ex.patient_id,
            criterion_type=ex.criterion_type,
            label=parsed.label,
            label_valid=parsed.label_valid,
            evidence_ids=parsed.evidence_ids,
            raw=raw,
            notes=parsed.notes + [f"spec decoding: {stats.as_dict()}"],
            latency_s=_time.perf_counter() - t0,
            n_prompt_tokens=int(ids.shape[1]),
            n_generated=len(out_ids),
        )
