"""Explicit KV-cache implementation and a cache-aware decoder for Qwen2.

The point of this module is *not* to call ``generate(use_cache=True)``.  It
re-implements the decoder loop so that the cache is an object this code owns,
appends to and truncates, and so that prefill (S > 1) and decode (S == 1) are
separate, measurable steps.

Key details that a cache implementation must get right:
  * positions: the first generated token is at absolute position S (prefill
    consumed positions 0..S-1); rotary embeddings must use that offset;
  * masking: during prefill each query sees keys <= its own position; during
    decode the single query may see every cached key;
  * cache growth: one (k, v) pair of shape (B, H_kv, 1, D) per layer per token;
  * numerical drift: prefill and decode use different matmul shapes, so bf16
    reductions can differ by ~1e-2 in logits while greedy tokens stay equal.
"""
from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import torch

from .attention import build_causal_mask, repeat_kv, tiled_attention


@dataclass
class KVCache:
    """Per-layer key/value store with explicit growth and truncation."""

    n_layers: int
    k: list[torch.Tensor | None] = field(default_factory=list)
    v: list[torch.Tensor | None] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.k:
            self.k = [None] * self.n_layers
            self.v = [None] * self.n_layers

    @property
    def seq_len(self) -> int:
        for t in self.k:
            if t is not None:
                return t.shape[2]
        return 0

    def append(self, layer: int, k_new: torch.Tensor, v_new: torch.Tensor) -> None:
        if self.k[layer] is None:
            self.k[layer], self.v[layer] = k_new, v_new
        else:
            self.k[layer] = torch.cat([self.k[layer], k_new], dim=2)
            self.v[layer] = torch.cat([self.v[layer], v_new], dim=2)

    def get(self, layer: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.k[layer], self.v[layer]

    def truncate(self, length: int) -> None:
        """Roll the cache back to ``length`` tokens (needed on rejection)."""
        for i in range(self.n_layers):
            if self.k[i] is not None:
                self.k[i] = self.k[i][:, :, :length].contiguous()
                self.v[i] = self.v[i][:, :, :length].contiguous()

    def clone(self) -> "KVCache":
        c = KVCache(self.n_layers)
        for i in range(self.n_layers):
            if self.k[i] is not None:
                c.k[i] = self.k[i].clone()
                c.v[i] = self.v[i].clone()
        return c

    def nbytes(self) -> int:
        return sum(t.numel() * t.element_size() for t in self.k + self.v if t is not None)


class CachedDecoder:
    """Cache-aware greedy decoder driving HF *layers* manually.

    Only the layer modules (projections, norms, rotary embedding) are reused
    from the checkpoint; the forward pass, the attention, the mask and the
    cache update are implemented here.
    """

    def __init__(self, model, *, use_tiled: bool = False, block_size: int = 64,
                 dtype: torch.dtype = torch.float32):
        self.model = model.to(dtype)
        self.model.eval()
        self.dtype = dtype
        self.use_tiled = use_tiled
        self.block_size = block_size
        cfg = model.config
        self.n_layers = cfg.num_hidden_layers
        self.n_heads = cfg.num_attention_heads
        self.n_kv = getattr(cfg, "num_key_value_heads", cfg.num_attention_heads)
        self.head_dim = getattr(cfg, "head_dim", cfg.hidden_size // self.n_heads)
        self.scale = 1.0 / math.sqrt(self.head_dim)
        self.layers = model.model.layers
        self.embed = model.model.embed_tokens
        self.norm = model.model.norm
        self.lm_head = model.lm_head
        self.rotary = model.model.rotary_emb

    # -- one forward pass over `input_ids` continuing an existing cache -----
    def forward(self, input_ids: torch.Tensor, cache: KVCache, only_last: bool = False):
        b, s = input_ids.shape
        offset = cache.seq_len
        device = input_ids.device
        h = self.embed(input_ids)
        position_ids = torch.arange(offset, offset + s, device=device).unsqueeze(0).expand(b, -1)
        cos, sin = self.rotary(h, position_ids)

        if not cache.k:
            cache.k = [None] * self.n_layers
            cache.v = [None] * self.n_layers

        for idx, layer in enumerate(self.layers):
            attn = layer.self_attn
            x = layer.input_layernorm(h)
            q = attn.q_proj(x).view(b, s, self.n_heads, self.head_dim).transpose(1, 2)
            k_new = attn.k_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
            v_new = attn.v_proj(x).view(b, s, self.n_kv, self.head_dim).transpose(1, 2)
            q, k_new = _apply_rope(q, k_new, cos, sin)

            cache.append(idx, k_new, v_new)
            k_all, v_all = cache.get(idx)
            k_rep = repeat_kv(k_all, self.n_heads // self.n_kv)
            v_rep = repeat_kv(v_all, self.n_heads // self.n_kv)

            mask = None
            if s > 1:
                # queries sit at absolute positions offset..offset+s-1 and may
                # attend to cached keys up to their own position; passing the
                # offset is what makes multi-token verification passes
                # (speculative decoding) correct
                mask = build_causal_mask(s, k_all.shape[2], device=device,
                                         dtype=self.dtype, offset=offset)
            if self.use_tiled:
                o = tiled_attention(q, k_rep, v_rep, block_q=self.block_size,
                                    block_k=self.block_size, causal=(s > 1),
                                    attn_mask=None, scale=self.scale)
            else:
                from .attention import reference_attention

                o = reference_attention(q, k_rep, v_rep, causal=(s > 1),
                                        attn_mask=mask, scale=self.scale)
            o = o.transpose(1, 2).contiguous().view(b, s, self.n_heads * self.head_dim)
            h = h + attn.o_proj(o)
            h = h + layer.mlp(layer.post_attention_layernorm(h))

        h = self.norm(h)
        return self.lm_head(h)

    # -- uncached reference path ------------------------------------------
    def forward_uncached(self, input_ids: torch.Tensor):
        """Full recomputation from scratch: the reference path for Task 3."""
        cache = KVCache(self.n_layers)
        return self.forward(input_ids, cache)

    # -- greedy generation --------------------------------------------------
    @torch.no_grad()
    def greedy(self, input_ids: torch.Tensor, max_new_tokens: int, eos_id: int | None = None,
               use_cache: bool = True, stop_fn=None):
        out = input_ids
        tokens: list[int] = []
        prefill_s = 0.0
        decode_s = 0.0
        cache = KVCache(self.n_layers)
        t0 = time.perf_counter()
        if use_cache:
            logits = self.forward(out, cache)
            next_tok = int(logits[0, -1].argmax())
            prefill_s = time.perf_counter() - t0
            for _ in range(max_new_tokens):
                tokens.append(next_tok)
                if eos_id is not None and next_tok == eos_id:
                    break
                if stop_fn is not None and stop_fn(tokens):
                    break
                t1 = time.perf_counter()
                logits = self.forward(torch.tensor([[next_tok]], device=out.device), cache)
                next_tok = int(logits[0, -1].argmax())
                decode_s += time.perf_counter() - t1
            return tokens, {"prefill_s": prefill_s, "decode_s": decode_s,
                            "cache_bytes": cache.nbytes()}
        # uncached: recompute the whole prefix for every token
        logits = self.forward_uncached(out)
        next_tok = int(logits[0, -1].argmax())
        prefill_s = time.perf_counter() - t0
        for _ in range(max_new_tokens):
            tokens.append(next_tok)
            if eos_id is not None and next_tok == eos_id:
                break
            if stop_fn is not None and stop_fn(tokens):
                break
            t1 = time.perf_counter()
            out = torch.cat([out, torch.tensor([[next_tok]], device=out.device)], dim=1)
            logits = self.forward_uncached(out)
            next_tok = int(logits[0, -1].argmax())
            decode_s += time.perf_counter() - t1
        return tokens, {"prefill_s": prefill_s, "decode_s": decode_s, "cache_bytes": 0}


def _apply_rope(q, k, cos, sin):
    """Qwen2 rotary embedding on (B, H, S, D) tensors."""
    def rotate_half(x):
        x1 = x[..., : x.shape[-1] // 2]
        x2 = x[..., x.shape[-1] // 2 :]
        return torch.cat((-x2, x1), dim=-1)

    cos = cos.unsqueeze(1)   # (B,1,S,D)
    sin = sin.unsqueeze(1)
    q_out = q * cos + rotate_half(q) * sin
    k_out = k * cos + rotate_half(k) * sin
    return q_out, k_out


def load_draft(model, n_layers: int):
    """Self-speculative draft: the same checkpoint, first ``n_layers`` layers.

    Tokens, embeddings and lm_head are shared with the target, so the draft
    vocabulary is identical by construction (the assignment's compatibility
    requirement) and no extra download is needed.
    """
    import copy

    draft = copy.deepcopy(model)
    draft.model.layers = draft.model.layers[:n_layers]
    draft.config.num_hidden_layers = n_layers
    return draft
