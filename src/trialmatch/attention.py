"""Scaled dot-product attention: reference and tiled/online-softmax variants.

Both functions implement the same mathematical operation

    Attention(Q, K, V) = softmax(Q Kᵀ / sqrt(d) + mask) V

with optional grouped-query attention (GQA), optional causal masking and an
optional additive mask, so their outputs can be compared element-wise.

The tiled version never materialises the full S x S score matrix: it walks the
key/value sequence in blocks, keeps a running maximum ``m`` and running
normaliser ``l`` (the *online softmax* of Milakov & Gimelshein, 2018, and the
FlashAttention formulation of Dao et al., 2022), and rescales the accumulated
output by ``exp(m_old - m_new)`` whenever a new block raises the maximum.
"""
from __future__ import annotations

import math

import torch


def repeat_kv(x: torch.Tensor, n_rep: int) -> torch.Tensor:
    """Expand (B, H_kv, S, D) -> (B, H_kv * n_rep, S, D) like HF's repeat_kv."""
    if n_rep == 1:
        return x
    b, h, s, d = x.shape
    return x[:, :, None].expand(b, h, n_rep, s, d).reshape(b, h * n_rep, s, d)


def build_causal_mask(q_len: int, kv_len: int, device=None, dtype=torch.float32,
                      offset: int = 0) -> torch.Tensor:
    """Additive (0 / -inf) mask of shape (1, 1, q_len, kv_len).

    ``offset`` is the number of keys already in the cache: query i sits at
    absolute position ``offset + i`` and may attend to keys <= that position.
    """
    q_pos = torch.arange(offset, offset + q_len, device=device)[:, None]
    k_pos = torch.arange(kv_len, device=device)[None, :]
    allowed = k_pos <= q_pos
    mask = torch.zeros(q_len, kv_len, device=device, dtype=dtype)
    mask = mask.masked_fill(~allowed, float("-inf"))
    return mask[None, None]


def reference_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    causal: bool = True,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    return_weights: bool = False,
):
    """Materialised (quadratic-memory) reference implementation."""
    b, hq, sq, d = q.shape
    hkv = k.shape[1]
    if hkv != hq:
        k = repeat_kv(k, hq // hkv)
        v = repeat_kv(v, hq // hkv)
    scale = scale if scale is not None else 1.0 / math.sqrt(d)

    scores = torch.matmul(q, k.transpose(-2, -1)) * scale          # (B,H,Sq,Sk)
    mask = attn_mask
    if causal:
        cm = build_causal_mask(sq, k.shape[2], device=q.device, dtype=scores.dtype,
                               offset=k.shape[2] - sq)
        mask = cm if mask is None else mask + cm
    if mask is not None:
        scores = scores + mask
    # numerically stable softmax (max-subtraction happens inside torch)
    weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(weights, v)
    if return_weights:
        return out, weights
    return out


def tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    *,
    block_q: int = 64,
    block_k: int = 64,
    causal: bool = True,
    attn_mask: torch.Tensor | None = None,
    scale: float | None = None,
    return_stats: bool = False,
):
    """Blocked attention with online softmax (no S x S materialisation).

    Memory is O(Sq * block_k + block_q * block_k) per head instead of O(Sq*Sk).
    """
    b, hq, sq, d = q.shape
    hkv, sk = k.shape[1], k.shape[2]
    if hkv != hq:
        k = repeat_kv(k, hq // hkv)
        v = repeat_kv(v, hq // hkv)
    scale = scale if scale is not None else 1.0 / math.sqrt(d)
    offset = sk - sq

    out = torch.empty(b, hq, sq, d, dtype=q.dtype, device=q.device)
    n_blocks = 0
    for qi in range(0, sq, block_q):
        q_blk = q[:, :, qi : qi + block_q]                       # (B,H,bq,D)
        bq = q_blk.shape[2]
        # 1) running state
        m = torch.full((b, hq, bq, 1), float("-inf"), dtype=torch.float32, device=q.device)
        l = torch.zeros((b, hq, bq, 1), dtype=torch.float32, device=q.device)
        acc = torch.zeros((b, hq, bq, d), dtype=torch.float32, device=q.device)
        if causal:
            # keys beyond the last query of this block can never be attended
            k_end = min(sk, qi + bq + offset)
        else:
            k_end = sk
        for ki in range(0, k_end, block_k):
            k_blk = k[:, :, ki : ki + block_k]
            v_blk = v[:, :, ki : ki + block_k]
            s = torch.matmul(q_blk, k_blk.transpose(-2, -1)).float() * scale
            if causal:
                q_pos = torch.arange(offset + qi, offset + qi + bq, device=q.device)[:, None]
                k_pos = torch.arange(ki, ki + k_blk.shape[2], device=q.device)[None, :]
                s = s.masked_fill(k_pos > q_pos, float("-inf"))
            if attn_mask is not None:
                s = s + attn_mask[:, :, qi : qi + bq, ki : ki + k_blk.shape[2]]
            # 2) online softmax update
            m_new = torch.maximum(m, s.amax(dim=-1, keepdim=True))
            # rows that are fully masked keep m = -inf -> avoid nan
            m_new = torch.nan_to_num(m_new, neginf=0.0)
            p = torch.exp(s - m_new)
            alpha = torch.exp(m - m_new)
            l = l * alpha + p.sum(dim=-1, keepdim=True)
            acc = acc * alpha + torch.matmul(p, v_blk.float())
            m = m_new
            n_blocks += 1
        out[:, :, qi : qi + bq] = (acc / l.clamp_min(1e-38)).to(q.dtype)
    if return_stats:
        return out, {"n_q_blocks": math.ceil(sq / block_q),
                     "n_k_blocks_total": n_blocks,
                     "peak_score_elems": block_q * min(sk, block_k + offset + sq)}
    return out


def sdpa_reference(q, k, v, *, causal=True, scale=None):
    """PyTorch fused reference (SDPA) used only as a cross-check, not as the
    implementation under test.

    ``is_causal=True`` aligns the triangle top-left, which is only correct
    when Sq == Sk; for cache-style rectangles (Sq < Sk, queries offset by
    the cached prefix) an explicit offset-aware additive mask is required,
    otherwise SDPA would wrongly hide the cached keys from early queries.
    """
    hkv = k.shape[1]
    if hkv != q.shape[1]:
        k = repeat_kv(k, q.shape[1] // hkv)
        v = repeat_kv(v, q.shape[1] // hkv)
    sq, sk = q.shape[2], k.shape[2]
    if causal and sq != sk:
        mask = build_causal_mask(sq, sk, device=q.device, dtype=q.dtype,
                                 offset=sk - sq)
        return torch.nn.functional.scaled_dot_product_attention(
            q, k, v, attn_mask=mask, scale=scale
        )
    return torch.nn.functional.scaled_dot_product_attention(
        q, k, v, is_causal=causal, scale=scale
    )
