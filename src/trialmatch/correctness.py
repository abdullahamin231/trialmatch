"""Correctness verification (rubric D2, E2/E3, F2/F3).

Three test families, all with machine-readable outputs under results/:

1. Attention  (no model required): the tiled/online-softmax implementation
   must agree with the reference scaled dot-product attention on controlled
   shapes - causal and non-causal, with and without GQA, including the
   single-query decode case (Sq=1 with an offset, i.e. attending over a
   populated cache).

2. KV cache   (rubric D2): greedy decoding with the explicit cache must
   produce EXACTLY the same tokens as the uncached full-recompute reference
   across at least three input/output lengths; HF ``generate`` with
   use_cache True/False is included as an external cross-check.

3. Speculative decoding (rubric F2/F3): the speculative greedy sequence must
   equal the target model's plain greedy sequence for several gammas, with
   acceptance statistics reported.

Run modes:
    python -m trialmatch.correctness              # tiny random models + attention
    python -m trialmatch.correctness --real       # configured SmolLM2 checkpoints
    python -m trialmatch.correctness --attention-only
"""
from __future__ import annotations

import argparse
import json
import time

import torch

from .attention import reference_attention, sdpa_reference, tiled_attention
from .config import RESULTS_DIR
from .kv_cache import KVCache

RESULTS_DIR.mkdir(exist_ok=True)


# ---------------------------------------------------------------------------
# 1. Attention correctness (model-free)
# ---------------------------------------------------------------------------
def test_attention() -> dict:
    torch.manual_seed(0)
    cases = [
        # (B, H, Sq, Sk, D, H_kv, causal, tag)
        (1, 2, 16, 16, 32, 2, True,  "small causal"),
        (1, 2, 16, 16, 32, 2, False, "small non-causal"),
        (2, 4, 64, 128, 64, 4, True, "rectangular causal (decode-style offset)"),
        (1, 4, 64, 128, 64, 4, False, "rectangular non-causal"),
        (1, 8, 256, 256, 64, 2, True, "GQA causal 256"),
        (1, 4, 1, 129, 64, 4, True,  "single query over cache (Sq=1, offset=128)"),
        (1, 4, 1, 129, 64, 1, True,  "single query, extreme GQA (H_kv=1)"),
        (1, 6, 512, 512, 64, 6, True, "long causal 512"),
    ]
    rows, all_ok = [], True
    for B, H, Sq, Sk, D, Hkv, causal, tag in cases:
        q = torch.randn(B, H, Sq, D)
        k = torch.randn(B, Hkv, Sk, D)
        v = torch.randn(B, Hkv, Sk, D)
        ref = reference_attention(q, k, v, causal=causal)
        out, stats = tiled_attention(q, k, v, causal=causal,
                                     block_q=32, block_k=32, return_stats=True)
        err = (ref - out).abs()
        sdpa = sdpa_reference(q, k, v, causal=causal)
        sdpa_err = (ref - sdpa).abs()
        ok = bool(err.max() < 2e-4 and sdpa_err.max() < 2e-4)
        all_ok &= ok
        rows.append({
            "case": tag, "shape": f"B{B} H{H} Sq{Sq} Sk{Sk} D{D} Hkv{Hkv}",
            "causal": causal,
            "max_abs_err_vs_reference": float(err.max()),
            "mean_abs_err_vs_reference": float(err.mean()),
            "max_abs_err_vs_sdpa": float(sdpa_err.max()),
            "q_blocks": stats["n_q_blocks"],
            "k_blocks_scanned": stats["n_k_blocks_total"],
            "pass": ok,
        })
        print(f"{'PASS' if ok else 'FAIL'}  {tag:46s} "
              f"max_err={err.max():.3e} mean_err={err.mean():.3e}")

    # A fully-masked row is out-of-domain for the reference implementation:
    # torch.softmax over all -inf yields NaN by convention.  The tiled
    # version must nevertheless stay finite (running max is clamped), which
    # is what we assert here.
    q1 = torch.randn(1, 1, 1, 8)
    k1 = torch.randn(1, 1, 1, 8)
    v1 = torch.randn(1, 1, 1, 8)
    mask = torch.full((1, 1, 1, 1), float("-inf"))   # the only key is blocked
    out_ref = reference_attention(q1, k1, v1, causal=False, attn_mask=mask)
    out_masked_tiled = tiled_attention(q1, k1, v1, causal=False, attn_mask=mask)
    nan_free = bool(torch.isfinite(out_masked_tiled).all())
    print(f"{'PASS' if nan_free else 'FAIL'}  fully-masked row: tiled finite={nan_free} "
          f"(reference NaN by torch convention: {bool(torch.isnan(out_ref).all())})")
    all_ok &= nan_free

    res = {"suite": "attention", "pass": all_ok, "cases": rows,
           "fully_masked_row_finite": nan_free}
    (RESULTS_DIR / "correctness_attention.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _greedy_via_decoder(model, ids, n_new, use_cache):
    from .kv_cache import CachedDecoder

    dec = CachedDecoder(model)
    toks, info = dec.greedy(ids, n_new, use_cache=use_cache)
    return toks, info


def _greedy_via_hf(model, ids, n_new, use_cache):
    out = model.generate(
        ids, max_new_tokens=n_new, do_sample=False, num_beams=1,
        use_cache=use_cache, pad_token_id=model.config.vocab_size - 1,
    )
    return out[0, ids.shape[1]:].tolist()


# ---------------------------------------------------------------------------
# 2. KV-cache greedy equivalence (rubric D2: >= 3 input/output lengths)
# ---------------------------------------------------------------------------
def test_kv_equivalence(model, device: str = "cpu", n_new: int = 16) -> dict:
    model = model.to(device).eval()
    torch.manual_seed(1234)
    cases = [(8, n_new), (64, n_new), (192, 24)]   # (prompt_len, new_tokens)
    rows, all_ok = [], True
    for plen, ntok in cases:
        ids = torch.randint(0, model.config.vocab_size, (1, plen), device=device)
        got_c, info_c = _greedy_via_decoder(model, ids, ntok, use_cache=True)
        got_u, info_u = _greedy_via_decoder(model, ids, ntok, use_cache=False)
        hf_c = _greedy_via_hf(model, ids, ntok, use_cache=True)
        hf_u = _greedy_via_hf(model, ids, ntok, use_cache=False)
        same_ref = got_c == got_u
        same_hf = got_c == hf_c[: len(got_c)] and hf_c == hf_u
        ok = same_ref and same_hf
        all_ok &= ok
        rows.append({
            "prompt_len": plen, "new_tokens": ntok,
            "tokens_generated": len(got_c),
            "cached_equals_uncached": same_ref,
            "matches_hf_generate_use_cache": same_hf,
            "prefill_s_cached": round(info_c["prefill_s"], 4),
            "decode_s_cached": round(info_c["decode_s"], 4),
            "decode_s_uncached": round(info_u["decode_s"], 4),
            "pass": ok,
        })
        print(f"{'PASS' if ok else 'FAIL'}  KV eq len={plen:4d} new={ntok} "
              f"cached==uncached:{same_ref} hf_match:{same_hf}")
    res = {"suite": "kv_cache_equivalence", "pass": all_ok, "cases": rows}
    (RESULTS_DIR / "correctness_kv.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# 3. Speculative greedy equivalence (rubric F2/F3)
# ---------------------------------------------------------------------------
def test_spec_equivalence(target, draft, device: str = "cpu",
                          dtype: torch.dtype = torch.float64) -> dict:
    from .kv_cache import CachedDecoder
    from .speculative import SpeculativeDecoder

    # float64 for the random tiny models: with untrained weights the top-1
    # and top-2 logits are frequently near-ties, and the multi-token verify
    # pass rounds differently from single-token decode, flipping argmax.
    # Real trained checkpoints have confident distributions, so fp32 is fine
    # there (that is what --real uses).  Any residual mismatch is diagnosed
    # via the top1-top2 logit gap at the first diverging position.
    target, draft = target.to(device).to(dtype).eval(), draft.to(device).to(dtype).eval()
    tdec = CachedDecoder(target, dtype=dtype)
    ddec = CachedDecoder(draft, dtype=dtype)
    torch.manual_seed(77)
    rows, all_ok = [], True
    for plen, ntok in [(16, 24), (96, 32)]:
        ids = torch.randint(0, target.config.vocab_size, (1, plen), device=device)
        ref, _ = tdec.greedy(ids, ntok, use_cache=True)
        for gamma in (2, 4):
            spec = SpeculativeDecoder(tdec, ddec, gamma=gamma, device=device)
            got, stats = spec.greedy(ids, ntok)
            same = got == ref
            diag = {}
            if not same:
                i = next(j for j, (a, b) in enumerate(zip(got, ref)) if a != b)
                prefix = torch.cat([ids, torch.tensor([ref[:i]], device=ids.device)], dim=1)
                logits = tdec.forward(prefix, KVCache(tdec.n_layers))[0, -1]
                top2 = logits.topk(2).values
                diag = {
                    "first_diff_index": i,
                    "top1_minus_top2_logit_gap": float(top2[0] - top2[1]),
                    "note": ("fp tie-break: verify pass and single-token decode "
                             "round differently on near-ties"),
                }
            ok = same
            all_ok &= ok
            rows.append({
                "prompt_len": plen, "new_tokens": ntok, "gamma": gamma,
                "tokens_equal_greedy": same,
                "acceptance_rate": round(stats.acceptance_rate, 4),
                "rounds": stats.rounds, "corrections": stats.corrections,
                "bonus_tokens": stats.bonus_tokens,
                "diagnostics": diag,
                "pass": ok,
            })
            print(f"{'PASS' if ok else 'FAIL'}  SPEC eq len={plen} new={ntok} "
                  f"gamma={gamma} equal:{same} acc={stats.acceptance_rate:.2f}")
    res = {"suite": "speculative_equivalence", "pass": all_ok, "cases": rows}
    (RESULTS_DIR / "correctness_speculative.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
def _tiny_models():
    """Small random Llama checkpoints built from config - no downloads."""
    from transformers import LlamaConfig, LlamaForCausalLM

    torch.manual_seed(7)
    cfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                      num_hidden_layers=2, num_attention_heads=4,
                      num_key_value_heads=2, max_position_embeddings=512,
                      bos_token_id=0, eos_token_id=0, pad_token_id=0)
    target = LlamaForCausalLM(cfg)
    dcfg = LlamaConfig(vocab_size=128, hidden_size=64, intermediate_size=128,
                       num_hidden_layers=1, num_attention_heads=4,
                       num_key_value_heads=4, max_position_embeddings=512,
                       bos_token_id=0, eos_token_id=0, pad_token_id=0)
    draft = LlamaForCausalLM(dcfg)
    return target, draft


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--real", action="store_true",
                    help="use the configured SmolLM2 checkpoints instead of tiny ones")
    ap.add_argument("--attention-only", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    t0 = time.time()
    attn = test_attention()

    kv = spec = None
    if not args.attention_only:
        if args.real:
            from transformers import AutoModelForCausalLM

            from .config import DRAFT_MODEL, TARGET_MODEL

            target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL)
            draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL)
        else:
            target, draft = _tiny_models()
        kv = test_kv_equivalence(target, args.device)
        spec = test_spec_equivalence(
            target, draft, args.device,
            dtype=torch.float32 if args.real else torch.float64,
        )

    summary = {
        "attention": attn["pass"],
        "kv_cache": kv["pass"] if kv else None,
        "speculative": spec["pass"] if spec else None,
        "wall_clock_s": round(time.time() - t0, 1),
        "all_pass": attn["pass"] and (kv or {}).get("pass", True)
        and (spec or {}).get("pass", True),
    }
    (RESULTS_DIR / "correctness_summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))
    if not summary["all_pass"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
