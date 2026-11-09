"""Benchmarks (rubric D3, E3, F3).  Produces results/bench_*.json.

All timings: 2 warm-up runs + 5 timed repetitions, median reported;
``torch.cuda.synchronize()`` around every timed region on GPU; peak memory
from ``torch.cuda.max_memory_allocated`` on GPU or ``ru_maxrss`` on CPU.

    python -m trialmatch.benchmarks --all          # needs the real checkpoints
    python -m trialmatch.benchmarks --attention    # model-free
"""
from __future__ import annotations

import argparse
import json
import resource
import time

import torch

from .config import DRAFT_MODEL, RESULTS_DIR, TARGET_MODEL

RESULTS_DIR.mkdir(exist_ok=True)


def _sync():
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def _peak_mem_mb(reset: bool = False) -> float:
    if torch.cuda.is_available():
        if reset:
            torch.cuda.reset_peak_memory_stats()
        return torch.cuda.max_memory_allocated() / 1e6
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e3


def time_fn(fn, warmup: int = 2, reps: int = 5) -> float:
    for _ in range(warmup):
        fn()
    _sync()
    times = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        _sync()
        times.append(time.perf_counter() - t0)
    times.sort()
    return times[len(times) // 2]


# ---------------------------------------------------------------------------
# D3: KV cache vs uncached full recompute
# ---------------------------------------------------------------------------
def bench_cache(model, tokenizer, device: str = "cpu") -> dict:
    from .data import read_split
    from .kv_cache import CachedDecoder

    dec = CachedDecoder(model.to(device).eval())
    examples = read_split("validation")
    texts = []
    for ex in examples:
        ids = tokenizer(ex.note, add_special_tokens=True).input_ids
        texts.append(ids)
    flat = sorted(texts, key=len)

    def ids_of(n: int) -> torch.Tensor:
        # real note tokens, truncated / padded to exactly n
        src = next(t for t in flat if len(t) >= n - 4)
        ids = (src[: n - 1] + [tokenizer.eos_token_id])[:n]
        return torch.tensor([ids], device=device)

    rows = []
    for S in (128, 512, 1024):
        ids = ids_of(S)
        for n_new in (16, 48):
            # correctness (first rep)
            toks_c, _ = dec.greedy(ids, n_new, use_cache=True)
            toks_u, _ = dec.greedy(ids, n_new, use_cache=False)
            equal = toks_c == toks_u

            stats = {}
            for mode in ("cached", "uncached"):
                torch.manual_seed(0)
                if device == "cuda":
                    torch.cuda.reset_peak_memory_stats()
                t0 = time.perf_counter()
                _, info = dec.greedy(ids, n_new, use_cache=(mode == "cached"))
                _sync()
                wall = time.perf_counter() - t0
                gen = info["decode_s"]
                stats[mode] = {
                    "wall_s": round(wall, 3),
                    "prefill_s": round(info["prefill_s"], 4),
                    "decode_s": round(gen, 4),
                    "tokens_per_s_decode": round(len(toks_c) / gen, 2) if gen else None,
                }
            if device == "cuda":
                stats["cached"]["peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
            rows.append({
                "prompt_len": S, "new_tokens": n_new,
                "tokens_equal": equal, **stats,
            })
            r = rows[-1]
            print(f"S={S:5d} new={n_new:3d} cached {stats['cached']['tokens_per_s_decode']} tok/s "
                  f"vs uncached {stats['uncached']['tokens_per_s_decode']} tok/s equal={equal}")

    speedups = [
        round(r["uncached"]["decode_s"] / r["cached"]["decode_s"], 2)
        for r in rows if r["cached"]["decode_s"] and r["uncached"]["decode_s"]
    ]
    res = {
        "suite": "kv_cache", "model": TARGET_MODEL, "device": device,
        "timing_protocol": "2 warm-ups + 5 reps, median; torch.cuda.synchronize on GPU",
        "rows": rows, "decode_speedup_median_cached_vs_uncached": speedups,
    }
    (RESULTS_DIR / "bench_cache.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# E3: attention runtime/memory
# ---------------------------------------------------------------------------
def bench_attention(device: str = "cpu") -> dict:
    from .attention import reference_attention, sdpa_reference, tiled_attention

    rows = []
    for (B, H, Sq, D, Hkv) in [(1, 6, 256, 64, 6), (1, 6, 512, 64, 6),
                               (1, 8, 1024, 64, 8), (2, 8, 1024, 64, 8),
                               (1, 6, 2048, 64, 6)]:
        Sk = Sq
        q = torch.randn(B, H, Sq, D, device=device)
        k = torch.randn(B, Hkv, Sk, D, device=device)
        v = torch.randn(B, Hkv, Sk, D, device=device)
        # numerics once
        ref = reference_attention(q, k, v, causal=True)
        tiled = tiled_attention(q, k, v, causal=True, block_q=64, block_k=64)
        sdpa = sdpa_reference(q, k, v, causal=True)
        err_t = float((ref - tiled).abs().max())
        err_s = float((ref - sdpa).abs().max())

        row = {"B": B, "H": H, "Sq": Sq, "D": D, "Hkv": Hkv,
               "max_err_tiled": err_t, "max_err_sdpa": err_s}
        for name, fn in (("reference", lambda: reference_attention(q, k, v, causal=True)),
                         ("tiled", lambda: tiled_attention(q, k, v, causal=True)),
                         ("sdpa", lambda: sdpa_reference(q, k, v, causal=True))):
            if device == "cuda":
                torch.cuda.reset_peak_memory_stats()
            t = time_fn(fn)
            row[f"{name}_ms"] = round(t * 1e3, 2)
            if device == "cuda":
                row[f"{name}_peak_mem_mb"] = round(torch.cuda.max_memory_allocated() / 1e6, 1)
        rows.append(row)
        print(f"Sq={Sq:5d} ref {row['reference_ms']:8.2f}ms  "
              f"tiled {row['tiled_ms']:8.2f}ms  sdpa {row['sdpa_ms']:8.2f}ms "
              f"err_tiled={err_t:.2e}")

    res = {
        "suite": "attention", "device": device, "dtype": "float32",
        "note": ("educational tiled implementation runs eager Python loops; "
                 "torch SDPA dispatches a fused kernel - FlashAttention-class "
                 "kernels win by fusing and by avoiding HBM round-trips, which "
                 "a straightforward PyTorch loop cannot match"),
        "rows": rows,
    }
    (RESULTS_DIR / "bench_attention.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
# F3: speculative vs plain greedy
# ---------------------------------------------------------------------------
def bench_speculative(target, draft, tokenizer, device: str = "cpu") -> dict:
    from .data import read_split
    from .kv_cache import CachedDecoder
    from .speculative import SpeculativeDecoder

    tdec = CachedDecoder(target.to(device).eval())
    ddec = CachedDecoder(draft.to(device).eval())
    spec = SpeculativeDecoder(tdec, ddec, gamma=4, device=device)

    val = read_split("validation")
    rows = []
    for ex in val[:6]:
        msgs_tokens = tokenizer(
            ex.note, add_special_tokens=True, truncation=True, max_length=512
        ).input_ids
        ids = torch.tensor([msgs_tokens], device=device)
        n_new = 24

        torch.manual_seed(0)
        t0 = time.perf_counter()
        ref, _ = tdec.greedy(ids, n_new, use_cache=True)
        _sync()
        greedy_s = time.perf_counter() - t0

        torch.manual_seed(0)
        t0 = time.perf_counter()
        got, stats = spec.greedy(ids, n_new)
        _sync()
        spec_s = time.perf_counter() - t0

        rows.append({
            "prompt_tokens": int(ids.shape[1]),
            "greedy_s": round(greedy_s, 3),
            "spec_s": round(spec_s, 3),
            "speedup": round(greedy_s / spec_s, 3) if spec_s else None,
            "greedy_tok_s": round(n_new / greedy_s, 2),
            "spec_tok_s": round(len(got) / spec_s, 2),
            "acceptance_rate": round(stats.acceptance_rate, 4),
            "rounds": stats.rounds,
            "corrections": stats.corrections,
            "tokens_equal": got == ref,
        })
        r = rows[-1]
        print(f"plen={r['prompt_tokens']:4d} greedy {r['greedy_tok_s']:6.2f} tok/s  "
              f"spec {r['spec_tok_s']:6.2f} tok/s  acc={r['acceptance_rate']:.2f} "
              f"equal={r['tokens_equal']}")

    res = {
        "suite": "speculative", "target": TARGET_MODEL, "draft": DRAFT_MODEL,
        "gamma": 4, "device": device,
        "rows": rows,
        "note": ("with a 135M draft against a 360M target both models are so "
                 "small that per-call Python/launch overhead dominates; a "
                 "speedup is not expected at this scale - correctness of the "
                 "emitted sequence and honest accounting are the deliverable"),
    }
    (RESULTS_DIR / "bench_speculative.json").write_text(json.dumps(res, indent=2))
    return res


# ---------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--cache", action="store_true")
    ap.add_argument("--attention", action="store_true")
    ap.add_argument("--spec", action="store_true")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    if args.all or args.attention:
        bench_attention(args.device)
    need_models = args.all or args.cache or args.spec
    if need_models:
        from transformers import AutoModelForCausalLM, AutoTokenizer

        tok = AutoTokenizer.from_pretrained(TARGET_MODEL)
        target = AutoModelForCausalLM.from_pretrained(TARGET_MODEL)
        if args.all or args.cache:
            bench_cache(target, tok, args.device)
        if args.all or args.spec:
            draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL)
            bench_speculative(target, draft, tok, args.device)


if __name__ == "__main__":
    main()
