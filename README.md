# TrialMatch — Patient–Trial Eligibility Screening with Efficient Transformer Inference

GenAI Assignment #2: a Transformer-based system that matches patient descriptions to
individual inclusion/exclusion criteria of clinical trials, justifies every decision with
cited evidence from the patient note, and demonstrates **KV caching**, **tiled
online-softmax attention** and **speculative decoding** — implemented from scratch, not
wrapped around an existing repository.

**Educational screening prototype — never a clinical enrollment decision tool.**

## Repository layout

```
src/trialmatch/
  config.py        central configuration (models, split rule, label space, budgets)
  data.py          TrialGPT ingestion, leakage guards, mandated 37/8/8 patient split
  prompts.py       prompt construction shared by training / evaluation / app
  schema.py        output parsing + trie-constrained label decoding
  attention.py     reference SDPA and tiled online-softmax attention (Task 4)
  kv_cache.py      explicit KV-cache object + cache-aware decoder (Task 3)
  train_lora.py    LoRA fine-tuning (Task 1)
  pipeline.py      production prediction path (constrained, cache-aware)
  metrics.py       classification + evidence-retrieval metrics (Task 2)
  evaluate.py      evaluation runner: metrics, confusion matrices, error taxonomy
  speculative.py   greedy speculative decoding with cache resync (Task 5)
  correctness.py   correctness suite: attention, KV equivalence, speculation (D2/E3/F2)
  benchmarks.py    timing/memory benchmarks (D3/E3/F3)
  app.py           Gradio screening application + live benchmark view (Task 6)
report/            IEEE-format LaTeX paper (auto-filled tables via scripts/make_tables.py)
data/splits/       split manifest (patient IDs, counts, fingerprints)
results/           evaluation JSONs, benchmark logs, figures, training curves
```

## Reproducing

```bash
pip install -r requirements.txt
cd src
python -m trialmatch.data --prepare          # download dataset + build split
python -m trialmatch.train_lora --epochs 3 --batch-size 2 --grad-accum 4 \
       --lr 1e-3 --lora-r 16 --lora-alpha 32 --amp --dtype float32
python -m trialmatch.evaluate --variant both --splits validation,test
python -m trialmatch.correctness             # model-free + tiny-model suite
python -m trialmatch.correctness --real      # same suite on real checkpoints
python -m trialmatch.benchmarks --all
python -m trialmatch.app                     # http://127.0.0.1:7860
```

Training was executed on a free Colab T4 (fp16 autocast over fp32 master weights,
~2.1 s/optimizer step, 225 steps, 8.7M trainable LoRA parameters).

## Data

`ncbi/TrialGPT-Criterion-Annotations` — the GPT-4 prediction columns are dropped at
load time. Split: unique patient IDs sorted, shuffled with `numpy.random.default_rng(42)`,
first 37 → train, next 8 → validation, remaining 8 → test (602/206/206 annotations).
The manifest with patient lists, counts and SHA-256 fingerprints is published at
`data/splits/split_manifest.json`; a leakage check asserts no patient spans splits.

## Model weights

The LoRA adapter (SmolLM2-360M-Instruct base) is hosted externally — see
`artifacts/README.md` after training, or the submission links. Place the adapter under
`artifacts/lora_smollm2-360m/` to use the `tuned` variant in evaluation and the app.
