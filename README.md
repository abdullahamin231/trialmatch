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
cd .. && python scripts/make_tables.py       # report tables + figures
```

To compile the report after generating its tables and figures:

```bash
cd report
pdflatex -interaction=nonstopmode main.tex
bibtex main
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode main.tex
```

## Colab MCP (VS Code)

The workspace MCP configuration is in `.vscode/mcp.json`. It starts Google's
`colab-mcp` server with `uvx`; the first launch may download the server and its
Python 3.13+ environment. In VS Code, reload the window if the server is not listed,
then use its Colab connection tool and accept the connection in the signed-in Colab
tab. Keep only one agent connected to a given notebook session at a time.

Training was executed on a free Colab T4 (fp16 autocast over fp32 master weights,
~2.1 s/optimizer step, 225 steps, 8.7M trainable LoRA parameters).

## Data

`ncbi/TrialGPT-Criterion-Annotations` — the GPT-4 prediction columns are dropped at
load time. Split: unique patient IDs sorted, shuffled with `numpy.random.default_rng(42)`,
first 37 → train, next 8 → validation, remaining 8 → test (602/206/206 annotations).
The manifest with patient lists, counts and SHA-256 fingerprints is published at
`data/splits/split_manifest.json`; a leakage check asserts no patient spans splits.

## Model weights

The trained LoRA adapter for SmolLM2-360M-Instruct is available in the local workspace
at `artifacts/lora_smollm2-360m/` (34.8 MB weights). SHA-256 for
`adapter_model.safetensors`:
`d117c1048600687af21c67061a78c530d04c2d0f1308babb2f59e99d48e4848c`.
Weights are excluded from Git and published as a
[GitHub release](https://github.com/abdullahamin231/trialmatch/releases/tag/assignment-2-artifacts-2026-11).
Download the [adapter weights](https://github.com/abdullahamin231/trialmatch/releases/download/assignment-2-artifacts-2026-11/adapter_model.safetensors)
and pair them with the local `adapter_config.json` in that directory. The release also
contains the compiled report and a ZIP with the adapter, config, tuned test/validation
predictions, and error exports. The tuned variant loads when `adapter_config.json` and
`adapter_model.safetensors` are present at that path.

The Colab bundle in `results/trialmatch_colab_artifacts.zip` contains the adapter and
the tuned test/validation prediction and error exports. Test predictions contain 206
records, one per held-out annotation.
