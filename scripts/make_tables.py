"""Generate report/tables/*.tex + report/generated/macros.tex from results JSONs.

Run after the Colab artifacts land in results/:

    python scripts/make_tables.py

Everything is optional: missing inputs produce placeholder cells so the paper
always compiles.
"""
from __future__ import annotations

import json
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = ROOT / "report" / "generated"
TABLES = ROOT / "report" / "tables"
OUT.mkdir(parents=True, exist_ok=True)
TABLES.mkdir(parents=True, exist_ok=True)


def load(name: str):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt(x, nd=2):
    if x is None:
        return "--"
    return f"{x:.{nd}f}"


def esc(s: str) -> str:
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(char, char) for char in str(s))


# ---------------------------------------------------------------------------
# Table 1: classification metrics (base vs tuned, val vs test)
# ---------------------------------------------------------------------------
def classification_table():
    rows = []
    for variant in ("base", "tuned"):
        for split in ("validation", "test"):
            d = load(f"eval_{variant}_{split}.json")
            if not d:
                rows.append((variant, split, "--", "--", "--", "--", "--"))
                continue
            c = d["classification_overall"]
            ev = d["evidence_overall"]
            rows.append((variant, split, fmt(c["accuracy"] * 100, 1),
                         fmt(c["macro_f1"], 3),
                         fmt(ev["micro"]["f1"], 3), fmt(ev["macro"]["f1"], 3),
                         fmt(d["invalid_label_rate"], 3)))
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Criterion-level eligibility classification and evidence-sentence",
        r"retrieval on the shared 8/8 validation/test patient splits (206 annotations",
        r"each). Decoding: deterministic greedy with trie-constrained label slot.}",
        r"\label{tab:main}",
        r"\begin{tabular}{llccccc}",
        r"\toprule",
        r"Model & Split & Acc.\ (\%) & Macro-F1 & Ev.-F1 (micro) & Ev.-F1 (macro) & Invalid \\",
        r"\midrule",
    ]
    for v, s, a, f1, emi, ema, inv in rows:
        tex.append(f"{esc(v)} & {esc(s)} & {a} & {f1} & {emi} & {ema} & {inv} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "classification.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
# Table 2: per-class metrics for the tuned model (test)
# ---------------------------------------------------------------------------
def perclass_table():
    d = load("eval_tuned_test.json")
    if not d:
        (TABLES / "perclass.tex").write_text(
            "% pending: results/eval_tuned_test.json\n")
        return
    c = d["classification_overall"]
    inc = d["classification_by_type"]["inclusion"]
    exc = d["classification_by_type"]["exclusion"]
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{Per-class precision/recall/F1 of the fine-tuned model on the",
        r"held-out test patients (overall and by criterion type).}",
        r"\label{tab:perclass}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Scope & Precision & Recall & F1 \\",
        r"\midrule",
    ]
    for scope, m in (("overall", c), ("inclusion", inc), ("exclusion", exc)):
        if not m:
            continue
        for lab, v in m["per_class"].items():
            if v["support"]:
                tex.append(
                    f"{esc(scope)} -- {esc(lab)} & {fmt(v['precision'],3)} & "
                    f"{fmt(v['recall'],3)} & {fmt(v['f1'],3)} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    (TABLES / "perclass.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
# Case-level failure analysis from the held-out prediction export
# ---------------------------------------------------------------------------
def error_table():
    pred_path = RESULTS / "preds_tuned_test.jsonl"
    if not pred_path.exists():
        (TABLES / "errors.tex").write_text("% pending: results/preds_tuned_test.jsonl\n")
        return

    predictions = {
        record["annotation_id"]: record
        for record in (json.loads(line) for line in pred_path.read_text().splitlines())
    }
    selected_ids = (161, 165, 169, 176, 179, 185, 193, 307, 831, 832)
    rows = []
    for annotation_id in selected_ids:
        record = predictions.get(annotation_id)
        if record is None:
            continue
        criterion = record["criterion_text"]
        if len(criterion) > 82:
            criterion = criterion[:79].rsplit(" ", 1)[0] + "..."
        evidence = (
            f"gold {record['gold_evidence_ids']}; "
            f"pred {record['pred_evidence_ids']}"
        )
        rows.append((
            annotation_id,
            record["criterion_type"],
            criterion,
            f"{record['gold_label']} $\\rightarrow$ {record['pred_label']}",
            evidence,
            fmt(record["latency_s"], 3),
        ))

    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Selected case-level errors and outliers from the tuned held-out test",
        r"predictions. Evidence IDs refer to the original numbered patient note.}",
        r"\label{tab:errors}",
        r"\resizebox{\textwidth}{!}{%",
        r"\begin{tabular}{rlp{6.0cm}lp{3.2cm}r}",
        r"\toprule",
        r"ID & Type & Criterion & Gold $\rightarrow$ prediction & Evidence IDs (gold; prediction) & Latency (s) \\",
        r"\midrule",
    ]
    for annotation_id, ctype, criterion, labels, evidence, latency in rows:
        tex.append(
            f"{annotation_id} & {esc(ctype)} & {esc(criterion)} & {labels} & "
            f"{esc(evidence)} & {latency} \\\\"
        )
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table*}", ""]
    (TABLES / "errors.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
# Table 3: KV cache benchmark
# ---------------------------------------------------------------------------
def cache_table():
    d = load("bench_cache.json")
    if not d:
        (TABLES / "cache.tex").write_text("% pending: results/bench_cache.json\n")
        return
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        r"\caption{KV-cache vs.\ uncached full recompute (greedy, identical inputs;",
        f"device {esc(d.get('device','cpu'))}, median of 5 after 2 warm-ups).}}",
        r"\label{tab:cache}",
        r"\resizebox{\columnwidth}{!}{%",
        r"\begin{tabular}{rrcccc}",
        r"\toprule",
        r"$S$ & new & cached tok/s & uncached tok/s & speedup & equal \\",
        r"\midrule",
    ]
    for r in d["rows"]:
        c, u = r["cached"], r["uncached"]
        spd = (u["decode_s"] / c["decode_s"]) if c["decode_s"] else 0
        tex.append(f"{r['prompt_len']} & {r['new_tokens']} & "
                   f"{fmt(c['tokens_per_s_decode'])} & {fmt(u['tokens_per_s_decode'])} & "
                   f"{fmt(spd,1)}$\\times$ & {r['tokens_equal']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}}", r"\end{table}", ""]
    (TABLES / "cache.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
# Table 4: attention benchmark
# ---------------------------------------------------------------------------
def attention_table():
    d = load("bench_attention.json")
    if not d:
        (TABLES / "attention.tex").write_text("% pending: results/bench_attention.json\n")
        return
    caption = r"\caption{Forward-pass latency of one attention layer"
    if d.get("device"):
        caption += " (" + esc(d.get("device", "cpu")) + ", fp32).}"
    else:
        caption += ".}"
    tex = [
        r"\begin{table}[t]",
        r"\centering",
        caption,
        r"\label{tab:attention}",
        r"\begin{tabular}{rrrrr}",
        r"\toprule",
        r"$S_q$ & heads & reference (ms) & tiled (ms) & SDPA (ms) \\",
        r"\midrule",
    ]
    for r in d["rows"]:
        tex.append(f"{r['Sq']} & {r['H']} & {fmt(r['reference_ms'])} & "
                   f"{fmt(r['tiled_ms'])} & {fmt(r['sdpa_ms'])} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table}", ""]
    (TABLES / "attention.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
# Table 5: speculative decoding
# ---------------------------------------------------------------------------
def spec_table():
    d = load("bench_speculative.json")
    if not d:
        (TABLES / "spec.tex").write_text("% pending: results/bench_speculative.json\n")
        return
    tex = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Greedy speculative decoding ($\gamma=4$) vs.\ plain greedy on the",
        r"same prompts; draft SmolLM2-135M, target SmolLM2-360M.}",
        r"\label{tab:spec}",
        r"\begin{tabular}{rrccccc}",
        r"\toprule",
        r"prompt & greedy tok/s & spec tok/s & speedup & acc.\ rate & rounds & equal \\",
        r"\midrule",
    ]
    for r in d["rows"]:
        tex.append(f"{r['prompt_tokens']} & {fmt(r['greedy_tok_s'])} & "
                   f"{fmt(r['spec_tok_s'])} & {fmt(r['speedup'])}$\\times$ & "
                   f"{fmt(r['acceptance_rate'])} & {r['rounds']} & {r['tokens_equal']} \\\\")
    tex += [r"\bottomrule", r"\end{tabular}", r"\end{table*}", ""]
    (TABLES / "spec.tex").write_text("\n".join(tex))


# ---------------------------------------------------------------------------
# macros.tex: headline numbers referenced in prose
# ---------------------------------------------------------------------------
def macros():
    lines = ["% auto-generated by scripts/make_tables.py -- do not edit"]
    t = load("eval_tuned_test.json")
    b = load("eval_base_test.json")
    if t:
        c, ev = t["classification_overall"], t["evidence_overall"]
        lines += [
            f"\\newcommand{{\\TunedTestAcc}}{{{fmt(c['accuracy']*100,1)}\\%}}",
            f"\\newcommand{{\\TunedTestMacroF}}{{{fmt(c['macro_f1'],3)}}}",
            f"\\newcommand{{\\TunedTestEvFone}}{{{fmt(ev['micro']['f1'],3)}}}",
            f"\\newcommand{{\\TunedLatencyMean}}{{{fmt(t['latency_s']['mean'],2)}}}",
        ]
    if b:
        lines.append(f"\\newcommand{{\\BaseTestAcc}}{{{fmt(b['classification_overall']['accuracy']*100,1)}\\%}}")
    kv = load("correctness_kv.json")
    if kv:
        lines.append(f"\\newcommand{{\\KVEqCases}}{{{len(kv['cases'])}}}")
    sp = load("correctness_speculative.json")
    if sp:
        lines.append(f"\\newcommand{{\\SpecEqCases}}{{{len(sp['cases'])}}}")
    (OUT / "macros.tex").write_text("\n".join(lines) + "\n")


def figures():
    """Create the report figures from the saved evaluation and training logs."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    figure_dir = RESULTS / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)

    eval_path = RESULTS / "eval_tuned_test.json"
    if eval_path.exists():
        d = json.loads(eval_path.read_text())
        scopes = [
            ("Overall", d["classification_overall"]),
            ("Inclusion", d["classification_by_type"]["inclusion"]),
            ("Exclusion", d["classification_by_type"]["exclusion"]),
        ]
        fig, axes = plt.subplots(1, 3, figsize=(12, 4))
        for ax, (title, metrics) in zip(axes, scopes):
            matrix = np.asarray(metrics["confusion_matrix"])
            labels = metrics["labels"]
            ax.imshow(matrix, cmap="Blues")
            ax.set_title(title, fontsize=9)
            ax.set_xticks(range(len(labels)), labels, rotation=45, ha="right", fontsize=6)
            ax.set_yticks(range(len(labels)), labels, fontsize=6)
            ax.set_xlabel("Predicted", fontsize=7)
            ax.set_ylabel("Gold", fontsize=7)
            threshold = matrix.max() / 2 if matrix.size else 0
            for row in range(matrix.shape[0]):
                for col in range(matrix.shape[1]):
                    ax.text(
                        col, row, str(matrix[row, col]), ha="center", va="center",
                        fontsize=6, color="white" if matrix[row, col] > threshold else "black",
                    )
        fig.suptitle("Fine-tuned model: held-out test confusion matrices", fontsize=10)
        fig.tight_layout()
        fig.savefig(figure_dir / "cm_tuned_test.png", dpi=180)
        plt.close(fig)

    logs = sorted((RESULTS / "logs").glob("train_*.csv"))
    if logs:
        log_path = max(logs, key=lambda p: p.stat().st_mtime)
        steps, losses, val_steps, val_losses = [], [], [], []
        with log_path.open(newline="") as fh:
            for row in csv.DictReader(fh):
                steps.append(int(row["step"]))
                losses.append(float(row["loss"]))
                if row.get("val_loss"):
                    val_steps.append(int(row["step"]))
                    val_losses.append(float(row["val_loss"]))
        if steps:
            fig, ax = plt.subplots(figsize=(6, 3.2))
            ax.plot(steps, losses, label="Training loss", lw=1.4)
            if val_steps:
                ax.plot(val_steps, val_losses, "o-", label="Validation loss",
                        lw=1.1, ms=3)
            ax.set_xlabel("Optimizer step")
            ax.set_ylabel("Answer-token loss")
            ax.set_title("LoRA fine-tuning loss", fontsize=9)
            ax.grid(alpha=0.25)
            ax.legend(fontsize=8)
            fig.tight_layout()
            fig.savefig(figure_dir / "training_curve.png", dpi=180)
            plt.close(fig)


if __name__ == "__main__":
    classification_table()
    perclass_table()
    error_table()
    cache_table()
    attention_table()
    spec_table()
    macros()
    figures()
    print("report tables written to", TABLES)
