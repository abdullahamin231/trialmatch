"""Browser-based screening application (assignment Task 6).

    cd src && python -m trialmatch.app          # http://127.0.0.1:7860

Gradio Blocks UI with two views:

1. Screening - paste (or pick) a patient note and a trial's criteria, choose
   the checkpoint variant and the decoding mechanism, inspect every
   criterion-level decision with its cited evidence sentences, an aggregated
   preliminary verdict (explicit rule, never a learned patient-level
   outcome) and export a provenance-complete report.
2. Live inference benchmark - measures the cached / uncached / speculative
   decoders on the currently loaded note, in-browser (no static charts).

The app states clearly that it is an educational screening prototype and not
a clinical enrollment decision tool.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone

import gradio as gr
import torch

from .config import ADAPTER_DIR, LABELS_FOR_TYPE, TARGET_MODEL
from .data import Example, parse_note, read_split
from .pipeline import Prediction, Predictor

DISCLAIMER = (
    "EDUCATIONAL SCREENING PROTOTYPE - outputs are research artefacts from a "
    "fine-tuned 360M-parameter language model and must never be used for "
    "clinical trial enrollment decisions."
)

STATE: dict = {}


def _variant_choices() -> list:
    choices = ["base (SmolLM2-360M-Instruct, no fine-tuning)"]
    if (ADAPTER_DIR / "adapter_config.json").exists():
        choices.append("tuned (LoRA fine-tuned)")
    return choices


def load_predictor(variant_short: str, mode: str, gamma: int):
    """Instantiate the requested checkpoint + decoding mechanism."""
    variant = "tuned" if variant_short.startswith("tuned") else "base"
    if mode == "speculative":
        from .speculative import SpeculativeDecoder, SpeculativePredictor
        from .kv_cache import CachedDecoder
        from .config import DRAFT_MODEL
        from transformers import AutoModelForCausalLM

        if "spec" not in STATE:
            t = Predictor(variant, dtype=torch.float32, device=STATE["device"])
            draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL)
            ddec = CachedDecoder(draft)
            STATE["spec"] = (t, SpeculativeDecoder(t.decoder, ddec, gamma=gamma,
                                                   device=STATE["device"]))
        t, spec = STATE["spec"]
        spec.gamma = gamma
        return SpeculativePredictor(spec, t.tokenizer, variant=variant,
                                    device=STATE["device"])
    key = f"pred_{variant}"
    if key not in STATE:
        STATE[key] = Predictor(variant, dtype=torch.float32, device=STATE["device"])
    return STATE[key]


def _criterion_line(line: str) -> tuple[str, str]:
    """'inclusion: <text>' / 'exclusion: <text>' / bare text -> (type, text)."""
    s = line.strip()
    low = s.lower()
    for t in ("inclusion", "exclusion"):
        if low.startswith(t + ":"):
            return t, s[len(t) + 1:].strip()
    return "inclusion", s


def screen(patient_note: str, criteria_text: str, variant: str, mode: str,
           gamma: int, request=None):
    note = patient_note.strip()
    if not note:
        raise gr.Error("Please paste a patient note.")
    if not parse_note(note):
        raise gr.Error("The note must contain numbered sentences ('0. ...', '1. ...') "
                       "so evidence ids stay traceable.")
    lines = [ln for ln in criteria_text.splitlines() if ln.strip()]
    if not lines:
        raise gr.Error("Provide at least one criterion (prefix lines with "
                       "'inclusion:' or 'exclusion:').")

    predictor = load_predictor(variant, mode, gamma)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%SZ")
    rows, explanations = [], []
    positives, negatives, unknowns, na = 0, 0, 0, 0
    for i, line in enumerate(lines):
        ctype, text = _criterion_line(line)
        ex = Example(
            annotation_id=i, patient_id="app-user", trial_id="app",
            trial_title="user supplied", criterion_type=ctype,
            criterion_text=text, label="n/a", evidence_ids=[], note=note,
        )
        pred: Prediction = predictor.predict(ex)
        ev_text = [dict(parse_note(note)).get(j, "") for j in pred.evidence_ids]
        if pred.label in ("included", "not excluded"):
            positives += 1
        elif pred.label in ("not included", "excluded"):
            negatives += 1
        elif pred.label == "not enough information":
            unknowns += 1
        else:
            na += 1
        rows.append([
            ctype, text, pred.label, ", ".join(map(str, pred.evidence_ids)) or "-",
            " | ".join(ev_text) or "-",
        ])
        # Traceable template explanation: only from the note + the decision.
        if pred.evidence_ids:
            why = (f"Decision is grounded in note sentence(s) "
                   f"{pred.evidence_ids}: \"{ev_text[0][:160]}\".")
        else:
            why = ("No supporting sentence was identified. An absence of "
                   "evidence is NOT treated as proof of eligibility or "
                   "ineligibility - the criterion is reported as-is for "
                   "human review.")
        explanations.append(why)

    # Aggregated verdict: explicit, deterministic rule (never trained/scored).
    if na == len(lines):
        verdict = "not applicable"
    elif negatives == 0 and unknowns == 0 and positives > 0:
        verdict = "likely eligible (all criteria met)"
    elif negatives > 0:
        verdict = "likely NOT eligible (>=1 criterion failed)"
    elif unknowns > 0:
        verdict = "undetermined - missing information; manual review required"
    else:
        verdict = "mixed"

    provenance = {
        "generated": stamp,
        "disclaimer": DISCLAIMER,
        "model": TARGET_MODEL,
        "adapter": str(ADAPTER_DIR) if variant.startswith("tuned") else None,
        "decoding": {"mode": mode, "gamma": gamma, "greedy_deterministic": True},
        "counts": {"criteria": len(lines), "met": positives, "failed": negatives,
                   "unknown": unknowns, "not_applicable": na},
    }
    report = json.dumps({"verdict": verdict, "criteria": rows, "provenance": provenance},
                        indent=2)
    return rows, explanations, verdict, report, report


# ---------------------------------------------------------------------------
def demo_benchmark(note: str, out_tokens: int, gamma: int):
    """Live decoding benchmark on the loaded note (Task 6: not static)."""
    from .kv_cache import CachedDecoder, KVCache
    from .speculative import SpeculativeDecoder

    note = note.strip()
    if not note:
        raise gr.Error("Paste a patient note first (Screening tab).")
    text = "\n".join(f"{i}. {s}" for i, s in parse_note(note)[:40])
    tok = STATE["tokenizer"]
    ids = torch.tensor([tok(text, add_special_tokens=True).input_ids], device=STATE["device"])
    tdec: CachedDecoder = STATE["cached_decoder"]

    def measure(mode):
        torch.cuda.reset_peak_memory_stats() if STATE["device"] == "cuda" else None
        t0 = time.perf_counter()
        if mode == "uncached":
            toks, info = tdec.greedy(ids, out_tokens, use_cache=False)
        elif mode == "cached":
            toks, info = tdec.greedy(ids, out_tokens, use_cache=True)
        else:
            spec: SpeculativeDecoder = STATE["spec_dec"]
            toks, info = spec.greedy(ids, out_tokens)
            info = stats_to_info(info)
        dt = time.perf_counter() - t0
        return {
            "mode": mode, "seconds": round(dt, 3),
            "tokens_per_s": round(len(toks) / dt, 2),
            "peak_mem_mb": round((torch.cuda.max_memory_allocated() / 1e6)
                                 if STATE["device"] == "cuda" else 0.0, 1),
            "prefill_s": round(info.get("prefill_s", 0), 3),
            "decode_s": round(info.get("decode_s", dt), 3),
            "acceptance_rate": round(info.get("acceptance", 0), 3) if mode == "speculative" else None,
        }

    rows = [measure("uncached"), measure("cached"), measure("speculative")]
    equal = None
    return rows, equal


def stats_to_info(stats):
    d = stats.as_dict()
    return {"prefill_s": d["prefill_s"], "decode_s": d["decode_s"],
            "acceptance": d["acceptance_rate"]}


# ---------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    all_ex = read_split("validation") + read_split("test")
    example = all_ex[0]
    example_criteria = "\n".join(
        f"{ex.criterion_type}: {ex.criterion_text}" for ex in all_ex[:6]
    )

    with gr.Blocks(title="TrialMatch - educational screening prototype") as ui:
        gr.Markdown(f"# TrialMatch\n{DISCLAIMER}")
        with gr.Tab("Screening"):
            with gr.Row():
                with gr.Column():
                    note_in = gr.Textbox(label="Patient note (numbered sentences)",
                                         lines=10, value=example.note)
                    crit_in = gr.Textbox(label="Trial criteria (one per line, "
                                               "prefix 'inclusion:'/'exclusion:')",
                                         lines=6, value=example_criteria)
                    with gr.Row():
                        variant_in = gr.Radio(_variant_choices(), value=_variant_choices()[0],
                                              label="Checkpoint")
                        mode_in = gr.Radio(["cached", "uncached", "speculative"],
                                           value="cached", label="Decoding mechanism")
                    gamma_in = gr.Slider(2, 8, value=4, step=1, label="Speculative gamma")
                    run_btn = gr.Button("Run screening", variant="primary")
                with gr.Column():
                    verdict_out = gr.Label(label="Aggregated preliminary verdict "
                                                 "(explicit rule, not a learned outcome)")
                    table_out = gr.Dataframe(
                        headers=["type", "criterion", "predicted category",
                                 "evidence ids", "evidence text"], label="Per-criterion results")
                    expl_out = gr.Textbox(label="Traceable explanations", lines=6)
                    with gr.Row():
                        dl_json = gr.DownloadButton("Export JSON report")
                        dl_md = gr.DownloadButton("Export Markdown report")

            run_btn.click(screen, [note_in, crit_in, variant_in, mode_in, gamma_in],
                          [table_out, expl_out, verdict_out, dl_json, dl_md])

        with gr.Tab("Live inference benchmark"):
            gr.Markdown("Measures the decoders on the note above - live, not stored.")
            with gr.Row():
                out_tok = gr.Slider(8, 64, value=24, step=8, label="Output tokens")
                gamma_b = gr.Slider(2, 8, value=4, step=1, label="Speculative gamma")
                bench_btn = gr.Button("Run benchmark", variant="primary")
            bench_table = gr.Dataframe(label="Measured on this runtime")
            bench_btn.click(demo_benchmark, [note_in, out_tok, gamma_b], [bench_table])
    return ui


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    STATE["device"] = device
    from transformers import AutoTokenizer

    STATE["tokenizer"] = AutoTokenizer.from_pretrained(TARGET_MODEL)
    predictor = Predictor("tuned" if (ADAPTER_DIR / "adapter_config.json").exists()
                          else "base", dtype=torch.float32, device=device)
    STATE["cached_decoder"] = predictor.decoder
    STATE[f"pred_{predictor.variant}"] = predictor
    from .kv_cache import CachedDecoder
    from .config import DRAFT_MODEL
    from transformers import AutoModelForCausalLM
    from .speculative import SpeculativeDecoder

    draft = AutoModelForCausalLM.from_pretrained(DRAFT_MODEL)
    STATE["spec_dec"] = SpeculativeDecoder(predictor.decoder, CachedDecoder(draft),
                                           gamma=4, device=device)
    STATE["spec"] = (predictor, STATE["spec_dec"])
    build_ui().launch(server_name="0.0.0.0", server_port=7860, share=False)


if __name__ == "__main__":
    main()
