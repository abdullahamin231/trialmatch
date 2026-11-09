"""LoRA fine-tuning of the target model (assignment Task 1).

The model reads (patient note, criterion type, criterion text) and produces a
two-line structured answer: ``LABEL: <category>`` and ``EVIDENCE: [ids]``.
Loss is computed only on the answer tokens (prompt tokens are masked with
-100), so the model learns the decision + citation behaviour rather than
copying the note.

    python -m trialmatch.train_lora --epochs 2 --batch-size 4
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import torch

from .config import (
    ADAPTER_DIR,
    LOG_DIR,
    MAX_SEQ_LEN,
    TARGET_MODEL,
)
from .data import read_split
from .prompts import build_messages, target_text


def build_tokenizer(model_id: str):
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(model_id)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "right"
    return tok


def encode_example(tok, ex) -> tuple[list[int], list[int]]:
    prompt = tok.apply_chat_template(
        build_messages(ex), tokenize=False, add_generation_prompt=True
    )
    answer = target_text(ex) + tok.eos_token
    p_ids = tok(prompt, add_special_tokens=False).input_ids
    a_ids = tok(answer, add_special_tokens=False).input_ids
    ids = (p_ids + a_ids)[:MAX_SEQ_LEN]
    labels = ([-100] * len(p_ids) + a_ids)[:MAX_SEQ_LEN]
    return ids, labels


def make_batches(examples, tok, batch_size: int, shuffle: bool, seed: int = 0):
    order = list(range(len(examples)))
    if shuffle:
        random.Random(seed).shuffle(order)
    for i in range(0, len(order), batch_size):
        chunk = [examples[j] for j in order[i : i + batch_size]]
        enc = [encode_example(tok, e) for e in chunk]
        maxlen = max(len(ids) for ids, _ in enc)
        input_ids, attn, labels = [], [], []
        for ids, lab in enc:
            pad = maxlen - len(ids)
            input_ids.append(ids + [tok.pad_token_id] * pad)
            attn.append([1] * len(ids) + [0] * pad)
            labels.append(lab + [-100] * pad)
        yield (
            torch.tensor(input_ids),
            torch.tensor(attn),
            torch.tensor(labels),
        )


def build_model(dtype, use_lora: bool, r: int, alpha: int, dropout: float):
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(TARGET_MODEL, dtype=dtype)
    model.config.use_cache = False
    if use_lora:
        from peft import LoraConfig, get_peft_model

        cfg = LoraConfig(
            r=r,
            lora_alpha=alpha,
            lora_dropout=dropout,
            bias="none",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                            "gate_proj", "up_proj", "down_proj"],
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, cfg)
        model.print_trainable_parameters()
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--grad-accum", type=int, default=2)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=20)
    ap.add_argument("--max-steps", type=int, default=0)
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--dtype", default="bfloat16", choices=["float32", "bfloat16"])
    ap.add_argument("--out", default=str(ADAPTER_DIR))
    ap.add_argument("--resume", action="store_true",
                    help="continue from <out>/checkpoint-last if it exists")
    ap.add_argument("--save-every", type=int, default=10)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--amp", action="store_true",
                    help="fp16 autocast forward on GPU (recommended on T4)")
    ap.add_argument("--val-every", type=int, default=25,
                    help="optimizer steps between validation-loss evaluations")
    ap.add_argument("--val-examples", type=int, default=64)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    dtype = getattr(torch, args.dtype)

    tok = build_tokenizer(TARGET_MODEL)
    train = read_split("train")
    val = read_split("validation")

    model = build_model(dtype, True, args.lora_r, args.lora_alpha, args.lora_dropout)
    model.to(args.device)
    model.train()
    trainable = [p for p in model.parameters() if p.requires_grad]

    ckpt_dir = Path(args.out) / "checkpoint-last"
    start_step = 0
    if args.resume and (ckpt_dir / "adapter_model.safetensors").exists():
        from peft import PeftModel

        meta_path = ckpt_dir / "resume_state.json"
        start_step = json.loads(meta_path.read_text())["step"] if meta_path.exists() else 0
        model = PeftModel.from_pretrained(model, str(ckpt_dir), is_trainable=True)
        print(f"resumed adapter from {ckpt_dir} at step {start_step}", flush=True)
    opt = torch.optim.AdamW(trainable, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.95))

    steps_per_epoch = math.ceil(len(train) / args.batch_size)
    total_steps = int(args.epochs * steps_per_epoch / args.grad_accum)
    if args.max_steps:
        total_steps = min(total_steps, args.max_steps)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt,
        lambda s: min(1.0, (s + 1) / max(1, args.warmup))
        * max(0.0, 0.5 * (1 + math.cos(math.pi * s / max(1, total_steps)))),
    )

    log_path = LOG_DIR / f"train_{time.strftime('%Y%m%d_%H%M%S')}.csv"
    log = log_path.open("w")
    log.write("step,epoch,loss,lr,elapsed_s,val_loss\n")
    log.flush()
    # fixed validation batch (loss curve must be comparable across steps)
    val_batches = list(make_batches(val[: args.val_examples], tok,
                                    args.batch_size, shuffle=False))
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and args.device == "cuda")

    def val_loss() -> float:
        model.eval()
        tot, n = 0.0, 0
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16,
                                             enabled=args.amp and args.device == "cuda"):
            for input_ids, attn, labels in val_batches:
                out = model(input_ids=input_ids.to(args.device),
                            attention_mask=attn.to(args.device),
                            labels=labels.to(args.device))
                tot += float(out.loss) * len(input_ids)
                n += len(input_ids)
        model.train()
        return tot / max(1, n)
    cfg = {
        "base_model": TARGET_MODEL,
        "method": "LoRA (peft)",
        "lora_r": args.lora_r,
        "lora_alpha": args.lora_alpha,
        "lora_dropout": args.lora_dropout,
        "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj",
                           "gate_proj", "up_proj", "down_proj"],
        "trainable_params": sum(p.numel() for p in trainable),
        "total_params": sum(p.numel() for p in model.parameters()),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "grad_accum": args.grad_accum,
        "effective_batch": args.batch_size * args.grad_accum,
        "lr": args.lr,
        "schedule": "linear-warmup + cosine decay",
        "max_seq_len": MAX_SEQ_LEN,
        "dtype": args.dtype,
        "seed": args.seed,
        "train_examples": len(train),
        "optimizer_steps": total_steps,
        "loss_masking": "prompt tokens masked (-100); only answer tokens supervise",
    }
    print(json.dumps(cfg, indent=2))

    t_start = time.time()
    step = start_step
    running = 0.0
    micro = 0
    opt.zero_grad(set_to_none=True)
    for _ in range(start_step):          # fast-forward the LR schedule
        sched.step()
    for epoch in range(math.ceil(args.epochs)):
        for input_ids, attn, labels in make_batches(
            train, tok, args.batch_size, shuffle=True, seed=args.seed * 1000 + epoch
        ):
            if step < start_step:        # already trained before a restart
                micro += 1
                if micro % args.grad_accum == 0:
                    step += 1
                continue
            input_ids = input_ids.to(args.device)
            attn = attn.to(args.device)
            labels = labels.to(args.device)
            out = model(input_ids=input_ids, attention_mask=attn, labels=labels)
            loss = out.loss
            if args.amp:
                scaler.scale(loss / args.grad_accum).backward()
            else:
                (loss / args.grad_accum).backward()
            running += float(loss.detach())
            micro += 1
            if micro % args.grad_accum == 0:
                torch.nn.utils.clip_grad_norm_(trainable, 1.0)
                if args.amp:
                    scaler.step(opt)
                    scaler.update()
                else:
                    opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                avg = running / micro
                vl = f"{val_loss():.5f}" if step % args.val_every == 0 else ""
                log.write(f"{step},{epoch},{avg:.5f},{sched.get_last_lr()[0]:.3e},"
                          f"{time.time()-t_start:.1f},{vl}\n")
                log.flush()
                if step % 5 == 0 or step == 1:
                    print(f"step {step:4d} ep {epoch} loss {avg:.4f} "
                          f"({(time.time()-t_start)/max(1,step-start_step):.2f}s/step)", flush=True)
                if step % args.save_every == 0:
                    ckpt_dir.mkdir(parents=True, exist_ok=True)
                    model.save_pretrained(ckpt_dir)
                    (ckpt_dir / "resume_state.json").write_text(json.dumps({"step": step}))
                    print(f"  checkpoint @ step {step}", flush=True)
                if step >= total_steps:
                    break
        if step >= total_steps:
            break
        running = 0.0
        micro = 0

    out_dir = Path(args.out)
    model.save_pretrained(out_dir)
    tok.save_pretrained(out_dir)
    cfg["wall_clock_s"] = round(time.time() - t_start, 1)
    cfg["final_step"] = step
    cfg["train_log"] = str(log_path)
    (out_dir / "train_config.json").write_text(json.dumps(cfg, indent=2))
    log.close()
    print(f"saved adapter to {out_dir} in {cfg['wall_clock_s']}s")


if __name__ == "__main__":
    main()
