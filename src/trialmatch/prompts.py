"""Prompt construction shared by training, evaluation and the web app."""
from __future__ import annotations

from .config import MAX_SEQ_LEN, SYSTEM_PROMPT
from .data import Example, parse_note

INSTRUCTION = (
    "Answer with two lines:\n"
    "LABEL: <{labels}>\n"
    "EVIDENCE: [<supporting sentence ids>]"
)


def numbered_note(ex: Example) -> str:
    return "\n".join(f"{sid}. {text}" for sid, text in ex.sentences)


def build_user_prompt(ex: Example, max_sentences: int | None = None) -> tuple[str, list[int]]:
    """Return (user prompt, sentence ids that survived truncation)."""
    from .config import LABELS_FOR_TYPE

    labels = ", ".join(LABELS_FOR_TYPE.get(ex.criterion_type, LABELS_FOR_TYPE["inclusion"]))
    sents = ex.sentences
    if max_sentences is not None:
        sents = sents[:max_sentences]
    note = "\n".join(f"{sid}. {text}" for sid, text in sents)
    prompt = (
        f"{INSTRUCTION.format(labels=labels)}\n\n"
        f"PATIENT NOTE\n{note}\n\n"
        f"TRIAL: {ex.trial_title}\n"
        f"CRITERION TYPE: {ex.criterion_type}\n"
        f"CRITERION: {ex.criterion_text}\n"
    )
    return prompt, [sid for sid, _ in sents]


def build_messages(ex: Example) -> list[dict]:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": build_user_prompt(ex)[0]},
    ]


def target_text(ex: Example) -> str:
    """Supervision string. Only expert fields are used (label + sentence ids)."""
    return f"LABEL: {ex.label}\nEVIDENCE: {sorted(ex.evidence_ids)}"


def n_prompt_tokens(tokenizer, ex: Example) -> int:
    msgs = build_messages(ex)
    text = tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return len(tokenizer(text, add_special_tokens=False).input_ids)


def fits_budget(tokenizer, ex: Example, budget: int = MAX_SEQ_LEN - 32) -> bool:
    return n_prompt_tokens(tokenizer, ex) <= budget
