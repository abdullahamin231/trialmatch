"""Output schema: parsing, validation and constrained label decoding."""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from .config import LABELS, LABELS_FOR_TYPE

LABEL_RE = re.compile(r"LABEL\s*:\s*([^\n]*)", re.IGNORECASE)
EVIDENCE_RE = re.compile(r"EVIDENCE\s*:\s*(\[[^\]]*\]?)", re.IGNORECASE | re.DOTALL)
NUM_RE = re.compile(r"-?\d+")


@dataclass
class ParsedPrediction:
    label: str
    label_valid: bool
    evidence_ids: list[int]
    raw: str
    evidence_valid: bool = True
    notes: list[str] = field(default_factory=list)


def _canon_label(raw: str, criterion_type: str) -> tuple[str, bool]:
    """Map free text to one of the six canonical labels.

    Exact (case/whitespace-insensitive) matches and unambiguous prefixes of a
    legal label are accepted. Anything else is flagged invalid and mapped to
    "not enough information": an unparseable answer must never be converted
    into a positive eligibility decision.
    """
    legal = LABELS_FOR_TYPE.get(criterion_type, LABELS_FOR_TYPE["inclusion"])
    cleaned = re.sub(r"[^a-z ]", " ", raw.lower())
    cleaned = " ".join(cleaned.split())
    for lab in legal:
        if cleaned == lab:
            return lab, True
    for lab in sorted(legal, key=len, reverse=True):
        if cleaned.startswith(lab) and len(cleaned) - len(lab) <= 3:
            return lab, True
    for lab in LABELS:  # legal, but for the other criterion type
        if cleaned == lab:
            return "not enough information", False
    return "not enough information", False


def parse_prediction(text: str, criterion_type: str, max_sentence_id: int) -> ParsedPrediction:
    notes: list[str] = []
    m = LABEL_RE.search(text)
    if m:
        label, ok = _canon_label(m.group(1), criterion_type)
        if not ok:
            notes.append(f"illegal label text: {m.group(1).strip()!r}")
    else:
        m2 = re.search(
            r"(?i)\b(included|not included|excluded|not excluded|not enough information|not applicable)\b",
            text,
        )
        if m2:
            label, ok = _canon_label(m2.group(1), criterion_type)
            notes.append("LABEL field missing; recovered from free text")
        else:
            label, ok = "not enough information", False
            notes.append("no label produced")

    em = EVIDENCE_RE.search(text)
    ev: list[int] = []
    ev_valid = True
    if em:
        body = em.group(1)
        if not body.strip().endswith("]"):
            ev_valid = False
            notes.append("unterminated EVIDENCE list")
        ids = [int(x) for x in NUM_RE.findall(body)]
        ev = sorted({i for i in ids if 0 <= i <= max_sentence_id})
        if len(ev) != len(set(ids)):
            ev_valid = False
            notes.append("out-of-range sentence ids dropped")
    else:
        ev_valid = False
        notes.append("no EVIDENCE field produced")
    return ParsedPrediction(label, ok, ev, text, ev_valid, notes)


# ---------------------------------------------------------------------------
# Constrained label decoding
# ---------------------------------------------------------------------------
def build_trie(strings: list[str]) -> dict:
    root: dict = {}
    for s in strings:
        node = root
        for ch in s:
            node = node.setdefault(ch, {})
        node[""] = True
    return root


_TOKEN_STRINGS: dict[str, dict[int, str]] = {}


def _token_strings(tokenizer, model_id: str) -> dict[int, str]:
    """token id -> surface string, computed once per model (vocab is 151k)."""
    if model_id not in _TOKEN_STRINGS:
        _TOKEN_STRINGS[model_id] = {
            i: tokenizer.decode([i], clean_up_tokenization_spaces=False)
            for i in range(len(tokenizer))
        }
    return _TOKEN_STRINGS[model_id]


class ConstrainedLabelDecoder:
    """Character-trie constraint over the label slot.

    At every decoding step only tokens that keep the emitted text a prefix of
    some *legal* label are allowed. This eliminates cross-type label errors
    (e.g. "excluded" on an inclusion criterion) at the source instead of
    repairing them afterwards. The allowed token sets are precomputed once per
    (model, label set) pair, so per-step cost is a single index_select.
    """

    _CACHE: dict[tuple[str, str], list[tuple[dict, set[int]]]] = {}

    def __init__(self, tokenizer, labels: tuple[str, ...], model_id: str = "default"):
        self.tokenizer = tokenizer
        self.labels = tuple(labels)
        key = (model_id, "|".join(self.labels))
        if key not in self._CACHE:
            trie = build_trie(list(self.labels))
            tokstr = _token_strings(tokenizer, model_id)
            allowed_per_node: list[tuple[dict, set[int]]] = []

            def walk(node: dict) -> None:
                allowed: set[int] = set()
                for tok_id, piece in tokstr.items():
                    if not piece:
                        continue
                    n = node
                    ok = True
                    for ch in piece:
                        if ch in n:
                            n = n[ch]
                        else:
                            ok = False
                            break
                    if ok:
                        allowed.add(tok_id)
                allowed_per_node.append((node, allowed))
                for ch, child in node.items():
                    if ch != "":
                        walk(child)

            walk(trie)
            self._CACHE[key] = allowed_per_node
        self._table = {id(node): allowed for node, allowed in self._CACHE[key]}
        self.trie = build_trie(list(self.labels))
        self.active = False
        self.node: dict | None = None
        self.text = ""

    # -- decoding interface ------------------------------------------------
    def reset(self) -> None:
        self.active = False
        self.node = None
        self.text = ""

    def begin(self) -> None:
        self.active = True
        self.node = self.trie
        self.text = ""

    def mask_scores(self, scores):
        """scores: (1, vocab) -> masked scores, only while in the label slot."""
        import torch

        if not self.active or self.node is None:
            return scores
        allowed = self._table.get(id(self.node), set())
        if not allowed:
            return scores
        mask = torch.full_like(scores, float("-inf"))
        idx = torch.tensor(sorted(allowed), dtype=torch.long)
        mask[0, idx] = 0.0
        return scores + mask

    def advance(self, token_id: int) -> None:
        if not self.active or self.node is None:
            return
        piece = self.tokenizer.decode([token_id], clean_up_tokenization_spaces=False)
        for ch in piece:
            if ch in self.node:
                self.node = self.node[ch]
                self.text += ch
            else:
                self.active = False
                self.node = None
                return
        if self.node.get(""):
            self.active = False  # complete label reached
