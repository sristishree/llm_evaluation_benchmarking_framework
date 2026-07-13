from __future__ import annotations

import string
from collections import Counter

from bert_score import score as _bert_score_fn  # type: ignore[import-untyped]
from rouge_score import rouge_scorer as rouge_lib  # type: ignore[import-untyped]

from models.result import RunResult, Scores
from models.task import Task

_ROUGE = rouge_lib.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def _compute_bert_score(pred: str, ref: str) -> float | None:
    """BERTScore F1 (rescaled). The model is cached by bert_score after first call."""
    try:
        _, _, F1 = _bert_score_fn([pred], [ref], lang="en", rescale_with_baseline=True, verbose=False)
        return float(F1[0])
    except Exception:
        return None


def compute_scores(result: RunResult, task: Task) -> Scores:
    """Compute metric scores for a RunResult against the task's expected output."""
    if result.parsed_output is None or result.dry_run:
        return Scores()

    t = result.task_type
    pred = result.parsed_output
    gold = task.expected

    if t == "summarization":
        return _score_summarization(pred, gold)
    if t == "extraction":
        return _score_extraction(pred, gold)
    if t == "classification":
        return _score_classification(pred, gold)
    if t == "qa":
        return _score_qa(pred, gold)
    return Scores()


def _score_summarization(pred: str, gold: str) -> Scores:
    if not isinstance(pred, str) or not pred.strip():
        return Scores()
    s = _ROUGE.score(gold, pred)
    return Scores(
        rouge_1=s["rouge1"].fmeasure,
        rouge_2=s["rouge2"].fmeasure,
        rouge_l=s["rougeL"].fmeasure,
        bert_score=_compute_bert_score(pred, gold),
    )


def _score_extraction(pred: list[dict], gold: list[dict]) -> Scores:
    """Entity-level F1 (exact text+label match) stored as exact_match; token F1 over entity spans."""
    if not isinstance(pred, list):
        pred = []

    def to_set(entities: list[dict]) -> set[tuple[str, str]]:
        return {
            (e.get("text", "").lower().strip(), e.get("label", "").upper())
            for e in entities
            if e.get("text")
        }

    pred_set = to_set(pred)
    gold_set = to_set(gold)

    if not gold_set:
        return Scores(exact_match=1.0 if not pred_set else 0.0)

    tp = len(pred_set & gold_set)
    precision = tp / len(pred_set) if pred_set else 0.0
    recall = tp / len(gold_set)
    entity_f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0
        else 0.0
    )

    pred_tokens = Counter(_tokenize(" ".join(e.get("text", "") for e in pred)))
    gold_tokens = Counter(_tokenize(" ".join(e.get("text", "") for e in gold)))
    token_f1 = _counter_f1(pred_tokens, gold_tokens)

    return Scores(exact_match=entity_f1, token_f1=token_f1)


def _score_classification(pred: str | list[str], gold: str | list[str]) -> Scores:
    """Exact match for single-label; label-set F1 for multi-label."""
    if isinstance(gold, list):
        pred_set = {p.lower().strip() for p in ([pred] if isinstance(pred, str) else pred)}
        gold_set = {g.lower().strip() for g in gold}
        tp = len(pred_set & gold_set)
        precision = tp / len(pred_set) if pred_set else 0.0
        recall = tp / len(gold_set) if gold_set else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )
        return Scores(exact_match=f1)

    pred_str = pred if isinstance(pred, str) else (pred[0] if pred else "")
    return Scores(exact_match=1.0 if _normalize(pred_str) == _normalize(gold) else 0.0)


def _score_qa(pred: str, gold: str | list[str]) -> Scores:
    """SQuAD-style: exact match, token F1, and BERTScore against all acceptable answers."""
    if not isinstance(pred, str):
        pred = str(pred) if pred is not None else ""
    answers = gold if isinstance(gold, list) else [gold]
    bert_scores = [_compute_bert_score(pred, a) for a in answers]
    return Scores(
        exact_match=max(_exact_match(pred, a) for a in answers),
        token_f1=max(_token_f1(pred, a) for a in answers),
        bert_score=max((s for s in bert_scores if s is not None), default=None),
    )


# ------------------------------------------------------------------
# Text helpers
# ------------------------------------------------------------------

def _normalize(text: str) -> str:
    text = text.lower().translate(str.maketrans("", "", string.punctuation))
    return " ".join(text.split())


def _tokenize(text: str) -> list[str]:
    return _normalize(text).split()


def _exact_match(pred: str, gold: str) -> float:
    return 1.0 if _normalize(pred) == _normalize(gold) else 0.0


def _token_f1(pred: str, gold: str) -> float:
    return _counter_f1(Counter(_tokenize(pred)), Counter(_tokenize(gold)))


def _counter_f1(pred: Counter, gold: Counter) -> float:
    common = pred & gold
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / sum(pred.values())
    recall = num_same / sum(gold.values())
    return 2 * precision * recall / (precision + recall)
