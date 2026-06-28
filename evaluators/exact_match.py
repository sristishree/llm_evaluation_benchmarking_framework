from __future__ import annotations

import re
import string
from collections import Counter


# ---------------------------------------------------------------------------
# Shared normalisation
# ---------------------------------------------------------------------------

def _normalize(text: str) -> str:
    """Lowercase, strip articles, punctuation, and extra whitespace."""
    text = text.lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(ch for ch in text if ch not in string.punctuation)
    return " ".join(text.split())


# ---------------------------------------------------------------------------
# QA metrics  (SQuAD-style)
# ---------------------------------------------------------------------------

def qa_exact_match(prediction: str, ground_truths: list[str]) -> float:
    """1.0 if prediction matches any ground truth after normalization, else 0.0."""
    pred = _normalize(prediction)
    return float(any(pred == _normalize(gt) for gt in ground_truths))


def qa_token_f1(prediction: str, ground_truths: list[str]) -> float:
    """Max token-level F1 over all acceptable ground truths."""

    def _f1(pred: str, ref: str) -> float:
        pred_tokens = _normalize(pred).split()
        ref_tokens = _normalize(ref).split()
        common = Counter(pred_tokens) & Counter(ref_tokens)
        num_common = sum(common.values())
        if num_common == 0:
            return 0.0
        precision = num_common / len(pred_tokens)
        recall = num_common / len(ref_tokens)
        return 2 * precision * recall / (precision + recall)

    return max((_f1(prediction, gt) for gt in ground_truths), default=0.0)


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def classification_exact_match(prediction: str, expected: str | list[str]) -> float:
    """Case-insensitive exact match for single or multi-label classification."""
    pred = prediction.strip().lower()
    if isinstance(expected, list):
        return float(pred in [e.strip().lower() for e in expected])
    return float(pred == expected.strip().lower())


# ---------------------------------------------------------------------------
# NER / extraction metrics
# ---------------------------------------------------------------------------

def entity_f1(
    predicted: list[dict],
    expected: list[dict],
) -> dict[str, float]:
    """Micro-averaged, macro-averaged, and per-label entity F1 for extraction tasks.

    Both predicted and expected are lists of {"text": ..., "label": ...}.
    Matching is case-insensitive on text, case-insensitive on label.

    Keys returned:
      micro_precision, micro_recall, micro_f1   — aggregate across all entities
      macro_f1                                  — unweighted mean of per-label F1s
      per_label_f1_<label>                      — F1 for each individual entity type
    """
    pred_set = {(e["text"].strip().lower(), e["label"].strip().upper()) for e in predicted}
    exp_set = {(e["text"].strip().lower(), e["label"].strip().upper()) for e in expected}

    tp = len(pred_set & exp_set)
    fp = len(pred_set - exp_set)
    fn = len(exp_set - pred_set)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    metrics: dict[str, float] = {
        "micro_precision": round(precision, 4),
        "micro_recall": round(recall, 4),
        "micro_f1": round(f1, 4),
    }

    label_f1s = []
    for label in {e["label"].strip().upper() for e in expected}:
        pred_l = {pair for pair in pred_set if pair[1] == label}
        exp_l = {pair for pair in exp_set if pair[1] == label}
        tp_l = len(pred_l & exp_l)
        fp_l = len(pred_l - exp_l)
        fn_l = len(exp_l - pred_l)
        p_l = tp_l / (tp_l + fp_l) if (tp_l + fp_l) > 0 else 0.0
        r_l = tp_l / (tp_l + fn_l) if (tp_l + fn_l) > 0 else 0.0
        f1_l = (2 * p_l * r_l / (p_l + r_l)) if (p_l + r_l) > 0 else 0.0
        metrics[f"per_label_f1_{label.lower()}"] = round(f1_l, 4)
        label_f1s.append(f1_l)

    metrics["macro_f1"] = round(sum(label_f1s) / len(label_f1s) if label_f1s else 0.0, 4)

    return metrics
