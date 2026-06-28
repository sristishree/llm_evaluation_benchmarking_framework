from __future__ import annotations


def score_bert(reference: str, hypothesis: str, lang: str = "en") -> dict[str, float]:
    """BERTScore precision, recall, and F1.

    Downloads a transformer model on first call (~400 MB, cached after that).
    Used for summarization and QA tasks.
    """
    from bert_score import score as _bert_score  # lazy import — slow first call

    if not reference or not hypothesis:
        return {"bert_precision": 0.0, "bert_recall": 0.0, "bert_f1": 0.0}
    P, R, F1 = _bert_score([hypothesis], [reference], lang=lang, verbose=False)
    return {
        "bert_precision": round(P[0].item(), 4),
        "bert_recall": round(R[0].item(), 4),
        "bert_f1": round(F1[0].item(), 4),
    }
