from __future__ import annotations

from rouge_score import rouge_scorer as _rouge_scorer

_scorer = _rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)


def score_rouge(reference: str, hypothesis: str) -> dict[str, float]:
    """ROUGE-1, ROUGE-2, and ROUGE-L F-scores. Used for summarization tasks."""
    if not reference or not hypothesis:
        return {"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0}
    scores = _scorer.score(reference, hypothesis)
    return {
        "rouge1": round(scores["rouge1"].fmeasure, 4),
        "rouge2": round(scores["rouge2"].fmeasure, 4),
        "rougeL": round(scores["rougeL"].fmeasure, 4),
    }
