"""Judge calibration utilities.

Tests rubric prompt stability against human judgements on a small held-out slice
before trusting LLM-as-judge scores at scale.

Typical workflow
----------------
1. Run a benchmark to get LLM outputs (stored in results.db).
2. Have humans rate the same outputs on a 0-10 scale (external process).
3. Call ``calibrate()`` with the judge scores and human scores.
4. Inspect the CalibrationReport — low Pearson/Spearman or high bias signals
   that the rubric needs refinement before scaling up.

Usage
-----
    from evaluators.calibrate_judge import calibrate

    report = calibrate(
        judge_scores=[0.80, 0.60, 0.90, 0.40],
        human_scores=[0.75, 0.55, 0.85, 0.35],
    )
    print(report.summary())
    # n=4  Pearson r=0.999  Spearman r=1.000  MAE=0.050  Agreement±1pt=100.0%  Bias=+0.050

Known biases to watch for
--------------------------
- **Verbosity bias**   : judge_mean systematically higher for longer model outputs
- **Position bias**    : scores shift when expected/model ordering is swapped
                         (use LLMJudge(mitigate_position_bias=True) to reduce)
- **Self-enhancement** : judge from same provider as evaluand scores higher
                         (use a different judge_provider where possible)

Stability check
---------------
Run calibrate() twice with different rubric phrasings on the same slice. If
Pearson r between the two judge score vectors is > 0.90, the rubric is stable.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class CalibrationReport:
    """Agreement metrics between LLM judge scores and human scores."""
    n: int
    pearson_r: float | None      # linear correlation (None if n < 2)
    spearman_r: float | None     # rank correlation   (None if n < 2)
    mean_abs_error: float        # mean |judge - human|, both in [0, 1]
    agreement_1pt: float         # fraction where |judge - human| <= 0.1
    bias: float                  # judge_mean - human_mean (+: over-scores)
    judge_mean: float
    human_mean: float

    def summary(self) -> str:
        parts = [f"n={self.n}"]
        if self.pearson_r is not None:
            parts.append(f"Pearson r={self.pearson_r:.3f}")
        else:
            parts.append("Pearson r=n/a")
        if self.spearman_r is not None:
            parts.append(f"Spearman r={self.spearman_r:.3f}")
        else:
            parts.append("Spearman r=n/a")
        parts.append(f"MAE={self.mean_abs_error:.3f}")
        parts.append(f"Agreement±1pt={self.agreement_1pt:.1%}")
        direction = (
            "over-scores" if self.bias > 0.01
            else "under-scores" if self.bias < -0.01
            else "neutral"
        )
        parts.append(f"Bias={self.bias:+.3f} ({direction})")
        return "  ".join(parts)

    def to_dict(self) -> dict:
        return {
            "n":               self.n,
            "pearson_r":       self.pearson_r,
            "spearman_r":      self.spearman_r,
            "mean_abs_error":  self.mean_abs_error,
            "agreement_1pt":   self.agreement_1pt,
            "bias":            self.bias,
            "judge_mean":      self.judge_mean,
            "human_mean":      self.human_mean,
            "summary":         self.summary(),
        }


def calibrate(
    judge_scores: list[float],
    human_scores: list[float],
) -> CalibrationReport:
    """Compute agreement metrics between parallel judge and human score lists.

    Both lists must be in the same normalised range [0, 1].
    If your human scores are on a 0-10 scale, divide by 10 first.

    Raises ValueError if the lists differ in length or are empty.
    """
    j = list(judge_scores)
    h = list(human_scores)
    if len(j) != len(h):
        raise ValueError(
            f"judge_scores ({len(j)}) and human_scores ({len(h)}) must have equal length"
        )
    if not j:
        raise ValueError("Score lists must be non-empty")

    n = len(j)
    mae = sum(abs(a - b) for a, b in zip(j, h)) / n
    agree = sum(1 for a, b in zip(j, h) if abs(a - b) <= 0.1) / n
    j_mean = sum(j) / n
    h_mean = sum(h) / n

    return CalibrationReport(
        n=n,
        pearson_r=_pearson(j, h),
        spearman_r=_spearman(j, h),
        mean_abs_error=mae,
        agreement_1pt=agree,
        bias=j_mean - h_mean,
        judge_mean=j_mean,
        human_mean=h_mean,
    )


# ---------------------------------------------------------------------------
# Correlation helpers — no scipy dependency
# ---------------------------------------------------------------------------

def _pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    num   = sum((a - mx) * (b - my) for a, b in zip(x, y))
    denom = math.sqrt(
        sum((a - mx) ** 2 for a in x) * sum((b - my) ** 2 for b in y)
    )
    return round(num / denom, 6) if denom > 1e-12 else None


def _rank(lst: list[float]) -> list[float]:
    """Compute average ranks (handles ties)."""
    pairs = sorted(enumerate(lst), key=lambda t: t[1])
    ranks = [0.0] * len(lst)
    i = 0
    while i < len(pairs):
        j = i
        while j < len(pairs) - 1 and pairs[j + 1][1] == pairs[j][1]:
            j += 1
        avg = (i + j) / 2 + 1
        for k in range(i, j + 1):
            ranks[pairs[k][0]] = avg
        i = j + 1
    return ranks


def _spearman(x: list[float], y: list[float]) -> float | None:
    if len(x) < 2:
        return None
    return _pearson(_rank(x), _rank(y))
