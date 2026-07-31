"""Rubric dimension definitions and runtime override management.

Dimensions are task-type specific. The judge scores each independently, then
the overall score is their mean (or the model's own "overall" estimate when
the dimension set is empty).

Override precedence (highest → lowest):
  1. Inline rubric_overrides dict passed at run time
  2. rubric_overrides_file (server-side YAML)
  3. task.rubric (shipped with the task bank)

Overrides are flagged so leaderboard results remain interpretable.
"""
from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import yaml


class Dimension(NamedTuple):
    key: str      # stored in Scores.extra as "judge_{key}"
    label: str    # shown in UI
    prompt: str   # instruction appended to the judge user message


# ---------------------------------------------------------------------------
# Per-task-type dimension specifications
# ---------------------------------------------------------------------------

DIMENSIONS: dict[str, list[Dimension]] = {
    "summarization": [
        Dimension(
            "faithfulness",
            "Faithfulness",
            "Faithfulness (0-10): Does the summary contain only facts supported by the "
            "source text? Penalise every hallucinated detail or fabricated statement. "
            "A score of 0 means the summary contradicts or invents content; 10 means "
            "every claim is directly verifiable in the input.",
        ),
        Dimension(
            "coverage",
            "Coverage",
            "Coverage (0-10): Does the summary capture all key points, main arguments, "
            "and essential information from the source? Penalise significant omissions "
            "proportionally to their importance. Score 10 only if no important point is "
            "missing.",
        ),
        Dimension(
            "conciseness",
            "Conciseness",
            "Conciseness (0-10): Is the summary appropriately brief without padding, "
            "repetition, or filler? Do NOT penalise for length if the content is "
            "substantive. Penalise only redundancy and unnecessary elaboration.",
        ),
    ],
    "qa": [
        Dimension(
            "faithfulness",
            "Faithfulness",
            "Faithfulness (0-10): Is the answer grounded in the provided context or "
            "input? Penalise hallucinated facts that are absent from the passage. "
            "Score 10 only if every stated fact is traceable to the input.",
        ),
        Dimension(
            "completeness",
            "Completeness",
            "Completeness (0-10): Does the answer fully address all parts of the "
            "question, including any sub-questions? Penalise partial or evasive answers "
            "proportionally to the missing information.",
        ),
        Dimension(
            "conciseness",
            "Conciseness",
            "Conciseness (0-10): Is the answer focused and free of irrelevant tangents "
            "or unnecessary padding? Do NOT penalise for necessary context. Penalise "
            "only off-topic content that adds no value.",
        ),
    ],
    "extraction": [
        Dimension(
            "faithfulness",
            "Faithfulness",
            "Faithfulness (0-10): Are all extracted entities actually present, verbatim "
            "or near-verbatim, in the input text? Penalise every invented entity or span "
            "that does not appear in the source.",
        ),
        Dimension(
            "coverage",
            "Coverage",
            "Coverage (0-10): Are all entities listed in the expected output also present "
            "in the extracted output? Score 10 only if no expected entity is missing. "
            "Penalise missed entities proportionally.",
        ),
        Dimension(
            "precision",
            "Precision",
            "Precision (0-10): Are entity type labels and span boundaries correct? "
            "Penalise wrong entity types, partial span matches (e.g., first name only), "
            "and merged or split spans.",
        ),
    ],
    "classification": [
        Dimension(
            "accuracy",
            "Accuracy",
            "Accuracy (0-10): Is the predicted label exactly correct? For single-label "
            "tasks: 10 = exact match, 0 = wrong label. For multi-label tasks: score "
            "proportionally to F1 between predicted and expected label sets.",
        ),
        Dimension(
            "justifiability",
            "Justifiability",
            "Justifiability (0-10): Given the input text, is the predicted label a "
            "defensible and consistent choice, even if it differs from the ground truth? "
            "Penalise labels that are clearly inconsistent with the input. Score 10 even "
            "for a wrong label if it is semantically adjacent and plausible.",
        ),
    ],
}


def get_dimensions(task_type: str) -> list[Dimension]:
    """Return the ordered list of scoring dimensions for a task type."""
    return DIMENSIONS.get(task_type, [])


def get_rubric(task, override: str | None = None) -> tuple[str, bool]:
    """Resolve the effective rubric for a task.

    Returns (rubric_string, overridden_flag).

    If a single override string is supplied it replaces the task-bank default
    for every task in the run. The override exists only for the duration of
    that run — nothing is persisted.
    """
    if override:
        return override, True
    return task.rubric, False


def load_overrides_yaml(path: str | Path) -> dict[str, str]:
    """Load a {task_id → rubric_string} map from a YAML file.

    Expected format::

        squad2_001: "Evaluate whether the answer is a verbatim span..."
        cnn_news_017: |
          Multi-line rubric.
          Award full marks only if all bullet points are present.

    Raises FileNotFoundError or ValueError on invalid input.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Rubric overrides file not found: {p}")
    with p.open() as fh:
        data = yaml.safe_load(fh) or {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Rubric overrides file must be a YAML mapping of "
            f"task_id → rubric string; got {type(data).__name__}"
        )
    return {str(k): str(v) for k, v in data.items()}


def merge_overrides(
    file_path: str | None,
    inline: dict[str, str] | None,
) -> dict[str, str]:
    """Merge file-based and inline overrides; inline takes precedence."""
    result: dict[str, str] = {}
    if file_path:
        try:
            result.update(load_overrides_yaml(file_path))
        except (FileNotFoundError, ValueError):
            pass
    if inline:
        result.update(inline)
    return result
