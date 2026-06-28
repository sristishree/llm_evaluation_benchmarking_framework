from __future__ import annotations

from models.provider_config import ProviderConfig
from models.result import RunResult
from models.scored_result import ScoredResult
from models.task import Task
from evaluators.exact_match import (
    classification_exact_match,
    entity_f1,
    qa_exact_match,
    qa_token_f1,
)
from evaluators.lexical import score_rouge
from evaluators.llm_judge import (
    judge_classification,
    judge_extraction,
    judge_qa,
    judge_summarization,
)
from evaluators.semantic import score_bert


class Scorer:
    """Evaluate a RunResult against its Task using layered metrics.

    Metrics applied by task type:
      summarization  — ROUGE-1/2/L, BERTScore, LLM-as-judge (faithfulness, coverage, conciseness)
      extraction     — micro/per-label entity F1, LLM-as-judge (hallucination penalty)
      classification — exact match, LLM-as-judge (correctness)
      qa             — exact match, token F1, BERTScore, LLM-as-judge (grounding, abstention)

    The scorer never calls the runner — pass stored RunResult objects to re-score
    without re-running inference.

    Args:
        judge_config: Provider to use as LLM judge. Should differ from the provider
                      being evaluated to avoid self-enhancement bias. If None, the
                      LLM-as-judge layer is skipped.
        run_bert:     Whether to compute BERTScore. Disable to skip the model
                      download and speed up scoring in CI / dry-run contexts.
    """

    def __init__(
        self,
        judge_config: ProviderConfig | None = None,
        run_bert: bool = True,
    ) -> None:
        self.judge_config = judge_config
        self.run_bert = run_bert

    def score(self, task: Task, result: RunResult) -> ScoredResult:
        """Score a single RunResult against its Task."""
        metrics: dict[str, float] = {}
        judge_reasoning: str | None = None

        if result.parsed_output is None:
            return ScoredResult(
                task_id=result.task_id,
                task_type=result.task_type,
                provider=result.provider,
                model=result.model,
                metrics=metrics,
                judge_reasoning="skipped: no parsed output",
            )

        if task.task_type == "summarization":
            output = result.parsed_output  # str
            metrics |= score_rouge(task.expected, output)
            if self.run_bert:
                metrics |= score_bert(task.expected, output)
            if self.judge_config:
                j_scores, judge_reasoning = judge_summarization(task, output, self.judge_config)
                metrics |= j_scores

        elif task.task_type == "extraction":
            output = result.parsed_output  # list[dict]
            metrics |= entity_f1(output, task.expected)
            if self.judge_config:
                j_scores, judge_reasoning = judge_extraction(task, output, self.judge_config)
                metrics |= j_scores

        elif task.task_type == "classification":
            output = result.parsed_output  # str
            metrics["exact_match"] = classification_exact_match(output, task.expected)
            if self.judge_config:
                j_scores, judge_reasoning = judge_classification(task, output, self.judge_config)
                metrics |= j_scores

        elif task.task_type == "qa":
            output = result.parsed_output  # str
            expected_list = (
                task.expected if isinstance(task.expected, list) else [task.expected]
            )
            metrics["exact_match"] = qa_exact_match(output, expected_list)
            metrics["token_f1"] = round(qa_token_f1(output, expected_list), 4)
            if self.run_bert:
                best_ref = max(expected_list, key=len)
                metrics |= score_bert(best_ref, output)
            if self.judge_config:
                j_scores, judge_reasoning = judge_qa(task, output, self.judge_config)
                metrics |= j_scores

        return ScoredResult(
            task_id=result.task_id,
            task_type=result.task_type,
            provider=result.provider,
            model=result.model,
            metrics=metrics,
            judge_reasoning=judge_reasoning,
        )
