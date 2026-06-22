from __future__ import annotations

from models.task import ClassificationTask, ExtractionTask, QATask, SummarizationTask, Task

_JSON_ONLY = (
    "Respond ONLY with a valid JSON object. "
    "No explanation, no markdown, no code fences — just the JSON."
)


def build_messages(task: Task) -> list[dict[str, str]]:
    """Convert a Task into an OpenAI-style messages list."""
    if isinstance(task, SummarizationTask):
        return _summarization(task)
    if isinstance(task, ExtractionTask):
        return _extraction(task)
    if isinstance(task, ClassificationTask):
        return _classification(task)
    if isinstance(task, QATask):
        return _qa(task)
    raise ValueError(f"Unsupported task type: {type(task)}")


def _summarization(task: SummarizationTask) -> list[dict[str, str]]:
    constraints = []
    if task.max_words:
        constraints.append(f"The summary must be at most {task.max_words} words.")
    if task.bullet_points:
        constraints.append("The summary must be formatted as bullet points.")

    system = (
        "You are an expert document summarizer. "
        "Produce a faithful, concise summary of the provided text.\n"
        + ("".join(f"{c}\n" for c in constraints))
        + f"{_JSON_ONLY}\n"
        'Output format: {"summary": "<summary text>"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Summarize the following text:\n\n{task.input}"},
    ]


def _extraction(task: ExtractionTask) -> list[dict[str, str]]:
    types_str = ", ".join(task.entity_types)
    system = (
        "You are a named entity recognition (NER) system. "
        f"Extract all named entities from the text. Valid entity types: {types_str}.\n"
        "Return each entity's exact surface form and its type label. "
        "If no entities are present, return an empty list.\n"
        f"{_JSON_ONLY}\n"
        'Output format: {"entities": [{"text": "<entity text>", "label": "<TYPE>"}]}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Extract named entities from the following text:\n\n{task.input}"},
    ]


def _classification(task: ClassificationTask) -> list[dict[str, str]]:
    labels_str = ", ".join(f'"{lb}"' for lb in task.label_set)
    if task.multi_label:
        instruction = f"Assign one or more of these labels: {labels_str}."
        out_fmt = '{"labels": ["<label1>", "<label2>"]}'
    else:
        instruction = f"Assign exactly one of these labels: {labels_str}."
        out_fmt = '{"label": "<label>"}'

    system = (
        f"You are a text classification system. {instruction}\n"
        "Only use labels from the provided list. Do not invent new labels.\n"
        f"{_JSON_ONLY}\n"
        f"Output format: {out_fmt}"
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": f"Classify the following text:\n\n{task.input}"},
    ]


def _qa(task: QATask) -> list[dict[str, str]]:
    span_note = (
        "Your answer must be a verbatim span copied directly from the provided context. "
        if task.answer_extractive
        else ""
    )
    system = (
        "You are a question-answering system. "
        f"{span_note}"
        "Answer the question based solely on the provided information. "
        "Be concise.\n"
        f"{_JSON_ONLY}\n"
        'Output format: {"answer": "<answer text>"}'
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": task.input},
    ]
