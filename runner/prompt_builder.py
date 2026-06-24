from __future__ import annotations

from models.task import ClassificationTask, ExtractionTask, MediaAttachment, QATask, SummarizationTask, Task
from utils.media_utils import image_to_base64_url, pdf_to_text

_JSON_ONLY = (
    "Respond ONLY with a valid JSON object. "
    "No explanation, no markdown, no code fences — just the JSON."
)


def build_messages(task: Task, vision: bool = True) -> list[dict]:
    """Convert a Task into an OpenAI-style messages list.

    vision=False disables image blocks (text-only providers such as Ollama
    with a non-vision model). PDF attachments are always extracted as text
    regardless of this flag.
    """
    if isinstance(task, SummarizationTask):
        return _summarization(task, vision)
    if isinstance(task, ExtractionTask):
        return _extraction(task, vision)
    if isinstance(task, ClassificationTask):
        return _classification(task, vision)
    if isinstance(task, QATask):
        return _qa(task, vision)
    raise ValueError(f"Unsupported task type: {type(task)}")


# ------------------------------------------------------------------
# Task-specific builders
# ------------------------------------------------------------------

def _summarization(task: SummarizationTask, vision: bool) -> list[dict]:
    constraints = []
    if task.max_words:
        constraints.append(f"The summary must be at most {task.max_words} words.")
    if task.bullet_points:
        constraints.append("The summary must be formatted as bullet points.")

    system = (
        "You are an expert document summarizer. "
        "Produce a faithful, concise summary of the provided text.\n"
        + "".join(f"{c}\n" for c in constraints)
        + f"{_JSON_ONLY}\n"
        'Output format: {"summary": "<summary text>"}'
    )
    user_content = _build_user_content(
        f"Summarize the following text:\n\n{task.input}",
        task.media, vision,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]


def _extraction(task: ExtractionTask, vision: bool) -> list[dict]:
    types_str = ", ".join(task.entity_types)
    system = (
        "You are a named entity recognition (NER) system. "
        f"Extract all named entities from the text. Valid entity types: {types_str}.\n"
        "Return each entity's exact surface form and its type label. "
        "If no entities are present, return an empty list.\n"
        f"{_JSON_ONLY}\n"
        'Output format: {"entities": [{"text": "<entity text>", "label": "<TYPE>"}]}'
    )
    user_content = _build_user_content(
        f"Extract named entities from the following text:\n\n{task.input}",
        task.media, vision,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]


def _classification(task: ClassificationTask, vision: bool) -> list[dict]:
    labels_str = ", ".join(f'"{lb}"' for lb in task.label_set)
    system = (
        f"You are a text classification system. "
        f"Assign exactly one of these labels: {labels_str}.\n"
        "Only use labels from the provided list. Do not invent new labels.\n"
        f"{_JSON_ONLY}\n"
        'Output format: {"label": "<label>"}'
    )
    user_content = _build_user_content(
        f"Classify the following text:\n\n{task.input}",
        task.media, vision,
    )
    return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]


def _qa(task: QATask, vision: bool) -> list[dict]:
    span_note = (
        "Your answer must be a verbatim span copied directly from the provided context. "
        if task.answer_extractive else ""
    )
    system = (
        "You are a question-answering system. "
        f"{span_note}"
        "Answer the question based solely on the provided information. Be concise.\n"
        f"{_JSON_ONLY}\n"
        'Output format: {"answer": "<answer text>"}'
    )
    user_content = _build_user_content(task.input, task.media, vision)
    return [{"role": "system", "content": system}, {"role": "user", "content": user_content}]


# ------------------------------------------------------------------
# Media handling
# ------------------------------------------------------------------

def _build_user_content(
    text: str,
    media: list[MediaAttachment],
    vision: bool,
) -> str | list:
    """Build the user message content block.

    - PDF attachments are always extracted as text and prepended to `text`.
    - Image attachments are added as image_url blocks when vision=True;
      silently skipped when vision=False (text-only provider).
    - Returns a plain str when there are no images, a list otherwise
      (LiteLLM / OpenAI multimodal format).
    """
    # Extract all PDF text and prepend
    pdf_parts = [pdf_to_text(m.source) for m in media if m.type == "pdf"]
    full_text = "\n\n".join([*pdf_parts, text]) if pdf_parts else text

    # Collect image sources (only when provider supports vision)
    image_sources = [m.source for m in media if m.type == "image"] if vision else []

    if not image_sources:
        return full_text

    # Multimodal content list: text block first, then image blocks
    content: list[dict] = [{"type": "text", "text": full_text}]
    for src in image_sources:
        url = src if src.startswith("http") else image_to_base64_url(src)
        content.append({"type": "image_url", "image_url": {"url": url}})
    return content
