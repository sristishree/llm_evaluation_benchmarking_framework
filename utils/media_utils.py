"""Utilities for loading PDF and image inputs into LLM-ready formats.

PDF  → plain text via pymupdf (works with all providers)
Image → base64 data-URL (works with all vision-capable providers via LiteLLM)
"""
from __future__ import annotations

import base64
from pathlib import Path


# ------------------------------------------------------------------
# PDF
# ------------------------------------------------------------------

def pdf_to_text(path: str) -> str:
    """Extract all text from a PDF file, one page separated by blank lines."""
    try:
        import fitz  # pymupdf
    except ImportError:
        raise ImportError(
            "pymupdf is required for PDF support. Install it with: pip install pymupdf"
        )
    doc = fitz.open(path)
    pages = [page.get_text().strip() for page in doc]
    return "\n\n".join(p for p in pages if p)


# ------------------------------------------------------------------
# Image
# ------------------------------------------------------------------

_MIME_MAP = {
    "jpg": "image/jpeg",
    "jpeg": "image/jpeg",
    "png": "image/png",
    "gif": "image/gif",
    "webp": "image/webp",
}


def image_to_base64_url(path: str) -> str:
    """Encode a local image file as a base64 data-URL for multimodal LLM calls."""
    p = Path(path)
    mime = _MIME_MAP.get(p.suffix.lower().lstrip("."), "image/png")
    b64 = base64.b64encode(p.read_bytes()).decode()
    return f"data:{mime};base64,{b64}"
