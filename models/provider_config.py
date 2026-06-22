from __future__ import annotations

from pydantic import BaseModel


class ModelInfo(BaseModel):
    id: str               # LiteLLM model string — used verbatim in API calls
    display_name: str     # human-readable label for UI dropdowns


class ProviderConfig(BaseModel):
    """Runtime configuration for a single benchmark run.

    Built by the UI from user form input (provider + model selection +
    optional parameter overrides). API keys are NOT part of this model —
    they are resolved server-side from environment variables.
    """
    provider: str
    model: str
    temperature: float = 0.0
    max_tokens: int = 1024
    top_p: float = 1.0
    timeout_seconds: int = 60
    max_retries: int = 3
