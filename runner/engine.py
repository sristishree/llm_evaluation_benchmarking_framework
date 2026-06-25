"""Top-level entry points for running tasks against LLM providers.

Typical usage
-------------
    from runner.engine import run_task, run_batch, get_catalog
    from models.provider_config import ProviderConfig

    cfg = ProviderConfig(provider="anthropic", model="claude-sonnet-4-6")

    result  = run_task(task, cfg)
    results = run_batch(tasks, cfg, dry_run=True)

    # Inspect the catalog (drives UI dropdowns)
    catalog = get_catalog()
    models  = catalog["anthropic"]["available_models"]   # list of ModelInfo
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

import yaml

from models.provider_config import ModelInfo, ProviderConfig
from models.result import RunResult
from models.task import Task
from runner.llm_client import LLMClient

_CONFIG_PATH = Path(__file__).parent.parent / "config" / "providers.yaml"


def _load_yaml() -> dict:
    with _CONFIG_PATH.open() as fh:
        return yaml.safe_load(fh)


# ------------------------------------------------------------------
# Catalog helpers (used by the UI to populate dropdowns)
# ------------------------------------------------------------------

def get_catalog() -> dict[str, dict]:
    """Return the full provider catalog parsed from providers.yaml.

    Structure:
        {
          "anthropic": {
            "default_model": "claude-sonnet-4-6",
            "available_models": [ModelInfo(id=..., display_name=...), ...],
            "defaults": {"temperature": 0.0, ...},
          },
          ...
        }
    """
    raw = _load_yaml()
    catalog = {}
    for provider, entry in raw.items():
        catalog[provider] = {
            "default_model": entry["default_model"],
            "available_models": [ModelInfo(**m) for m in entry["available_models"]],
            "defaults": entry["defaults"],
        }
    return catalog


def available_providers() -> list[str]:
    """Return the list of supported provider names."""
    return list(_load_yaml().keys())


# ------------------------------------------------------------------
# Run helpers
# ------------------------------------------------------------------

def _make_client(config: ProviderConfig) -> LLMClient:
    raw = _load_yaml()
    entry = raw.get(config.provider, {})
    return LLMClient(
        provider=config.provider,
        model=config.model,
        temperature=config.temperature,
        max_tokens=config.max_tokens,
        top_p=config.top_p,
        timeout_seconds=config.timeout_seconds,
        max_retries=config.max_retries,
        structured_output=entry.get("structured_output", True),
        api_base=os.environ.get("LITELLM_BASE_URL"),
        vision=entry.get("vision", True),
    )


def run_task(task: Task, config: ProviderConfig, dry_run: bool = False) -> RunResult:
    """Run a single task with the given provider configuration."""
    return _make_client(config).call(task, dry_run=dry_run)


def run_batch(
    tasks: Iterable[Task],
    config: ProviderConfig,
    dry_run: bool = False,
) -> list[RunResult]:
    """Run tasks sequentially, reusing a single LLMClient instance."""
    client = _make_client(config)
    results = []
    for i, task in enumerate(tasks):
        result = client.call(task, dry_run=dry_run)
        results.append(result)
        status = "DRY RUN" if dry_run else f"{result.latency_ms:.0f}ms"
        print(f"  [{i+1:>4}] {task.task_id}  |  {status}")
    return results
