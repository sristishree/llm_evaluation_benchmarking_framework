# LLM Evaluation Benchmarking Framework

**Multi-Provider NLP Evaluation** — Version 0.1 | May 2025

A reproducible, multi-provider NLP benchmark that runs standardised tasks across Claude, GPT-4o, Gemini (and optionally Ollama), scores outputs with layered metrics, and surfaces results in an interactive dashboard.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Architecture](#architecture)
- [Build Phases](#build-phases)
- [Task Bank Design](#task-bank-design)
- [Key Principles & Pitfalls](#key-principles--pitfalls)
- [Areas to Explore](#areas-to-explore)
- [Recommended Stack](#recommended-stack)
- [Open Questions & Brainstorm](#open-questions--brainstorm)

---

## Project Overview

The goal is demonstrating evaluation rigour, infrastructure thinking, and NLP domain depth across multiple LLM providers. Each stage of the pipeline is independently testable with a clean interface to its neighbours.

---

## Architecture

```
Task Bank -> Prompt Runner -> LLM Providers -> Scorer -> Results Store -> Dashboard
```

The runner, scorer, and storage are three distinct modules — a deliberate choice that mirrors production ML pipeline design.

---

## Build Phases

### Phase 1 — Task Bank Design

Four task types:
- Summarization
- Named Entity Extraction
- Classification
- Q&A

Each task carries: `task_id`, `task_type`, `difficulty`, `domain`, `input`, `expected output`, `rubric`, `metadata`

**Datasets:** CNN/DailyMail, CoNLL-2003, SQuAD 2.0

**Difficulty tiers:** easy / medium / hard — reveal how providers degrade under pressure

---

### Phase 2 — Prompt Runner

A task-aware abstraction: (task + provider config) in, parsed validated response out.

- Structured output parsing with Pydantic models per task type
- Retry logic, timeout handling, dry-run mode for cost control
- Log all generation config: model version, temperature, `max_tokens` for reproducibility

---

### Phase 3 — Provider Integration

Using **LiteLLM** as the unified client across all providers.

**Minimum providers:** Claude (Anthropic), GPT-4o (OpenAI), Gemini (Google)
**Optional:** Local SLM via Ollama to demonstrate cost-quality tradeoffs

Use `response_format: json_schema` or tool use for structured outputs where supported.

**Input format support by task type:**

| Task Type | Text | PDF | Image |
|-----------|------|-----|-------|
| Summarization | ✅ | ✅ | — |
| NER | ✅ | — | ✅ |
| Classification | ✅ | — | ✅ |
| Q&A | ✅ | ✅ | ✅ |

---

### Phase 4 — Scoring Layer

Layered scoring is the differentiator — do not collapse to a single score early.

| Layer | Metric | Tasks |
|-------|--------|-------|
| Lexical | ROUGE-1/2/L | Summarization |
| Semantic | BERTScore | All generative tasks |
| LLM-as-Judge | Rubric dimensions (faithfulness, coverage, conciseness) | All |
| Exact match / token F1 | Extraction and classification | NER, Classification |

**Key rule:** Scorer is decoupled from runner — re-scoring must not require re-running inference.

---

### Phase 5 — Results Store

SQLite + structured JSON, keyed by `(task_id, provider, model_version, timestamp)`.

Schema must support queries like: *"How did Claude Sonnet perform on summarization vs GPT-4o last week?"*

---

### Phase 6 — Dashboard

Streamlit (fast) or React (more polished).

- Per-provider score breakdowns by task type and difficulty
- Task-level performance heatmaps (provider × difficulty)
- Cost vs quality scatter plot
- Side-by-side response viewer for qualitative review

---

## Task Bank Design

### Core Schema

| Field | Description |
|-------|-------------|
| `task_id` | Unique, stable identifier (e.g. `summ_news_001`) |
| `task_type` | `summarization` \| `extraction` \| `classification` \| `qa` |
| `difficulty` | `easy` \| `medium` \| `hard` |
| `domain` | e.g. news, legal, biomedical, product_reviews |
| `input` | Raw text passed to the model |
| `expected` | Ground truth output (format varies by task type) |
| `rubric` | Scoring instructions, especially for LLM-as-judge |
| `metadata` | source, date_added, notes |

Use Pydantic subclasses per task type so `expected` is properly typed: `str` for summarization, `Dict[str, List[str]]` for extraction, etc.

### Task Types, Expected Formats, and Metrics

| Task Type | Expected Format | Primary Metrics | LLM-Judge Dimensions |
|-----------|----------------|-----------------|----------------------|
| Summarization | Reference summary (`str`) | ROUGE-1/2/L, BERTScore | Faithfulness, Coverage, Conciseness |
| NER Extraction | `Dict[type, List[str]]` | Token F1 per entity type | Hallucination penalty |
| Classification | Label + confidence tier | Exact match, Macro F1 | Confidence calibration |
| Q&A | Answer `str` or abstention | EM, F1, BERTScore | Grounding, Abstention accuracy |

### Ground Truth Sourcing

| Task Type | Base Dataset | Hard Examples |
|-----------|-------------|---------------|
| Summarization | CNN/DailyMail *(reference summaries are noisy — document this)* | 5–10 (legal, biomedical) |
| NER Extraction | CoNLL-2003 | 5–10 (domain-specific entity types) |
| Classification | SST-2 (sentiment); BIG-Bench Hard (reasoning) | 5–10 adversarial phrasing |
| Q&A | SQuAD 2.0 *(includes unanswerable questions — critical for hallucination testing)* | 5–10 unanswerable cases |

### Difficulty Design

Most benchmarks fail because tasks are too easy — every provider scores 85–95% and results are flat. Deliberately include tasks that stress-test models: long inputs, domain jargon, adversarial phrasing, unanswerable questions, implicit sentiment. **Score spread is what makes the analysis worth reading.**

### Storage Format

Task bank files are stored as **JSONL** (one task per line) — supports streaming reads, easy appending, and line-level validation without loading the full file into memory.

### Versioning

Treat the task bank as a versioned dataset. Use semantic versioning (`v1.0`, `v1.1`) with a `CHANGELOG.md`. This makes results reproducible across time and signals MLOps thinking.

---

## Key Principles & Pitfalls

### Reproducibility
- Seed all runs; log model version, temperature, `max_tokens`, `top_p` for every call
- Pin provider SDK versions in `pyproject.toml`
- A benchmark that cannot be reproduced is not a benchmark

### Separation of Concerns
- Runner, scorer, and storage are three distinct modules with defined interfaces
- Re-scoring must be possible without re-running inference

### Cost Control
- Implement a task sampler before running at scale
- Budget roughly $10–15 for a full demo run across three providers if careful
- Log token counts per call and include estimated cost in the dashboard

### LLM-as-Judge Calibration
- Test rubric prompt stability against human judgements on a small slice before trusting at scale
- Known biases to mitigate: position bias, verbosity bias, self-enhancement bias
- Use a different provider as judge than the one being evaluated where possible

### Statistical Rigour
- Define metrics before running — post-hoc metric selection is p-hacking
- Include bootstrap confidence intervals on score differences between providers

---

## Areas to Explore

- **LLM-as-Judge (Zheng et al. 2023)** — Foundational paper on judge calibration. Covers position bias, verbosity bias, and self-enhancement bias. Essential reading before writing rubric prompts.
- **RAGAS** — Standard RAG evaluation framework. Overlaps conceptually even without a retrieval layer — worth understanding the scoring methodology.
- **DeepEval** — Scoring scaffolding framework that handles much of the metric infrastructure. Decide early whether to use it or build raw — building raw is more impressive for a portfolio.
- **MT-Bench and HELM** — Existing benchmarks worth studying for task design inspiration and difficulty calibration, even if you build your own.
- **Structured Outputs** — Explore `response_format: json_schema` (OpenAI-compatible APIs) and Claude tool use for constrained extraction tasks. Existing Pydantic fluency transfers directly here.
- **Bootstrap Confidence Intervals** — Even a simple bootstrap CI on score differences between providers elevates the analysis considerably from a statistical credibility standpoint.

---

## Recommended Stack

| Layer | Choice |
|-------|--------|
| Provider client | LiteLLM |
| Validation & schemas | Pydantic v2 — subclass per task type for typed `expected` fields |
| Lexical scoring | `rouge-score` |
| Semantic scoring | `bert-score` |
| Storage | SQLite + pandas |
| Dashboard | Streamlit (fast) or React (more polished) |
| Config | YAML + `pydantic-settings` |
| Dependency management | Poetry + Python 3.11 |

---

## Open Questions & Brainstorm

> This section is for ongoing brainstorming. Add ideas freely.

- [ ] **Ollama integration** — Include local SLM via Ollama to demonstrate cost-quality tradeoffs?
- [ ] **PDF input for Summarization** — Requires a PDF parser (e.g. unstructured/pymupdf). Worth the complexity for this project?
- [ ] **Q&A input formats** — Are all input formats (text, PDF, image) useful for Q&A, or overkill?
- [ ] **LiteLLM vs alternatives** — Explore multiple client providers before settling; justify which is the best option. If something is better than LiteLLM and feasible, use it.
- [ ] **Storage alternatives** — Explore options better than SQLite but still feasible (e.g. DuckDB, TinyDB, hosted Postgres)?
- [ ] **Dataset auto-update agent** — An agent that periodically updates/refreshes the task bank datasets?
- [ ] **RAG over results** — Save all reports to a DB and use an LLM to answer analysis questions in Q&A format. (Future project idea.)
- [ ] **BYOK (Bring Your Own Key)** — Explore integration for multi-user or demo scenarios.
- [ ] **Multi-domain task tagging** — `infer_domain` currently returns the first keyword match and stops. Documents covering multiple topics (e.g. political corruption spanning politics + crime) get a single label. Consider returning `list[str]` for `domain` or storing secondary matches in `metadata` for richer filtering and stratified sampling.
- [ ] **Global hosting & public leaderboard** — Host the dashboard globally (Railway, Render, or HF Spaces) rather than locally. A hosted URL on a resume/LinkedIn is far more effective than a repo that requires local setup. Cloud providers (Claude, GPT-4o, Gemini) work out of the box; Ollama users would need to run locally or expose via a tunnel — document this clearly rather than trying to solve it.
  - **Opt-in result sharing:** when a user runs a benchmark, let them opt in to making results public. Store model name, provider, task type, difficulty, scores per metric, timestamp, and whether the default or custom task bank was used — no API keys, no PII. Add a disclaimer: *"Results are user-submitted and not independently verified."*
  - **Leaderboard aggregation:** top-level view shows aggregate scores per model (e.g. "GPT-4o: avg summarization ROUGE-L 0.42 across 34 runs") with individual runs drillable. Filter by task type, difficulty tier, domain, and date range.
  - **Custom vs default task bank results:** tag results so default-bank runs remain comparable across users; custom-bank runs are shown separately or filterable.
- [ ] **User-provided task bank (custom mode)** — Two modes: *default* (curated task bank, comparable results, feeds public leaderboard) and *custom* (user uploads their own tasks in JSONL/JSON/YAML, validated against the Pydantic task schema on upload with immediate error feedback). Key UX considerations:
  - Let users set difficulty per task or leave blank — infer it via a quick LLM call.
  - Let users mix default + custom tasks in a single run to benchmark their data relative to the standard set.
  - Custom results on the public leaderboard are tagged and filterable so they don't pollute default benchmark comparisons.
