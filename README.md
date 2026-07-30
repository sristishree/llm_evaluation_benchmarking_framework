# LLM Evaluation Benchmarking Framework

Multi-provider LLM evaluation dashboard built with LiteLLM, FastAPI, and React. Models are evaluated across different task types (classification, NER, summarisation, QA) and difficulty levels, with scoring via ROUGE, BERTScore, Exact Match, and Token F1.

---

## Prerequisites

- Python 3.10+
- Node.js 18+
- API keys for the providers you want to test (OpenAI, Anthropic, Google) — or Ollama running locally for a fully offline setup

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Install frontend dependencies

```bash
cd dashboard/frontend
npm install
```

### 3. Configure environment variables

Copy the example env file and fill in your values:

```bash
cp .env.example .env
```

Edit `.env`:

```
LITELLM_BASE_URL=http://your-litellm-proxy:4000
LITELLM_API_KEY=sk-...
```

If you are calling providers directly (without a LiteLLM proxy), set their keys instead:

```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=...
```

### 4. Configure providers

Edit `config/providers.yaml` to enable the providers and models you want available in the UI. The file is pre-populated with examples for OpenAI (Azure), Google Gemini, and Ollama. Uncomment any block to activate it:

```yaml
anthropic:
  structured_output: true
  default_model: claude-sonnet-4-6
  available_models:
    - id: claude-sonnet-4-6
      display_name: "Claude Sonnet 4.6"
  # ...
```

---

## Running locally

The app has two processes that must run concurrently — the FastAPI backend and the Vite dev server.

### Terminal 1 — Backend (FastAPI)

Run from the **project root**:

```bash
uvicorn dashboard.api:app --reload --port 8000
```

The API will be available at `http://localhost:8000`.

### Terminal 2 — Frontend (React / Vite)

```bash
cd dashboard/frontend
npm run dev
```

Open `http://localhost:5173` in your browser. The Vite dev server proxies all `/api/*` requests to the backend on port 8000.

---

## Smoke test (no API keys needed)

To verify the pipeline end-to-end without making any real LLM calls:

```bash
python dry_run.py
```

This loads one task per task type, builds prompts, runs a placeholder inference step, scores the outputs, and prints a summary — all in-memory with no DB writes.

---

## Production build

To serve the frontend as static files from the FastAPI process instead of running two servers:

```bash
cd dashboard/frontend
npm run build
```

The compiled output goes to `dashboard/frontend/dist/`, which FastAPI serves automatically. Then run only the backend:

```bash
uvicorn dashboard.api:app --port 8000
```

Visit `http://localhost:8000`.
