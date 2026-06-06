# Testing & Running - a reviewer's guide

A step-by-step walkthrough to verify this project from a clean clone, in tiers - from a
**30-second offline check** to **full real-model runs**. Pick the depth you want.

> **Shortest path:** open [`examples/demo_output.md`](examples/demo_output.md) - a real,
> unedited multi-question run is already captured there. Or run the offline test suite
> (§2a): **118 tests, ~1 second, no model needed.**

---

## 0. Prerequisites

- **Python 3.10+**
- *(Optional, only for real answers)* **[Ollama](https://ollama.com)** with a model pulled,
  or an OpenAI API key.
- No API key or model is needed for the offline checks.

## 1. Setup (~1-2 min)

```bash
git clone https://github.com/Alicebecii/CaseStudy.git
cd CaseStudy
python -m venv .venv && source .venv/bin/activate

# Light install - enough for the tests + offline runs + BM25 retrieval:
pip install -e ".[dev]"

# OR full install - also adds dense (semantic) retrieval. Downloads PyTorch (~hundreds of MB):
pip install -e ".[dense,dev]"

# Add ".[ollama]" to use a local model, ".[openai]" for the OpenAI API.
```

The system runs with **no extras at all** (it falls back to BM25 retrieval + a mock LLM),
so the light install is enough to exercise everything except real LLM answers.

## 2. Offline checks - no model, instant

### a) Run the test suite
```bash
pytest
```
Expect all green: **118 passed**, or **117 passed + 1 skipped** on the light `.[dev]`
install (the one skip needs the optional Ollama dependency - add `.[ollama]` for the full
118). Fully offline: the LLM and embedder are deterministic fakes, so there is no network,
no API key, and no model download.

### b) Extract a document's structured outline (real output, no model)
```bash
agentic-extract --pdf examples/data/peerj-cs-1097.pdf --outline-json | head -30
```
Prints the paper's section tree as nested JSON.

### c) Run the whole pipeline with the mock backend (wiring check)
```bash
agentic-extract --pdf examples/data/peerj-cs-1097.pdf --question "What is this paper about?"
```
The mock backend returns a placeholder answer - this just confirms the pipeline runs
end-to-end. For real answers, continue to §3.

## 3. Real answers with a local model (Ollama)

```bash
ollama serve &            # if it isn't already running
ollama pull qwen2.5       # one-time, ~4.7 GB

# Env vars select the backend; set them inline (or `export` them):
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5 \
  agentic-extract --pdf examples/data/peerj-cs-1097.pdf \
  --question "What framework does the paper propose?" --show-trace
```

Expected shape of the output:

```
The paper proposes a framework that organises the reviewed methods by the
objective they pursue [c79] ...

Sources:
  [c79] p14  FRAMEWORK > Framework components
  [c93] p16  FRAMEWORK > Framework usage

Validation: grounded (confidence 0.83, attempts 1)

Trace:
  step 1: get_outline() -> ok
  step 2: search(query='framework') -> ok
  step 3: final answer
```

Prefer OpenAI?  `LLM_PROVIDER=openai OPENAI_API_KEY=sk-... OPENAI_MODEL=gpt-4o-mini agentic-extract ...`

## 4. Demo & evaluation

```bash
# Regenerate the captured demo (2 PDFs × 3 questions) -> examples/demo_output.md
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5 python examples/run_demo.py

# Accuracy measurement over a gold Q&A set -> per-question table + pass_rate / grounded_rate
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5 python examples/evaluate.py
```

## 5. Bonus features (all opt-in, off by default)

Prefix each with your backend vars (`LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5`).

```bash
# Cross-run memory - ask the SAME question twice; the 2nd is served instantly (attempts 0)
# if the 1st was grounded:
agentic-extract --pdf examples/data/peerj-cs-1097.pdf --question "..." --memory mem.jsonl

# Multi-agent - the agent can delegate a sub-question to a specialist sub-agent:
ENABLE_SPECIALIST=1 agentic-extract --pdf examples/data/peerj-cs-1097.pdf --question "..." --show-trace

# Vision - render a page and ask a vision model about figures/tables:
ollama pull llava
OLLAMA_VISION_MODEL=llava agentic-extract --pdf examples/data/peerj-cs-1097.pdf \
  --question "Describe the diagram on page 14"
```

## 6. How to read the output

- **Answer** - prose with inline `[chunk_id]` citations.
- **Sources** - each cited chunk resolved to its **page** and **section**.
- **Validation** - `grounded` / `weak` / `ungrounded`, with a confidence score and how many
  attempts the agent took.

The validator is **deliberately strict and honest**: it certifies `grounded` only when the
cited chunks' text actually supports the claims, so it flags loosely-paraphrased, mis-cited,
or cross-lingual answers as `weak`/`ungrounded` - usually a sign the (small) local model's
citations didn't fully hold up, *not* that the facts are wrong. The evaluation harness (§4)
reports **accuracy** and **grounding** as separate numbers for exactly this reason.

## 7. Troubleshooting

| Symptom | Fix |
|---|---|
| `connection refused` / no answer with Ollama | Ollama isn't running → `ollama serve` |
| Model errors / `model not found` | Pull it first → `ollama pull qwen2.5` |
| First dense run is slow | It downloads the embedding model once (~80 MB); or use `pip install -e ".[dev]"` for BM25-only |
| `--question is required` | Pass `--question "..."`, or use `--outline-json` (no question needed) |
| Env vars seem ignored | Set them **inline** before the command (`LLM_PROVIDER=ollama agentic-extract ...`) or `export` them first - there is no `.env` auto-loading |

---

See [README.md](README.md) for the architecture and [DESIGN.md](DESIGN.md) for the reasoning
behind each design choice.
