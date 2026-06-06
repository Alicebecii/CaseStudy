# Agentic Multi-Modal Document Q&A

An agent that answers questions over long PDF documents by navigating their
structure, retrieving evidence, and verifying its own answers. See
[DESIGN.md](DESIGN.md) for the architecture and the reasoning behind each choice,
and [TECHNICAL_NOTE.md](TECHNICAL_NOTE.md) for the two most important design decisions.

Instead of stuffing top-k chunks into one prompt (classic RAG), a from-scratch
ReAct agent reads the way a person does: it skims the table of contents, searches
for evidence, reads the relevant section, then answers with inline `[chunk_id]`
citations - and a separate validator checks that those citations actually hold up.

## Install

Requires Python 3.10+. The base install is light; heavier features are opt-in extras.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dense,dev]"      # dense retrieval + test tools
# optional backends:
pip install -e ".[ollama]"          # local models via Ollama
pip install -e ".[openai]"          # the OpenAI API
```

With no extras the system still runs: it falls back to BM25-only retrieval and the
mock LLM backend, so it works offline with no model download and no API key.

## Run

```bash
agentic-extract --pdf path/to/document.pdf --question "Your question?"
# show the agent's step-by-step reasoning:
agentic-extract --pdf path/to/document.pdf --question "Your question?" --show-trace
# dump the document's structured outline as JSON (no question needed):
agentic-extract --pdf path/to/document.pdf --outline-json
# enable cross-run memory: recall a prior grounded answer to a near-duplicate
# question (served instantly), and record each outcome for next time:
agentic-extract --pdf path/to/document.pdf --question "..." --memory mem.jsonl
```

The command prints the answer, its sources (cited chunk → page and section), and a
validation verdict (`grounded` / `weak` / `ungrounded`) with a confidence score.

### Choosing a backend

The backend is selected by environment variables (every value has a safe default;
see [.env.example](.env.example)). Copy it to `.env` and adjust, or set inline:

```bash
# Local model on Ollama (free, runs on a GPU):
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5 agentic-extract --pdf doc.pdf --question "..."

# OpenAI (used for a stronger final-verification run):
LLM_PROVIDER=openai OPENAI_API_KEY=sk-... agentic-extract --pdf doc.pdf --question "..."

# Mock backend (default): deterministic, offline - for wiring/CI, not real answers.
agentic-extract --pdf doc.pdf --question "..."
```

Optional capabilities, all opt-in and off by default:
`ENABLE_SPECIALIST=1` adds a specialist sub-agent the main agent can delegate to;
`OLLAMA_VISION_MODEL=llava` (after `ollama pull llava`) adds a `view_page` tool that renders
a page and asks a vision model about figures/tables; `--memory PATH` enables cross-run memory.

## Demo

[`examples/demo_output.md`](examples/demo_output.md) is a real, unedited run of the
agent (on `qwen2.5` via Ollama) over two documents - the Turkish assignment brief and
an open-access English paper - with three questions each. Regenerate it with:

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5 python examples/run_demo.py
```

The validation verdicts there are strict and honest: the deterministic check certifies
an answer only when its cited chunks' text actually supports the claims, so it flags
loosely-paraphrased, mis-cited, or cross-lingual answers. That conservatism is the
reliability layer working as designed (see the note at the top of the demo file).

## Test

```bash
pytest            # ~90 unit tests, fully offline (mock backend, fake embedder/clients)
```

The suite needs no network, no API key, and no model download: the LLM and embedder
are exercised through deterministic fakes, so it is fast and reproducible.

## Layout

```
src/agentic_extraction/
  core/          shared data shapes, interfaces, and errors (depended on by all)
  config.py      typed settings read from the environment
  preprocessing/ PDF parsing, outline, section-aware chunking
  retrieval/     BM25, dense, hybrid (RRF) fusion behind one Retriever
  llm/           pluggable LLM backends (mock, ollama, openai) + tool protocol
  agent/         four navigation tools and the from-scratch ReAct orchestrator
  validator/     deterministic grounding check + optional LLM critic
  memory/        cross-run memory store (recall/record, opt-in via --memory)
  evaluation.py  scoring of answers against a gold Q&A set
  cli.py         command line entry point
examples/        run_demo.py + captured demo_output.md (+ a CC-BY sample paper)
tests/           offline unit tests for every layer
```

Every swappable part sits behind a small contract in `core/` and is wired in one
place (a per-package factory), so an implementation can be added, removed, or
rewritten without rippling across callers. See [DESIGN.md](DESIGN.md) for the full
rationale and trade-offs.
