# Agentic Multi-Modal Document Q&A

Answer questions over long PDFs with an agent that **reads like a person** - it skims the
table of contents, searches for evidence, reads the relevant section, then answers with
inline `[chunk_id]` citations - and a **separate validator** checks that those citations
actually hold up before the answer is trusted.

> Not classic RAG. Instead of stuffing top-k chunks into one prompt and hoping, a
> from-scratch ReAct loop *decides* what to look at, and grounding is verified, not assumed.

```
                  ┌──────────────┐      ┌─────────────────────┐
   PDF ─────────▶ │ PREPROCESSING│ ───▶ │  HYBRID RETRIEVAL   │
                  │ parse·outline│      │  BM25 + dense (RRF) │
                  │ ·chunk       │      └──────────┬──────────┘
                  └──────────────┘                 ▲ search()
                                                   │
   question ───────────────────────▶  ┌────────────┴────────────────────┐
                                       │  AGENT - from-scratch ReAct loop │
                                       │  think → act → observe → answer  │
                                       │  tools: get_outline · search ·   │
                                       │         read_section · read_page │
                                       └────────────┬─────────────────────┘
                                                    │ answer + [chunk_id] citations
                                                    ▼
                                       ┌──────────────────────────────────┐
                                       │  VALIDATOR                        │
                                       │  deterministic grounding + LLM    │
                                       │  critic → grounded? else 1 retry  │
                                       └────────────┬─────────────────────┘
                                                    ▼
                              answer · sources (chunk → page · section) · verdict
```

Every swappable part (LLM, embedder, retriever, validator, memory) sits behind a small
contract in `core/` and is wired in one place - so a backend can be added, removed, or
rewritten without rippling across callers. See **[DESIGN.md](DESIGN.md)** for the full
rationale and trade-offs, and **[TECHNICAL_NOTE.md](TECHNICAL_NOTE.md)** for the two key
decisions.

## Capabilities

| Required (MVP) | | Bonus | |
|---|---|---|---|
| PDF input + text extraction | ✅ | Structured outline → JSON | ✅ |
| Retrieval (BM25 + dense, RRF) | ✅ | Evaluation harness (accuracy) | ✅ |
| Agent loop with tool-calling | ✅ | Cross-run memory | ✅ |
| Validation layer | ✅ | Multi-agent (specialist sub-agent) | ✅ |
| CLI (`--pdf` `--question`) | ✅ | Visual content (vision tool) | ✅ |

Runs offline with zero setup (mock backend + BM25), or with a free local model (Ollama)
or the OpenAI API by changing one environment variable.

## Quickstart

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dense,dev]"          # base + dense retrieval + tests
#   ...add ".[ollama]" for local models, ".[openai]" for the OpenAI API

# Ask a question (local model on Ollama):
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5 \
  agentic-extract --pdf paper.pdf --question "What framework does the paper propose?"
```

### Example output

```
The paper proposes a framework that organises the reviewed methods by the
objective they pursue [c79], and shows how it can explore methods from the
perspective of objectives, targets, and prescribed processes [c93].

Sources:
  [c79] p14  FRAMEWORK > Framework components
  [c93] p16  FRAMEWORK > Framework usage

Validation: grounded (confidence 0.83, attempts 1)
```

Add `--show-trace` to see the agent's steps (`get_outline → search → final answer`).
A real, unedited multi-question run over two PDFs lives in
**[examples/demo_output.md](examples/demo_output.md)**.

## Backends

Selected by environment variables (every value has a safe default - see
[.env.example](.env.example)):

```bash
LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5  agentic-extract ...   # free, local, on a GPU
LLM_PROVIDER=openai OPENAI_API_KEY=sk-...  agentic-extract ...   # hosted, for verification
#                              (default)    agentic-extract ...   # mock: offline, for CI
```

## Other commands & optional features

```bash
# Structured outline as JSON (no model needed):
agentic-extract --pdf paper.pdf --outline-json

# Cross-run memory: a prior grounded answer to a near-duplicate question is served instantly:
agentic-extract --pdf paper.pdf --question "..." --memory mem.jsonl

# Accuracy measurement over a gold Q&A set:
python examples/evaluate.py            # uses examples/gold.json

# Multi-agent: the main agent can delegate a sub-question to a specialist sub-agent:
ENABLE_SPECIALIST=1 agentic-extract --pdf paper.pdf --question "..."

# Vision: render a page and ask a vision model about figures/tables (after `ollama pull llava`):
OLLAMA_VISION_MODEL=llava agentic-extract --pdf paper.pdf --question "Describe the diagram on page 14"
```

All optional features are off by default - with no flags or env vars, behaviour is identical
to the base system.

## Reading the validation verdict

The verdict is **deliberately strict and honest**: the deterministic check certifies an
answer (`grounded`) only when its cited chunks' text actually supports the claims, so it
flags loosely-paraphrased, mis-cited, or cross-lingual answers as `weak`/`ungrounded` - and
on a failure the agent gets one bounded retry. That conservatism *is* the reliability layer
working; refusing to over-claim is a feature, not a bug.

## Test

```bash
pytest          # 118 tests, fully offline - no network, no API key, no model download
```

The LLM and embedder are exercised through deterministic fakes, so the suite is fast and
reproducible everywhere.

## Layout

```
src/agentic_extraction/
  core/          shared data shapes, interfaces, and errors (depended on by all)
  config.py      typed settings read from the environment
  preprocessing/ PDF parsing, outline detection, section-aware chunking
  retrieval/     BM25, dense, hybrid (RRF) fusion behind one Retriever
  llm/           pluggable backends (mock · ollama · openai) + a uniform tool protocol
  agent/         four navigation tools + the from-scratch ReAct orchestrator
  validator/     deterministic grounding check + optional LLM critic, with one retry
  memory/        cross-run memory store (recall/record, opt-in via --memory)
  evaluation.py  scoring of answers against a gold Q&A set
  cli.py         command-line entry point (the outer error boundary)
examples/        run_demo.py · evaluate.py · captured demo_output.md · a CC-BY sample paper
tests/           offline unit tests for every layer
```
