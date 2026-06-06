# Architecture Design: Agentic Multi-Modal Document Q&A

**Author:** Seyyid Ali Cebeci · **Status:** Design (written before implementation)

This document is the primary deliverable. It explains *what* we build, but more
importantly *why* each component is shaped the way it is and *what we traded away*.
The code in `src/agentic_extraction/` is the executable form of these decisions.

---

## 1. Problem framing

We must answer questions over **long, multi-modal documents** (PDFs that mix prose,
section structure, tables and figures). The brief contrasts two paradigms:

- **RAG:** embed the whole document, retrieve top-k chunks for the question, stuff
  them into one prompt, generate. Simple and fast, but it is *stateless* and
  *single-shot*. It cannot decide to look at the table of contents first, cannot
  follow a reference from section 3 to section 7, and conflates "semantically near"
  with "actually answers the question".
- **Agentic:** an agent that reads the way a person does. It skims the structure,
  navigates to the relevant section, reads closely, cross-checks, then answers. It is
  a *loop* with *tools* and *intermediate reasoning*.

We build the agentic system, and we deliberately keep a thin retrieval layer *inside*
it as one tool among several. The agent decides when to retrieve, when to read a whole
section instead, and when it has enough evidence to answer. That decision-making is the
thing the brief is really testing, so it is where we invest.

### Design principles (applied throughout)
1. **The agent orchestrates; tools do the work.** All document access is a tool call,
   so the agent's behaviour is observable and testable.
2. **Every external dependency is behind an interface** (LLM, embedder). The system
   must run with a free local model *and* a paid API by changing one env var, and must
   run in CI with no model at all.
3. **Grounding over fluency.** An answer is only as good as the sources it cites; an
   uncited claim is treated as a failure, not a feature.
4. **Fail loudly and locally.** A corrupt PDF, a dead API, or an empty search result
   must produce a clear, contained error rather than a confident hallucination.

---

## 2. System overview

```
                  ┌──────────────────────────────────────────────────┐
   PDF  ─────────▶│ PREPROCESSING                                     │
                  │  pdf_parser (PyMuPDF) -> text spans + font sizes  │
                  │  outline      -> heading tree (structural map)    │
                  │  chunker      -> section-aware chunks             │
                  │                 {id, text, section, page}         │
                  └───────────────────────┬──────────────────────────┘
                                          │  DocumentIndex (in memory)
                                          ▼
                  ┌──────────────────────────────────────────────────┐
                  │ RETRIEVAL  (one tool the agent may call)          │
                  │  BM25 (lexical)  +  dense (local embeddings)      │
                  │  fused by Reciprocal Rank Fusion -> ranked chunks │
                  └───────────────────────┬──────────────────────────┘
                                          ▲ search(query)
   question ──┐                           │
              ▼                           │
   ┌────────────────────────────────────────────────────────────────┐
   │ AGENT  (from-scratch ReAct loop, bounded steps)                 │
   │                                                                  │
   │   TOOLS:  get_outline()      -> see the document's structure    │
   │           search(query)      -> hybrid retrieval                │
   │           read_section(id)   -> full text of one section        │
   │           read_page(n)       -> raw text of one page            │
   │                                                                  │
   │   loop:  think -> choose tool -> observe -> ... -> final answer  │
   │          with inline [chunk_id] citations                       │
   │                          │ uses                                 │
   │                          ▼                                       │
   │              ┌───────────────────────┐                          │
   │              │  LLMProvider (ABC)     │  mock | ollama | openai  │
   │              └───────────────────────┘                          │
   └───────────────────────────┬────────────────────────────────────┘
                               ▼  draft answer + cited chunk ids
   ┌────────────────────────────────────────────────────────────────┐
   │ VALIDATOR                                                        │
   │  (a) deterministic grounding check: do the cited chunks really   │
   │      support the answer? (lexical overlap, runs without an LLM)  │
   │  (b) optional LLM critic: "is this answer entailed by sources?"  │
   │  verdict: GROUNDED / WEAK / UNGROUNDED  (+ one retry on failure) │
   └───────────────────────────┬────────────────────────────────────┘
                               ▼
                    answer + sources + confidence  ──▶ CLI
```

The data contract that ties the modules together is a single `Chunk` record:
`{id, text, section_path, page}`. Retrieval ranks chunks, the agent cites chunk ids,
and the validator checks those ids, so provenance is preserved end-to-end and a
citation always resolves back to an exact span on an exact page.

---

## 3. The six required design questions

### 3.1 Document pre-processing: how is text vs. visual content separated? Which tools?

**Tool: PyMuPDF (`fitz`).** Chosen over `pdfplumber` and Adobe PDF Extract because a
single library gives us all three things we need: (a) text with per-span **font size
and flags**, (b) the embedded outline/bookmarks when present, and (c) **page to image**
rendering for the multimodal extension, with no extra dependency and a permissive
licence. `pdfplumber` is excellent for ruled tables but has no image rendering;
Adobe's API is high quality but is a paid network dependency, which violates our
"runs offline and free" principle.

**Text vs. visual separation.** PyMuPDF exposes each page as a list of *blocks*; text
blocks carry spans, image blocks carry xrefs. We keep them separate from the start:
- **Text path:** spans are grouped into lines/paragraphs and fed to the outline +
  chunker (the MVP focus).
- **Visual path:** image/figure regions and full-page renders are kept addressable by
  page number so the `read_page` tool (and the multimodal bonus) can reach them. The
  MVP processes text; the visual path is wired but only fully exploited if we add a
  vision LLM.

**Failure handling.** Parsing is wrapped so an encrypted, truncated, or non-PDF file
raises one typed `DocumentError` with the page index, rather than a deep PyMuPDF stack
trace. An empty text extraction (e.g. a scanned PDF with no text layer) is detected and
reported as "no extractable text, OCR required", which is the honest answer.

### 3.2 Structural navigation: how does the agent find the right section? How is structure represented?

A human opens a long PDF and reads the **table of contents** first. We give the agent
the same affordance.

**Building the structure.** We prefer the PDF's embedded outline (`doc.get_toc()`) when
the author provided one. When absent, we **infer headings from typography**: spans whose
font size is meaningfully larger than the modal body-text size, are short, and start a
line, are promoted to headings; relative size buckets give heading *levels*. This yields
a hierarchical tree.

**Representation.** A `Section` tree where each node is
`{id, title, level, page_start, children}`. The agent never sees raw 50-page text up
front; it calls `get_outline()` and receives a compact, indented map (a few hundred
tokens) of the whole document. It then navigates by `read_section(id)`. This is the
concrete mechanism that makes the system scale to long documents: **navigation is
O(outline), not O(document)**, so context stays small and the agent's choices are
explicit and auditable.

*Trade-off:* typography-based heading detection is a heuristic and can misclassify
unusual layouts. We accept this because (a) embedded outlines are used when available,
and (b) retrieval is an independent second path to the same content, so a missed
heading degrades navigation but never blocks an answer.

### 3.3 Retrieval: how are text-based and visual search integrated?

**Hybrid lexical + dense, fused with Reciprocal Rank Fusion (RRF).**

- **BM25** (lexical) nails exact terms the user typed verbatim, such as names, numbers,
  acronyms and API symbols, where embeddings are weakest.
- **Dense** (a local `sentence-transformers` model, default `all-MiniLM-L6-v2`) catches
  paraphrase and synonymy where BM25 is blind.
- **RRF** fuses the two ranked lists by reciprocal rank (`Σ 1/(k+rank)`). We chose RRF
  over score-normalisation because BM25 and cosine scores live on incomparable scales;
  RRF only needs ranks, so it is robust and parameter-light.

The embedder sits behind its own interface and uses a **local** model, so retrieval
costs nothing and is fully reproducible by a reviewer with no API key.

**Integrating visual search.** Chunks carry a `page` number, which is the join key
between modalities: a text hit on page 7 lets the agent call `read_page(7)` to pull the
figure/table on that page. So in the MVP, visual content is reached *via* text locality
rather than via a separate image index. The clean extension (bonus) is a parallel image
embedding index fused into the same RRF step; the architecture already returns ranked,
page-stamped results, so adding a third ranked list is additive.

### 3.4 Agent architecture: how many agents, what roles, how do they communicate?

**One orchestrator agent + one validator, by deliberate choice.** The brief is explicit
that a *perfect single-agent system beats a half-finished multi-agent one*, so we make
the single agent excellent and keep multi-agent as a clean extension point.

- **Orchestrator (ReAct loop, from scratch).** Receives the question, is told the
  available tools, and runs `think -> act (tool call) -> observe` until it emits a final
  answer or hits a step bound. We write the loop ourselves (no LangChain) so the
  control flow, and the reasoning about it, is fully visible, which is what the
  architecture grade rewards.
- **Validator.** A separate, narrowly-scoped checker (section 3.5). Keeping it out of
  the orchestrator avoids the "judge marking its own homework" failure mode.

**Communication & tool protocol.** Tools are plain Python functions with typed
signatures and JSON-serialisable results. The agent calls them through a *uniform
tool-call protocol* that works on every backend: where the LLM supports native
tool-calling (OpenAI) we use it; where it does not (small Ollama models) we fall back to
a strict JSON action protocol the model emits as text and we parse. Same tools, same
loop, swappable brain.

*Extension point (multi-agent bonus):* the orchestrator could dispatch a `search`
sub-question to a retrieval specialist and a `read_page` to a vision specialist and
merge their findings. The tool boundary is already the agent boundary, so this is a
later addition, not a rewrite.

### 3.5 Validation & reliability: how are wrong answers prevented?

Two layers, cheap-and-deterministic first:

1. **Deterministic grounding check (no LLM).** Every claim in the answer should rest on
   a cited chunk. We verify that the answer actually cites chunk ids, and that there is
   real lexical overlap between the answer's content terms and the cited chunks' text.
   Crucially this runs **without an LLM**, so it works under the mock backend and makes
   our unit tests meaningful rather than circular.
2. **Optional LLM critic.** When a real backend is configured, a second LLM call asks
   "is this answer fully entailed by these sources? list any unsupported claim." It is a
   different prompt and (optionally) a different model than the author, so it is a genuine
   second opinion.

The validator returns a verdict (`GROUNDED` / `WEAK` / `UNGROUNDED`) plus the offending
claims. On a failing verdict the agent gets **one bounded retry** with the critique fed
back ("these claims were unsupported, search again or say you don't know"). If it still
fails, the system returns the best answer *with an explicit low-confidence warning*
rather than presenting a guess as fact. Refusing to over-claim is itself a reliability
feature.

### 3.6 Memory: is cross-task learning possible? How?

Yes. Two useful, distinct scopes:

- **Within a run (working memory):** the ReAct transcript, that is every
  think/act/observe step, is the agent's short-term memory and is what lets it
  cross-verify across sections.
- **Across runs (long-term memory):** a small append-only store keyed by document, holding
  `(question, answer, cited_chunk_ids, verdict)`. On a new question we can (a) short-cut
  near-duplicate questions, and (b) seed retrieval with chunks that previously answered
  related questions. We keep it as an explicit, inspectable store rather than an opaque
  vector cache so its effect on answers stays auditable, consistent with principle #4.

The long-term store ships in the MVP as a functional extension point
(`memory/store.py`: an append-only `JsonMemoryStore` behind the `MemoryStore` contract,
with `record`/`recall`). It is tested on its own but not yet wired into the loop - doing
that wiring (recall-to-short-cut, retrieval seeding above) is the cross-task-learning
bonus, and the store is what makes it a small addition rather than a redesign.

---

## 4. Module layout

One module per concern, matching the code-quality rubric:

```
src/agentic_extraction/
  config.py        env-driven settings; selects the LLM provider + models
  preprocessing/   pdf_parser.py · outline.py · chunker.py
  retrieval/       bm25.py · dense.py · hybrid.py · index.py
  llm/             base.py (ABC) · mock_provider.py · ollama_provider.py · openai_provider.py
  agent/           tools.py · orchestrator.py
  validator/       grounding.py
  memory/          store.py      (extension point)
  cli.py           --pdf <path> --question "<q>"
```

Dependencies point one direction: `preprocessing -> retrieval -> agent -> validator -> cli`.
`llm` is a leaf depended on by `agent`/`validator`; nothing depends back on the CLI.

---

## 5. Key trade-offs, summarised

| Decision | Chosen | Rejected | Why |
|---|---|---|---|
| PDF library | PyMuPDF | pdfplumber, Adobe API | text + typography + image render in one offline, free lib |
| Retrieval | Hybrid BM25+dense, RRF | BM25-only / dense-only | lexical exactness *and* semantic recall; RRF needs no score calibration |
| Embeddings | Local sentence-transformers | API embeddings | zero cost, reproducible, no key needed |
| Framework | From scratch | LangChain / LlamaIndex | control flow stays visible & testable (the 40% lever) |
| Agent count | 1 orchestrator + 1 validator | full multi-agent | brief rewards depth; multi-agent kept as clean seam |
| Validation | Deterministic check + optional critic | LLM-only | works offline / in CI; not self-judging |
| LLM access | Pluggable mock / ollama / openai | single hard-coded SDK | free dev + CI, paid run only for final verification |

---

## 6. Failure modes & responses

| Failure | Response |
|---|---|
| Corrupt / encrypted / non-PDF | typed `DocumentError`, clear message, no crash |
| PDF has no text layer (scanned) | detected; reports "OCR required" instead of empty answer |
| Empty retrieval result | agent is told "no matches"; may broaden query or say it cannot find it |
| LLM API error / timeout | bounded retry, then a clear failure surfaced to the CLI |
| Agent exceeds step budget | loop stops, returns best-grounded partial answer + warning |
| Ungrounded answer | validator forces one retry, then returns low-confidence flagged answer |

---

## 7. Verification strategy

- **Unit tests (pytest, mock backend, no cost):** preprocessing extracts text + outline
  from a generated PDF; retrieval ranks a known-relevant chunk first; the agent loop
  drives tools to a final answer; the validator distinguishes grounded from ungrounded.
- **Local end-to-end:** `LLM_PROVIDER=ollama … cli --pdf data/<paper>.pdf --question …`.
- **Final paid check:** one `LLM_PROVIDER=openai` run to confirm real-LLM behaviour.
- **Demo:** three representative questions over one academic paper, captured in
  `examples/demo_output.md`.
