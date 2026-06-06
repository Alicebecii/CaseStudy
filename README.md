# Agentic Multi-Modal Document Q&A

An agent that answers questions over long PDF documents by navigating their
structure, retrieving evidence, and verifying its own answers. See
[DESIGN.md](DESIGN.md) for the architecture and the reasoning behind each choice.

> This README is a work in progress and will be completed with full setup and
> usage instructions in the final batch.

## Status

Built in small, checkpointed batches. Current state:

- [x] Architecture design (`DESIGN.md`)
- [x] Foundation: package skeleton, core contracts, configuration
- [x] LLM layer (mock + ollama + openai)
- [x] Preprocessing (parse, outline, chunk)
- [x] Retrieval (hybrid BM25 + dense)
- [ ] Agent (tools + ReAct loop)
- [ ] Validator + CLI
- [ ] Tests, demo, technical note

## Layout

```
src/agentic_extraction/
  core/          shared data shapes, interfaces, and errors (depended on by all)
  config.py      typed settings read from the environment
  preprocessing/ PDF parsing, outline, chunking
  retrieval/     BM25, dense, hybrid fusion
  llm/           pluggable LLM backends (mock, ollama, openai)
  agent/         tools and the ReAct orchestrator
  validator/     answer grounding checks
  memory/        cross-run memory (extension point)
  cli.py         command line entry point
```
