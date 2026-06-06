"""Command-line entry point: answer a question about a PDF.

This is the one place that wires the whole pipeline together and turns the library into a
runnable tool (the brief's CLI requirement): load the PDF, build the retriever, the LLM
backend, and the agent, run the validated answer, and print it.

It is also the system's outer error boundary. Any typed
:class:`AgenticExtractionError` - a bad PDF, a dead model, an invalid config - is caught
here and shown as a clean one-line message with a non-zero exit code, never a stack trace
(DESIGN.md principle #4).
"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence

from .agent import create_agent
from .config import Settings
from .core.errors import AgenticExtractionError
from .core.models import Chunk, ValidatedAnswer
from .llm import create_llm_provider
from .memory import JsonMemoryStore
from .preprocessing import load_document, outline_to_dict
from .retrieval import create_retriever
from .validator import answer_with_validation, create_validator


def main(argv: Sequence[str] | None = None) -> int:
    """Parse arguments, run the pipeline, print the result. Returns a process exit code."""
    args = _parse_args(argv)
    try:
        settings = Settings.from_env()
        document = load_document(args.pdf, settings)

        if args.outline_json:
            print(json.dumps(outline_to_dict(document.outline), indent=2, ensure_ascii=False))
            return 0

        if not args.question:
            print("error: --question is required (or use --outline-json)", file=sys.stderr)
            return 1

        retriever = create_retriever(document, settings)
        llm = create_llm_provider(settings)
        agent = create_agent(document, retriever, llm, settings)
        validator = create_validator(settings, llm)
        memory = JsonMemoryStore(args.memory) if args.memory else None
        result = answer_with_validation(agent, validator, document, args.question, memory=memory)
    except AgenticExtractionError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1

    _print_result(result, document.chunks, show_trace=args.show_trace)
    return 0


def _parse_args(argv: Sequence[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="agentic-extract",
        description="Answer a question over a PDF using an agent that navigates, retrieves, "
        "and validates its own answer.",
    )
    parser.add_argument("--pdf", required=True, help="Path to the PDF to query.")
    parser.add_argument("--question", help="The question to answer.")
    parser.add_argument(
        "--show-trace",
        action="store_true",
        help="Print the agent's step-by-step reasoning trace.",
    )
    parser.add_argument(
        "--outline-json",
        action="store_true",
        help="Print the document's structured outline as JSON and exit (no question needed).",
    )
    parser.add_argument(
        "--memory",
        metavar="PATH",
        help="Enable cross-run memory at this JSONL path (recall + record answers).",
    )
    return parser.parse_args(argv)


def _print_result(
    result: ValidatedAnswer,
    chunks: Sequence[Chunk],
    show_trace: bool,
) -> None:
    answer = result.answer
    validation = result.validation
    chunks_by_id = {chunk.id: chunk for chunk in chunks}

    print(answer.answer or "(no answer produced)")

    if answer.cited_chunk_ids:
        print("\nSources:")
        for chunk_id in answer.cited_chunk_ids:
            chunk = chunks_by_id.get(chunk_id)
            if chunk is None:
                print(f"  [{chunk_id}] (unresolved citation)")
            else:
                path = " > ".join(chunk.section_path) if chunk.section_path else "-"
                print(f"  [{chunk_id}] p{chunk.page}  {path}")

    print(
        f"\nValidation: {validation.verdict.value} "
        f"(confidence {validation.confidence:.2f}, attempts {result.attempts})"
    )
    if validation.unsupported_claims:
        print("Statements not grounded in the cited sources:")
        for claim in validation.unsupported_claims:
            print(f"  - {claim}")

    if show_trace:
        print("\nTrace:")
        for line in answer.trace:
            print(f"  {line}")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
