#!/usr/bin/env python
"""Evaluation runner: score the system over a gold Q&A set and print accuracy.

Drives the same pipeline as the CLI over a gold file (default ``examples/gold.json``) and
scores each answer with :mod:`agentic_extraction.evaluation` - grounding verdict, keyword
coverage, and section match - then prints a per-question table and aggregate metrics.

    LLM_PROVIDER=ollama OLLAMA_MODEL=qwen2.5 python examples/evaluate.py
    python examples/evaluate.py path/to/gold.json path/to/document.pdf

Gold items are keyed on expected keywords and section titles (not chunk ids), so the set
stays valid across chunking changes.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from agentic_extraction.agent import create_agent
from agentic_extraction.config import Settings
from agentic_extraction.evaluation import GoldItem, aggregate, score
from agentic_extraction.llm import create_llm_provider
from agentic_extraction.preprocessing import load_document
from agentic_extraction.retrieval import create_retriever
from agentic_extraction.validator import answer_with_validation, create_validator

REPO = Path(__file__).resolve().parent.parent


def main(argv: list[str]) -> int:
    gold_path = Path(argv[0]) if argv else REPO / "examples" / "gold.json"
    gold = json.loads(gold_path.read_text(encoding="utf-8"))
    pdf = Path(argv[1]) if len(argv) > 1 else REPO / gold.get("pdf", "examples/data/peerj-cs-1097.pdf")

    settings = Settings.from_env()
    document = load_document(str(pdf), settings)
    chunks_by_id = {chunk.id: chunk for chunk in document.chunks}
    retriever = create_retriever(document, settings)
    llm = create_llm_provider(settings)
    validator = create_validator(settings, llm)
    agent = create_agent(document, retriever, llm, settings)

    rows = []
    for item in gold["questions"]:
        item_gold = GoldItem(
            question=item["question"],
            expected_keywords=tuple(item.get("expected_keywords", [])),
            expected_section=item.get("expected_section"),
        )
        print(f"  Q: {item_gold.question}", file=sys.stderr)
        result = answer_with_validation(agent, validator, document, item_gold.question)
        rows.append(score(result, item_gold, chunks_by_id))

    print(f"\n{'pass':<6}{'grounded':<10}{'kw_cov':<8}{'section':<8}{'conf':<6}  question")
    print("-" * 84)
    for row in rows:
        print(
            f"{str(row['passed']):<6}{str(row['grounded']):<10}{row['keyword_coverage']:<8}"
            f"{str(row['section_match']):<8}{row['confidence']:<6}  {row['question'][:42]}"
        )
    agg = aggregate(rows)
    print("-" * 84)
    print(
        f"n={agg['n']}  pass_rate={agg['pass_rate']}  grounded_rate={agg['grounded_rate']}  "
        f"mean_kw_coverage={agg['mean_keyword_coverage']}  mean_confidence={agg['mean_confidence']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
