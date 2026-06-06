"""Tests for Batch 6: the CLI entry point.

Runs the real pipeline end-to-end on a generated PDF with the mock LLM backend, so it is
fast and offline. Dense retrieval is forced off (BM25 only) so no embedding model loads
during the test.
"""

from __future__ import annotations

import pytest

from agentic_extraction import cli
import agentic_extraction.retrieval.factory as retrieval_factory

fitz = pytest.importorskip("fitz")


@pytest.fixture(autouse=True)
def _mock_backend_and_no_dense(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "mock")
    # Keep the CLI test fast: never load an embedding model.
    monkeypatch.setattr(retrieval_factory, "_dense_available", lambda: False)


def _make_pdf(path) -> str:
    doc = fitz.open()
    page = doc.new_page()
    page.insert_text((72, 72), "Quarterly Report", fontsize=22)
    page.insert_text((72, 110), "Revenue", fontsize=16)
    page.insert_text((72, 140), "Total revenue in 2023 was 100 million lira.", fontsize=11)
    doc.save(str(path))
    doc.close()
    return str(path)


def test_cli_runs_and_prints_answer_and_validation(tmp_path, capsys) -> None:
    pdf = _make_pdf(tmp_path / "report.pdf")
    code = cli.main(["--pdf", pdf, "--question", "What was the revenue?"])
    out = capsys.readouterr().out
    assert code == 0
    assert "Validation:" in out  # the verdict line is always printed


def test_cli_show_trace_prints_steps(tmp_path, capsys) -> None:
    pdf = _make_pdf(tmp_path / "report.pdf")
    cli.main(["--pdf", pdf, "--question", "revenue?", "--show-trace"])
    out = capsys.readouterr().out
    assert "Trace:" in out
    assert "final answer" in out


def test_cli_reports_bad_pdf_cleanly(tmp_path, capsys) -> None:
    bad = tmp_path / "not.pdf"
    bad.write_text("plain text, not a pdf")
    code = cli.main(["--pdf", str(bad), "--question", "anything"])
    err = capsys.readouterr().err
    assert code == 1
    assert err.startswith("error:")  # clean message, no traceback
