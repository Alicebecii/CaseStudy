"""Smoke tests for Batch 1: the package imports and the contracts hold together.

These do not test behaviour yet (there is none). They confirm the foundation is
sound: the package installs, core types construct, and configuration reads from
the environment as expected.
"""

from __future__ import annotations

import os

from agentic_extraction import __version__
from agentic_extraction.config import ProviderName, Settings
from agentic_extraction.core import (
    Chunk,
    Document,
    LLMProvider,
    Retriever,
    Section,
    Validator,
)


def test_package_version() -> None:
    assert __version__ == "0.1.0"


def test_core_models_construct() -> None:
    section = Section(id="s1", title="Introduction", level=1, page_start=0)
    chunk = Chunk(id="c1", text="hello", section_path=("Introduction",), page=0)
    doc = Document(
        source_path="x.pdf",
        n_pages=1,
        chunks=(chunk,),
        outline=(section,),
        page_texts=("hello",),
    )
    assert doc.chunks[0].id == "c1"
    assert doc.outline[0].title == "Introduction"


def test_interfaces_are_abstract() -> None:
    # Each interface must refuse direct instantiation; that is what forces
    # callers to depend on the contract and the factory to supply a concrete.
    for interface in (LLMProvider, Retriever, Validator):
        try:
            interface()  # type: ignore[abstract]
        except TypeError:
            continue
        raise AssertionError(f"{interface.__name__} should be abstract")


def test_settings_defaults() -> None:
    settings = Settings()
    assert settings.llm_provider is ProviderName.MOCK
    assert settings.ollama_model == "qwen2.5"
    assert settings.top_k == 5


def test_settings_from_env(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "ollama")
    monkeypatch.setenv("TOP_K", "9")
    settings = Settings.from_env()
    assert settings.llm_provider is ProviderName.OLLAMA
    assert settings.top_k == 9


def test_settings_rejects_bad_provider(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "not-a-provider")
    try:
        Settings.from_env()
    except Exception as exc:  # noqa: BLE001 - we assert on the type below
        assert exc.__class__.__name__ == "ConfigError"
        return
    raise AssertionError("expected ConfigError for an invalid provider")
