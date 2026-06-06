"""Agentic multi-modal document question answering.

A small, modular system that answers questions over long PDF documents using an
agent that navigates structure, retrieves evidence, and verifies its own answers.

The package is organised so that every swappable part (LLM backend, embedder,
retriever, validator) sits behind a contract defined in :mod:`agentic_extraction.core`.
Concrete implementations are wired together in one place (the factory), so changing
an implementation never ripples across the codebase.
"""

__version__ = "0.1.0"
