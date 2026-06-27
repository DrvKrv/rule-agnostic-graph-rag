"""Tests for Layer 1 extraction resilience (no live API calls).

These monkeypatch the per-chunk extraction so the orchestration logic in
``extract_graph_from_documents`` can be exercised deterministically and offline.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core import llm_layers
from models import CorporateEdge, CorporateNode, GraphPayload


def _corpus() -> str:
    # Long enough to segment into several token-window chunks.
    return ("Parent Corp guarantees Subsidiary Corp debt. " * 4000)


def _payload(tag: str) -> GraphPayload:
    return GraphPayload(
        nodes=[CorporateNode(id=f"Node {tag}", entity_type="Parent")],
        edges=[CorporateEdge(source=f"Node {tag}", target="Sub", liability_exposure_usd=1.0)],
    )


def test_all_chunks_succeed(monkeypatch):
    calls = {"n": 0}

    def fake_chunk(client, chunk_text, source_documents, query_topic, domain_focus):
        calls["n"] += 1
        return _payload(str(calls["n"]))

    monkeypatch.setattr(llm_layers, "_extract_chunk_payload", fake_chunk)
    monkeypatch.setattr(llm_layers, "_responses_client", lambda api_key: object())

    progress = []
    result = llm_layers.extract_graph_from_documents(
        document_corpus=_corpus(),
        source_documents=["a.htm"],
        query_topic="Financial Liability Cascade",
        api_key="test-key",
        progress_callback=lambda done, total: progress.append((done, total)),
    )

    assert len(result.segments) == calls["n"] > 1
    assert "failed" not in result.extraction_summary
    # Progress is reported once per chunk and ends at completion.
    assert progress[-1][0] == progress[-1][1] == len(result.segments)


def test_partial_failure_keeps_other_chunks(monkeypatch):
    calls = {"n": 0}

    def flaky_chunk(client, chunk_text, source_documents, query_topic, domain_focus):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated transient API error")
        return _payload(str(calls["n"]))

    monkeypatch.setattr(llm_layers, "_extract_chunk_payload", flaky_chunk)
    monkeypatch.setattr(llm_layers, "_responses_client", lambda api_key: object())

    result = llm_layers.extract_graph_from_documents(
        document_corpus=_corpus(),
        source_documents=["a.htm"],
        query_topic="Financial Liability Cascade",
        api_key="test-key",
    )

    # One chunk failed but the rest were retained.
    assert len(result.segments) == calls["n"] - 1
    assert "1 of" in result.extraction_summary
    assert "failed extraction" in result.extraction_summary


def test_all_chunks_fail_raises(monkeypatch):
    def always_fail(client, chunk_text, source_documents, query_topic, domain_focus):
        raise RuntimeError("boom")

    monkeypatch.setattr(llm_layers, "_extract_chunk_payload", always_fail)
    monkeypatch.setattr(llm_layers, "_responses_client", lambda api_key: object())

    with pytest.raises(ValueError, match="failed for every chunk"):
        llm_layers.extract_graph_from_documents(
            document_corpus=_corpus(),
            source_documents=["a.htm"],
            query_topic="Financial Liability Cascade",
            api_key="test-key",
        )
