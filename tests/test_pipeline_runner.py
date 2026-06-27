"""Tests for the cancellable background pipeline runner (no real LLM calls).

The worker calls module-level helpers in ``core.pipeline_runner``; these tests
monkeypatch those names with fast fakes so the threading, progress, commit, and
cancellation logic can be validated deterministically and offline.
"""

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import core.pipeline_runner as runner
from core.pipeline_runner import PipelineParams, compute_layer_2, start_pipeline
from core.sample_data import load_sample_extraction_result
from models import (
    CorporateEdge,
    CorporateNode,
    ExtractionResult,
    GraphPayload,
    QueryRoute,
    SynthesisResponse,
)


def _params():
    return PipelineParams(
        file_payloads=[("a.htm", b"<html>x</html>")],
        user_query="trace the parent to the subsidiary",
        query_topic="Financial Liability Cascade",
        start_override=None,
        target_override=None,
        api_key="test-key",
    )


def _fake_extraction_result():
    graph = GraphPayload(
        nodes=[
            CorporateNode(id="Parent", entity_type="Parent"),
            CorporateNode(id="Sub", entity_type="Subsidiary"),
        ],
        edges=[
            CorporateEdge(source="Parent", target="Sub", liability_exposure_usd=100.0),
        ],
    )
    return ExtractionResult(
        graph=graph,
        extraction_summary="fake",
        documents_processed=["a.htm"],
        segments=[],
    )


def _install_fakes(monkeypatch, *, extract=None):
    monkeypatch.setattr(runner, "build_document_corpus", lambda payloads: ("corpus text", ["a.htm"]))
    monkeypatch.setattr(
        runner,
        "segment_text_by_tokens",
        lambda text: [{"chunk_index": i, "token_start": i, "token_end": i + 1, "text": "c"} for i in range(3)],
    )
    if extract is None:
        def extract(document_corpus, source_documents, query_topic, api_key, progress_cb=None, cancel_cb=None):
            payload = GraphPayload(nodes=[CorporateNode(id="Parent")], edges=[])
            for i in range(3):
                if cancel_cb:
                    cancel_cb()
                if progress_cb:
                    progress_cb(i + 1, 3, payload)
            return _fake_extraction_result()

    monkeypatch.setattr(runner, "extract_graph_from_documents", extract)
    monkeypatch.setattr(
        runner,
        "route_query",
        lambda **kwargs: QueryRoute(mechanism="Financial Liability Cascade", start_node="Parent", target_node="Sub"),
    )
    monkeypatch.setattr(
        runner,
        "synthesize_answer",
        lambda **kwargs: SynthesisResponse(answer="done", entities_referenced=["Parent", "Sub"]),
    )


def _join(thread, timeout=5.0):
    thread.join(timeout=timeout)
    assert not thread.is_alive(), "worker thread did not finish in time"


def test_successful_run_commits_results(monkeypatch):
    _install_fakes(monkeypatch)
    state, thread = start_pipeline(_params())
    _join(thread)

    assert state.status == "done"
    assert state.progress == 1.0
    assert state.extraction_result is not None
    assert state.calculation_result is not None
    assert state.calculation_result.computed is True
    assert state.synthesis_result is not None
    assert state.chunks_done == 3


def test_cancellation_discards_partial_work(monkeypatch):
    started = {"flag": False}

    def slow_extract(document_corpus, source_documents, query_topic, api_key, progress_cb=None, cancel_cb=None):
        started["flag"] = True
        for _ in range(200):
            if cancel_cb:
                cancel_cb()  # raises PipelineCancelled once cancel is requested
            time.sleep(0.01)
        return _fake_extraction_result()

    _install_fakes(monkeypatch, extract=slow_extract)
    state, thread = start_pipeline(_params())

    # Wait until extraction has actually begun, then cancel.
    for _ in range(200):
        if started["flag"]:
            break
        time.sleep(0.01)
    state.request_cancel()
    _join(thread)

    assert state.status == "cancelled"
    # Nothing was committed to the result fields -> partial work fully discarded.
    assert state.extraction_result is None
    assert state.calculation_result is None
    assert state.synthesis_result is None
    assert state.cancel_event.is_set()


def test_error_is_captured(monkeypatch):
    def boom(*args, **kwargs):
        raise ValueError("extraction exploded")

    _install_fakes(monkeypatch, extract=boom)
    state, thread = start_pipeline(_params())
    _join(thread)

    assert state.status == "error"
    assert "extraction exploded" in (state.error or "")
    assert state.synthesis_result is None


def test_compute_layer_2_on_sample_fixture():
    extraction_result = load_sample_extraction_result()
    calculation, trace = compute_layer_2(
        extraction_result, "Financial Liability Cascade", None, None
    )
    assert calculation is not None
    assert calculation.computed is True
    assert calculation.final_value is not None
    assert "COMPLETE" in trace
