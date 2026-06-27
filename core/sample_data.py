"""Hardcoded multi-tiered fixture graph (Phase 1, Task A).

This lets the deterministic Layer 2 engine be exercised end-to-end without an
OpenAI key or any document upload, so the graph math can be validated in
isolation from the LLM extraction pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

from models import ExtractionResult, GraphPayload

_SAMPLE_PATH = Path(__file__).resolve().parent.parent / "data" / "sample_graph.json"


def load_sample_payload() -> GraphPayload:
    """Load the bundled multi-tiered corporate graph fixture."""
    raw = json.loads(_SAMPLE_PATH.read_text(encoding="utf-8"))
    return GraphPayload.model_validate(raw)


def load_sample_extraction_result() -> ExtractionResult:
    """Wrap the fixture graph as an ExtractionResult for the demo pipeline."""
    payload = load_sample_payload()
    summary = (
        "Demo mode: loaded a hardcoded multi-tiered corporate guarantee graph with "
        f"{len(payload.nodes)} entities and {len(payload.edges)} relationships. No LLM "
        "extraction was performed; this validates the deterministic Layer 2 engine in isolation."
    )
    return ExtractionResult(
        graph=payload,
        extraction_summary=summary,
        documents_processed=["sample_graph.json (bundled demo fixture)"],
        segments=[],
    )
