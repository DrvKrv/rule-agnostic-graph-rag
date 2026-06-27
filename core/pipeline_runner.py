"""Background, cancellable orchestration for the LLM-Graph-LLM pipeline.

Streamlit executes a script top-to-bottom on every interaction, so a long
multi-minute pipeline (dozens of GPT-5 extraction calls over a large SEC filing)
would otherwise block the UI entirely - the user could not see progress and
could not stop a run that is burning tokens.

This module runs the pipeline on a background thread and exposes a thread-safe
``PipelineState`` that the Streamlit script polls between reruns. The worker:

* reports per-stage progress, a human-readable status, and an ETA,
* checks a cancellation flag before every billable OpenAI call so a Cancel
  press stops further token spend almost immediately, and
* keeps all results *inside* the state object and never mutates committed
  session state. The UI only commits results when the run finishes successfully,
  so cancelling cleanly discards every partial node/edge ("connection") that was
  built mid-run -- nothing to undo, because nothing was committed.

The worker never touches ``streamlit`` so it is safe to run without a
``ScriptRunContext`` and is unit-testable in isolation.
"""

from __future__ import annotations

import threading
import time
import traceback
from dataclasses import dataclass, field
from typing import Optional

from core.document_loader import build_document_corpus, segment_text_by_tokens
from core.engine import GovernanceGraphEngine
from core.llm_layers import (
    extract_graph_from_documents,
    route_query,
    synthesize_answer,
)
from core.routing import build_instruction
from models import CalculationResult, ExtractionResult


class PipelineCancelled(Exception):
    """Raised inside the worker when the user requests cancellation."""


@dataclass
class PipelineParams:
    """Immutable inputs captured at the moment the user presses Start."""

    file_payloads: list[tuple[str, bytes]]
    user_query: str
    query_topic: str
    start_override: Optional[str]
    target_override: Optional[str]
    api_key: str


# Coarse progress checkpoints. Extraction dominates wall-clock time (one slow
# GPT-5 call per chunk), so it owns the bulk of the bar.
_P_INGEST = 0.03
_P_EXTRACT_START = 0.05
_P_EXTRACT_END = 0.82
_P_ROUTE = 0.86
_P_LAYER2 = 0.90
_P_SYNTH_START = 0.93
_P_DONE = 1.0


class PipelineState:
    """Thread-safe container shared between the worker and the Streamlit UI."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.cancel_event = threading.Event()

        self.status: str = "idle"  # idle|running|done|cancelled|error
        self.stage: str = ""
        self.message: str = "Idle."
        self.progress: float = 0.0
        self.eta_seconds: Optional[float] = None

        self.started_at: Optional[float] = None
        self.finished_at: Optional[float] = None

        self.chunks_total: int = 0
        self.chunks_done: int = 0
        self.partial_nodes: int = 0
        self.partial_edges: int = 0

        self.logs: list[tuple[float, str, str]] = []
        self.stage_timings: dict[str, float] = {}

        # Results live here and are only copied into committed session state
        # by the UI once status == "done".
        self.extraction_result: Optional[ExtractionResult] = None
        self.calculation_result: Optional[CalculationResult] = None
        self.synthesis_result = None
        self.routing_trace: str = ""
        self.compute_trace: str = ""

        self.error: Optional[str] = None

        self._extraction_started_at: Optional[float] = None

    # -- mutation helpers ------------------------------------------------
    def log(self, level: str, message: str) -> None:
        with self._lock:
            self.logs.append((time.time(), level, message))
            if len(self.logs) > 500:
                self.logs = self.logs[-500:]

    def set(self, **kwargs) -> None:
        with self._lock:
            for key, value in kwargs.items():
                setattr(self, key, value)

    def mark_stage(self, stage: str, message: str, progress: float) -> None:
        now = time.time()
        with self._lock:
            self.stage = stage
            self.message = message
            self.progress = progress
            self.stage_timings[stage] = now
            self.logs.append((now, "INFO", f"[{stage}] {message}"))

    # -- cancellation ----------------------------------------------------
    def request_cancel(self) -> None:
        self.cancel_event.set()
        self.set(message="Cancellation requested; stopping after the current step...")
        self.log("WARN", "Cancellation requested by user.")

    def raise_if_cancelled(self) -> None:
        if self.cancel_event.is_set():
            raise PipelineCancelled()

    # -- read-only snapshot for the UI ----------------------------------
    def snapshot(self) -> dict:
        with self._lock:
            return {
                "status": self.status,
                "stage": self.stage,
                "message": self.message,
                "progress": self.progress,
                "eta_seconds": self.eta_seconds,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
                "chunks_total": self.chunks_total,
                "chunks_done": self.chunks_done,
                "partial_nodes": self.partial_nodes,
                "partial_edges": self.partial_edges,
                "logs": list(self.logs),
                "error": self.error,
            }

    def elapsed(self) -> Optional[float]:
        if self.started_at is None:
            return None
        end = self.finished_at or time.time()
        return end - self.started_at


def compute_layer_2(
    extraction_result: ExtractionResult,
    query_topic: str,
    start_override: Optional[str],
    target_override: Optional[str],
) -> tuple[Optional[CalculationResult], str]:
    """Pure Layer 2 execution: deterministic NetworkX cascade, no Streamlit.

    Returns ``(calculation_result, compute_trace)``. ``calculation_result`` is
    ``None`` when the graph is too small or start/target cannot be determined.
    """

    engine = GovernanceGraphEngine()
    engine.load_graph_from_payload(extraction_result.graph)

    instruction = build_instruction(
        domain=query_topic,
        graph=engine.graph,
        start_node=start_override or None,
        target_node=target_override or None,
    )
    if instruction is None:
        return None, "[SKIPPED] Graph too small or start/target could not be determined."

    calculation = engine.execute_rule_agnostic_cascade(instruction)
    trace = (
        f"[{'COMPLETE' if calculation.computed else 'NO RESULT'}] "
        f"{instruction.aggregation_method} on '{instruction.edge_weight_to_track}' "
        f"from {instruction.start_node} -> {instruction.target_node}. "
        f"Paths found: {len(calculation.paths)}. "
        f"Final: {calculation.final_value} {calculation.unit}."
    )
    return calculation, trace


def _run_pipeline(state: PipelineState, params: PipelineParams) -> None:
    """Worker body. Never imports/uses streamlit."""

    try:
        state.set(status="running", started_at=time.time(), error=None)
        state.mark_stage("ingest", "Parsing and chunking uploaded SEC filings...", _P_INGEST)
        state.raise_if_cancelled()

        document_corpus, source_documents = build_document_corpus(params.file_payloads)
        chunks = segment_text_by_tokens(document_corpus)
        state.set(chunks_total=len(chunks))
        state.log(
            "INFO",
            f"Built corpus from {len(source_documents)} file(s); {len(chunks)} chunk(s) to extract.",
        )
        state.raise_if_cancelled()

        # ---- Layer 1: extraction --------------------------------------
        state.mark_stage(
            "extract",
            f"Layer 1: extracting entities/relationships from {len(chunks)} chunk(s)...",
            _P_EXTRACT_START,
        )
        state.set(_extraction_started_at=time.time())

        def progress_cb(done: int, total: int, payload) -> None:
            now = time.time()
            started = state._extraction_started_at or now
            per_chunk = (now - started) / max(done, 1)
            remaining = total - done
            # Reserve a little for routing + the single large synthesis call.
            eta = per_chunk * remaining + per_chunk * 2.0
            nodes = state.partial_nodes + len(payload.nodes)
            edges = state.partial_edges + len(payload.edges)
            frac = done / total if total else 1.0
            state.set(
                chunks_done=done,
                partial_nodes=nodes,
                partial_edges=edges,
                progress=_P_EXTRACT_START + (_P_EXTRACT_END - _P_EXTRACT_START) * frac,
                eta_seconds=eta,
                message=(
                    f"Layer 1: extracted chunk {done}/{total} "
                    f"({nodes} entities, {edges} relationships so far)."
                ),
            )
            state.log(
                "INFO",
                f"Chunk {done}/{total}: +{len(payload.nodes)} nodes, +{len(payload.edges)} edges.",
            )

        extraction_result = extract_graph_from_documents(
            document_corpus=document_corpus,
            source_documents=source_documents,
            query_topic=params.query_topic,
            api_key=params.api_key,
            progress_cb=progress_cb,
            cancel_cb=state.raise_if_cancelled,
        )
        state.set(
            extraction_result=extraction_result,
            routing_trace=(
                f"[COMPLETE] Extracted {len(extraction_result.segments)} segment payload(s). "
                f"Flat preview contains {len(extraction_result.graph.nodes)} nodes and "
                f"{len(extraction_result.graph.edges)} edges from {len(source_documents)} "
                f"document(s) under domain '{params.query_topic}'."
            ),
        )
        state.raise_if_cancelled()

        # ---- Stage 1 routing ------------------------------------------
        state.mark_stage("route", "Stage 1 routing: mapping the query to a mechanism and entities...", _P_ROUTE)
        effective_topic = params.query_topic
        routed_start = params.start_override or None
        routed_target = params.target_override or None
        if not (routed_start and routed_target):
            try:
                node_ids = [node.id for node in extraction_result.graph.nodes]
                route = route_query(
                    user_query=params.user_query,
                    default_topic=params.query_topic,
                    node_ids=node_ids,
                    api_key=params.api_key,
                )
                effective_topic = route.mechanism or params.query_topic
                routed_start = routed_start or route.start_node
                routed_target = routed_target or route.target_node
                state.set(
                    routing_trace=state.routing_trace
                    + f"\n[ROUTE] mechanism='{effective_topic}', start='{routed_start}', target='{routed_target}'."
                )
                state.log("INFO", f"Routed mechanism={effective_topic}, start={routed_start}, target={routed_target}.")
            except Exception as route_exc:  # noqa: BLE001 - routing is best-effort
                state.set(
                    routing_trace=state.routing_trace
                    + f"\n[ROUTE] LLM routing skipped ({route_exc}); using deterministic selection."
                )
                state.log("WARN", f"LLM routing skipped: {route_exc}")
        state.raise_if_cancelled()

        # ---- Layer 2: deterministic cascade ---------------------------
        state.mark_stage("layer2", "Layer 2: running deterministic graph cascade computation...", _P_LAYER2)
        calculation, compute_trace = compute_layer_2(
            extraction_result, effective_topic, routed_start, routed_target
        )
        state.set(calculation_result=calculation, compute_trace=compute_trace)
        state.log("INFO", compute_trace)
        state.raise_if_cancelled()

        # ---- Layer 3: synthesis ---------------------------------------
        state.mark_stage("synthesize", "Layer 3: synthesizing audit-ready response...", _P_SYNTH_START)
        synthesis_result = synthesize_answer(
            user_query=params.user_query,
            query_topic=effective_topic,
            graph_payload=extraction_result.graph,
            source_documents=source_documents,
            extraction_summary=extraction_result.extraction_summary,
            api_key=params.api_key,
            calculation=calculation,
        )
        state.set(synthesis_result=synthesis_result)
        state.raise_if_cancelled()

        state.set(
            status="done",
            stage="done",
            progress=_P_DONE,
            eta_seconds=0.0,
            finished_at=time.time(),
            message="Pipeline complete.",
        )
        state.log("INFO", "Pipeline complete.")

    except PipelineCancelled:
        state.set(
            status="cancelled",
            finished_at=time.time(),
            eta_seconds=None,
            message=(
                "Cancelled by user. No results were committed and every partial "
                "connection built so far was discarded; no further API calls were made."
            ),
        )
        state.log("WARN", "Pipeline cancelled; partial work discarded.")
    except Exception as exc:  # noqa: BLE001 - surface any failure to the UI
        state.set(
            status="error",
            finished_at=time.time(),
            eta_seconds=None,
            error=str(exc),
            message=f"Pipeline failed: {exc}",
        )
        state.log("ERROR", f"{exc}\n{traceback.format_exc()}")


def start_pipeline(params: PipelineParams) -> tuple[PipelineState, threading.Thread]:
    """Spawn the worker thread and return the shared state plus the thread."""

    state = PipelineState()
    thread = threading.Thread(
        target=_run_pipeline,
        args=(state, params),
        name="llm-pipeline-worker",
        daemon=True,
    )
    thread.start()
    return state, thread
