import os
import time

import streamlit as st
from dotenv import load_dotenv

from core.document_loader import (
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    MAX_UPLOAD_FILES,
)
from core.llm_layers import (
    EXTRACTION_MODEL,
    ROUTING_MODEL,
    SYNTHESIS_MODEL,
    resolve_api_key,
)
from core.pipeline_runner import (
    PipelineParams,
    compute_layer_2,
    start_pipeline,
)
from core.sample_data import load_sample_extraction_result
from ui.components import (
    render_calculation_result,
    render_extraction_result,
    render_graph_visualization,
    render_synthesis_result,
    render_upload_status,
)

load_dotenv()

st.set_page_config(page_title="Graph-RAG Corporate Governance Engine", layout="wide")

st.title("Rule-Agnostic Graph-RAG Engine")
st.subheader("Corporate Governance & Liability Cascade Analyzer")

# Minimum gap between two pipeline launches; protects against accidental
# double-clicks / rapid resubmission firing several expensive runs at once.
START_COOLDOWN_SECONDS = 3.0
# How often the UI polls the background worker while a run is in progress.
POLL_INTERVAL_SECONDS = 0.5

_COMMITTED_KEYS = {
    "extraction_result": None,
    "calculation_result": None,
    "synthesis_result": None,
    "pipeline_error": None,
    "routing_trace": "[IDLE] Awaiting document ingestion...",
    "compute_trace": "[IDLE] Layer 2 has not run yet.",
}
_RUNTIME_KEYS = {
    "pipeline_state": None,
    "pipeline_thread": None,
    "pipeline_handled": True,
    "last_start_ts": 0.0,
    "flash": None,
}

for key, default in {**_COMMITTED_KEYS, **_RUNTIME_KEYS}.items():
    if key not in st.session_state:
        st.session_state[key] = default


def _format_duration(seconds) -> str:
    if seconds is None:
        return "—"
    seconds = max(0, int(round(seconds)))
    if seconds < 60:
        return f"{seconds}s"
    minutes, secs = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m {secs:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h {minutes:02d}m"


def _is_running() -> bool:
    state = st.session_state.pipeline_state
    return state is not None and state.status == "running"


def _commit_results(state) -> None:
    st.session_state.extraction_result = state.extraction_result
    st.session_state.calculation_result = state.calculation_result
    st.session_state.synthesis_result = state.synthesis_result
    st.session_state.routing_trace = state.routing_trace or _COMMITTED_KEYS["routing_trace"]
    st.session_state.compute_trace = state.compute_trace or _COMMITTED_KEYS["compute_trace"]
    st.session_state.pipeline_error = None


def _handle_terminal_state() -> None:
    """Commit or discard worker output once, when a run reaches a terminal state."""
    state = st.session_state.pipeline_state
    if state is None or st.session_state.pipeline_handled:
        return
    if state.status == "running":
        return

    if state.status == "done":
        _commit_results(state)
    elif state.status == "cancelled":
        # Revert: committed results are left exactly as they were before the run,
        # so every partial connection built mid-run is discarded.
        st.session_state.flash = (
            "warning",
            "Run cancelled. Partial work was discarded and prior results (if any) "
            "were left untouched. No further API tokens were spent.",
        )
    elif state.status == "error":
        st.session_state.pipeline_error = state.error or "Pipeline failed."
        st.session_state.flash = ("error", st.session_state.pipeline_error)

    st.session_state.pipeline_handled = True


_handle_terminal_state()

with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload SEC EDGAR filings (.htm, .txt; up to 10)",
        type=["htm", "txt"],
        accept_multiple_files=True,
        help="Upload one to ten SEC filing text or HTML documents for entity and relationship extraction.",
    )

    if uploaded_files and len(uploaded_files) > MAX_UPLOAD_FILES:
        st.error(
            f"Maximum {MAX_UPLOAD_FILES} files allowed. "
            f"Remove {len(uploaded_files) - MAX_UPLOAD_FILES} file(s)."
        )
        uploaded_files = uploaded_files[:MAX_UPLOAD_FILES]

    uploaded_names = [file.name for file in uploaded_files] if uploaded_files else []
    render_upload_status(uploaded_names)

    st.header("2. Execution Parameters")
    st.caption(
        "Your natural-language query is auto-routed (Stage 1 GPT-5) to a mechanism and the "
        "two entities in play. The controls below act as a default/override."
    )
    query_topic = st.selectbox(
        "Routing Domain (default / fallback)",
        ["Financial Liability Cascade", "Voting Power Structure", "Tax Leakage Tracing"],
    )
    start_override = st.text_input(
        "Start entity (optional)",
        help="Upstream entity to traverse from. Leave blank to auto-select the controlling parent.",
    )
    target_override = st.text_input(
        "Target entity (optional)",
        help="Downstream entity to traverse to. Leave blank to auto-select the furthest subsidiary.",
    )

    st.header("3. API Gateway Configuration")
    api_key_input = st.text_input(
        "OpenAI API Key Token",
        type="password",
        value=os.environ.get("OPENAI_API_KEY", ""),
        help="Uses OPENAI_API_KEY from the environment when left blank.",
    )

    st.header("4. Demo Mode")
    st.caption(
        "Run the deterministic Layer 2 engine on a bundled multi-tiered fixture graph "
        "without uploading files or providing an API key."
    )
    run_demo = st.button("Run Layer 2 Demo (no API key)", disabled=_is_running())

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Network Topology View")
    if st.session_state.extraction_result:
        render_graph_visualization(
            st.session_state.extraction_result.graph,
            st.session_state.calculation_result,
        )
    else:
        st.info("Upload SEC filing .htm or .txt files and run extraction to populate the entity map.")

with col2:
    st.write("### Runtime State & Token Tracing")
    st.text_area(
        "Extraction LLM Routing Matrix",
        value=st.session_state.routing_trace,
        height=120,
        disabled=True,
    )
    st.text_area(
        "Graph Computation Layer",
        value=st.session_state.compute_trace,
        height=120,
        disabled=True,
    )

st.write("---")
st.write("### Natural Language Reasoning Interface")
user_query = st.text_input(
    "Query the corporate architecture:",
    placeholder="e.g., If Subsidiary C defaults, what is Parent A's total exposure under our current guarantee caps?",
)

running = _is_running()
btn_start, btn_cancel, _ = st.columns([1, 1, 4])
with btn_start:
    start_clicked = st.button(
        "Start",
        type="primary",
        disabled=running,
        use_container_width=True,
        help="Run the full LLM-Graph-LLM pipeline on the uploaded filings.",
    )
with btn_cancel:
    cancel_clicked = st.button(
        "Cancel",
        disabled=not running,
        use_container_width=True,
        help="Stop the run, discard every connection built so far, and stop spending API tokens.",
    )

if running:
    st.warning(
        "A run is in progress. Pressing **Cancel** will immediately stop further API "
        "calls and **undo all work done so far** — every entity/relationship extracted "
        "in this run is discarded and prior results are restored."
    )

# Live progress + feedback while the worker runs (and the final summary after).
state = st.session_state.pipeline_state
if state is not None and state.status != "idle":
    snap = state.snapshot()
    status = snap["status"]
    if status == "running":
        st.progress(min(max(snap["progress"], 0.0), 1.0))
        cols = st.columns(3)
        cols[0].metric("Stage", snap["stage"] or "—")
        cols[1].metric("Elapsed", _format_duration(state.elapsed()))
        cols[2].metric("Est. time remaining", _format_duration(snap["eta_seconds"]))
        if snap["chunks_total"]:
            st.caption(
                f"Extracting chunk {snap['chunks_done']}/{snap['chunks_total']} · "
                f"{snap['partial_nodes']} entities / {snap['partial_edges']} relationships discovered so far "
                "(not yet committed)."
            )
        st.info(snap["message"])
    elif status == "cancelled":
        st.warning(snap["message"])
    elif status == "error":
        st.error(snap["message"])
    elif status == "done":
        st.success(f"{snap['message']} (took {_format_duration(state.elapsed())})")

# Developer panel: full behind-the-scenes view of the worker.
with st.expander("Developer panel (behind the scenes)", expanded=False):
    if state is None:
        st.caption("No pipeline run has been started yet this session.")
    else:
        snap = state.snapshot()
        thread = st.session_state.pipeline_thread
        meta_cols = st.columns(2)
        with meta_cols[0]:
            st.markdown("**Run state**")
            st.json(
                {
                    "status": snap["status"],
                    "stage": snap["stage"],
                    "progress": round(snap["progress"], 4),
                    "elapsed_seconds": round(state.elapsed() or 0.0, 2),
                    "eta_seconds": snap["eta_seconds"],
                    "chunks_done": snap["chunks_done"],
                    "chunks_total": snap["chunks_total"],
                    "partial_nodes": snap["partial_nodes"],
                    "partial_edges": snap["partial_edges"],
                    "cancel_requested": state.cancel_event.is_set(),
                    "worker_thread_alive": bool(thread and thread.is_alive()),
                    "error": snap["error"],
                }
            )
        with meta_cols[1]:
            st.markdown("**Configuration**")
            st.json(
                {
                    "extraction_model": EXTRACTION_MODEL,
                    "routing_model": ROUTING_MODEL,
                    "synthesis_model": SYNTHESIS_MODEL,
                    "chunk_token_size": CHUNK_TOKEN_SIZE,
                    "chunk_token_overlap": CHUNK_TOKEN_OVERLAP,
                    "max_upload_files": MAX_UPLOAD_FILES,
                }
            )

        st.markdown("**Event log**")
        if snap["logs"]:
            log_lines = [
                f"{time.strftime('%H:%M:%S', time.localtime(ts))}  [{level}]  {msg}"
                for ts, level, msg in snap["logs"]
            ]
            st.code("\n".join(log_lines[-200:]), language="text")
        else:
            st.caption("No log events yet.")

if st.session_state.flash:
    level, text = st.session_state.flash
    getattr(st, level, st.info)(text)
    st.session_state.flash = None


# --- Demo mode (deterministic, no API key) ---------------------------------
if run_demo and not running:
    st.session_state.pipeline_error = None
    st.session_state.synthesis_result = None
    extraction_result = load_sample_extraction_result()
    st.session_state.extraction_result = extraction_result
    st.session_state.routing_trace = (
        "[DEMO] Loaded bundled fixture graph; no LLM extraction performed."
    )
    calculation, compute_trace = compute_layer_2(
        extraction_result, query_topic, start_override, target_override
    )
    st.session_state.calculation_result = calculation
    st.session_state.compute_trace = compute_trace
    st.rerun()


# --- Start / Cancel handling ----------------------------------------------
if cancel_clicked and running:
    st.session_state.pipeline_state.request_cancel()
    st.rerun()

if start_clicked and not running:
    now = time.time()
    if now - st.session_state.last_start_ts < START_COOLDOWN_SECONDS:
        st.toast("Please wait a moment before starting another run.")
    elif not uploaded_files:
        st.warning("Upload at least one .htm or .txt SEC filing before running the pipeline.")
    elif not user_query.strip():
        st.warning("Enter a query before running the pipeline.")
    else:
        try:
            api_key = resolve_api_key(api_key_input)
        except ValueError as exc:
            st.session_state.pipeline_error = str(exc)
            st.session_state.flash = ("error", str(exc))
        else:
            params = PipelineParams(
                file_payloads=[(file.name, file.getvalue()) for file in uploaded_files],
                user_query=user_query.strip(),
                query_topic=query_topic,
                start_override=start_override or None,
                target_override=target_override or None,
                api_key=api_key,
            )
            pipeline_state, pipeline_thread = start_pipeline(params)
            st.session_state.pipeline_state = pipeline_state
            st.session_state.pipeline_thread = pipeline_thread
            st.session_state.pipeline_handled = False
            st.session_state.last_start_ts = now
            st.rerun()


if st.session_state.pipeline_error:
    st.error(st.session_state.pipeline_error)

if st.session_state.extraction_result:
    render_extraction_result(st.session_state.extraction_result)

if st.session_state.calculation_result:
    render_calculation_result(st.session_state.calculation_result)

if st.session_state.synthesis_result:
    render_synthesis_result(st.session_state.synthesis_result)


# --- Poll the worker while running ----------------------------------------
if _is_running():
    time.sleep(POLL_INTERVAL_SECONDS)
    st.rerun()
