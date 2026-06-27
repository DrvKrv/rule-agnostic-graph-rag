import os

import streamlit as st
from dotenv import load_dotenv

from core.document_loader import (
    CHUNK_TOKEN_OVERLAP,
    CHUNK_TOKEN_SIZE,
    MAX_UPLOAD_FILES,
    build_document_corpus,
)
from core.engine import GovernanceGraphEngine
from core.llm_layers import extract_graph_from_documents, resolve_api_key, synthesize_answer
from core.routing import build_instruction
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

if "extraction_result" not in st.session_state:
    st.session_state.extraction_result = None
if "calculation_result" not in st.session_state:
    st.session_state.calculation_result = None
if "synthesis_result" not in st.session_state:
    st.session_state.synthesis_result = None
if "pipeline_error" not in st.session_state:
    st.session_state.pipeline_error = None
if "routing_trace" not in st.session_state:
    st.session_state.routing_trace = "[IDLE] Awaiting document ingestion..."
if "compute_trace" not in st.session_state:
    st.session_state.compute_trace = "[IDLE] Layer 2 has not run yet."


def run_layer_2(extraction_result, query_topic, start_override, target_override):
    """Execute the deterministic graph engine and update session state."""
    engine = GovernanceGraphEngine()
    engine.load_graph_from_payload(extraction_result.graph)

    instruction = build_instruction(
        domain=query_topic,
        graph=engine.graph,
        start_node=start_override or None,
        target_node=target_override or None,
    )
    if instruction is None:
        st.session_state.calculation_result = None
        st.session_state.compute_trace = (
            "[SKIPPED] Graph too small or start/target could not be determined."
        )
        return None

    calculation = engine.execute_rule_agnostic_cascade(instruction)
    st.session_state.calculation_result = calculation
    st.session_state.compute_trace = (
        f"[{'COMPLETE' if calculation.computed else 'NO RESULT'}] "
        f"{instruction.aggregation_method} on '{instruction.edge_weight_to_track}' "
        f"from {instruction.start_node} -> {instruction.target_node}. "
        f"Paths found: {len(calculation.paths)}. "
        f"Final: {calculation.final_value} {calculation.unit}."
    )
    return calculation


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
    query_topic = st.selectbox(
        "Routing Domain Override",
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
    run_demo = st.button("Run Layer 2 Demo (no API key)")

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

run_pipeline = st.button("Execute LLM Pipeline", type="primary")


if run_demo:
    st.session_state.pipeline_error = None
    st.session_state.synthesis_result = None
    extraction_result = load_sample_extraction_result()
    st.session_state.extraction_result = extraction_result
    st.session_state.routing_trace = (
        "[DEMO] Loaded bundled fixture graph; no LLM extraction performed."
    )
    run_layer_2(extraction_result, query_topic, start_override, target_override)
    st.rerun()


if run_pipeline:
    st.session_state.pipeline_error = None
    st.session_state.extraction_result = None
    st.session_state.calculation_result = None
    st.session_state.synthesis_result = None

    if not uploaded_files:
        st.warning("Upload at least one .htm or .txt SEC filing before running the pipeline.")
    elif not user_query.strip():
        st.warning("Enter a query before running the pipeline.")
    else:
        try:
            api_key = resolve_api_key(api_key_input)
            file_payloads = [(file.name, file.getvalue()) for file in uploaded_files]

            with st.spinner("Layer 1: Extracting entities and relationships from SEC filing text..."):
                document_corpus, source_documents = build_document_corpus(file_payloads)
                extraction_result = extract_graph_from_documents(
                    document_corpus=document_corpus,
                    source_documents=source_documents,
                    query_topic=query_topic,
                    api_key=api_key,
                )
                st.session_state.extraction_result = extraction_result
                st.session_state.routing_trace = (
                    f"[COMPLETE] Extracted {len(extraction_result.segments)} segment payload(s) "
                    f"using {CHUNK_TOKEN_SIZE}-token chunks with {CHUNK_TOKEN_OVERLAP}-token overlap. "
                    f"Flat preview contains {len(extraction_result.graph.nodes)} nodes and "
                    f"{len(extraction_result.graph.edges)} edges from {len(source_documents)} "
                    f"document(s) under domain '{query_topic}'."
                )

            with st.spinner("Layer 2: Running deterministic graph cascade computation..."):
                calculation = run_layer_2(
                    extraction_result, query_topic, start_override, target_override
                )

            with st.spinner("Layer 3: Synthesizing audit-ready response..."):
                synthesis_result = synthesize_answer(
                    user_query=user_query.strip(),
                    query_topic=query_topic,
                    graph_payload=extraction_result.graph,
                    source_documents=source_documents,
                    extraction_summary=extraction_result.extraction_summary,
                    api_key=api_key,
                    calculation=calculation,
                )
                st.session_state.synthesis_result = synthesis_result
        except ValueError as exc:
            st.session_state.pipeline_error = str(exc)
        except Exception as exc:
            st.session_state.pipeline_error = f"Pipeline failed: {exc}"

if st.session_state.pipeline_error:
    st.error(st.session_state.pipeline_error)

if st.session_state.extraction_result:
    render_extraction_result(st.session_state.extraction_result)

if st.session_state.calculation_result:
    render_calculation_result(st.session_state.calculation_result)

if st.session_state.synthesis_result:
    render_synthesis_result(st.session_state.synthesis_result)
