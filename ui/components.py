import json

import streamlit as st

from models import ExtractionResult, GraphPayload, SynthesisResponse


def render_upload_status(filenames: list[str]) -> None:
    if not filenames:
        st.info("No PDFs uploaded yet.")
        return

    st.success(f"{len(filenames)} PDF(s) ready for extraction.")
    for name in filenames:
        st.write(f"- `{name}`")


def render_extraction_result(result: ExtractionResult) -> None:
    st.subheader("Extraction Output")
    st.write(result.extraction_summary)

    col_nodes, col_edges = st.columns(2)
    with col_nodes:
        st.markdown("**Nodes**")
        st.dataframe(
            [node.model_dump() for node in result.graph.nodes],
            use_container_width=True,
            hide_index=True,
        )
    with col_edges:
        st.markdown("**Edges**")
        st.dataframe(
            [edge.model_dump() for edge in result.graph.edges],
            use_container_width=True,
            hide_index=True,
        )

    with st.expander("Raw extracted graph JSON"):
        st.code(json.dumps(result.graph.model_dump(), indent=2), language="json")


def render_synthesis_result(result: SynthesisResponse) -> None:
    st.subheader("Synthesis Output")
    st.markdown(result.answer)

    if result.entities_referenced:
        st.markdown("**Entities referenced**")
        st.write(", ".join(result.entities_referenced))

    if result.assumptions:
        st.markdown("**Assumptions**")
        for item in result.assumptions:
            st.write(f"- {item}")

    if result.data_gaps:
        st.markdown("**Data gaps**")
        for item in result.data_gaps:
            st.write(f"- {item}")


def render_graph_preview(payload: GraphPayload) -> None:
    if not payload.nodes:
        st.info("No entities extracted yet.")
        return

    lines = [f"- **{node.id}** ({node.entity_type})" for node in payload.nodes]
    st.markdown("**Extracted entity map**")
    st.markdown("\n".join(lines))
