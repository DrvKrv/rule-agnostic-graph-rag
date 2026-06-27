import json
import tempfile
from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components

from models import CalculationResult, ExtractionResult, GraphPayload, SynthesisResponse

_ENTITY_COLORS = {
    "Parent": "#2563eb",
    "Subsidiary": "#10b981",
    "Shell": "#f59e0b",
    "Operating Company": "#8b5cf6",
}


def render_upload_status(filenames: list[str]) -> None:
    if not filenames:
        st.info("No SEC filing files uploaded yet.")
        return

    st.success(f"{len(filenames)} SEC filing file(s) ready for extraction.")
    for name in filenames:
        st.write(f"- `{name}`")


def render_extraction_result(result: ExtractionResult) -> None:
    st.subheader("Extraction Output")
    st.write(result.extraction_summary)
    st.caption(
        "Layer 1 segment payloads are returned independently. The tables below are a flat preview "
        "of those literal payloads before Layer 2 graph computation."
    )

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

    with st.expander("Layer 1 extraction segments JSON"):
        segment_payloads = [segment.model_dump() for segment in result.segments]
        st.code(json.dumps(segment_payloads, indent=2), language="json")


def render_calculation_result(calc: CalculationResult) -> None:
    st.subheader("Layer 2 Deterministic Computation")

    instr = calc.instruction
    st.caption(
        f"Tracked `{instr.edge_weight_to_track}` from **{instr.start_node}** to "
        f"**{instr.target_node}** using `{calc.aggregation_method}`"
        + (f" constrained by `{instr.edge_constraint}`." if instr.edge_constraint else ".")
    )

    if calc.computed and calc.final_value is not None:
        if calc.unit == "USD":
            display_value = f"${calc.final_value:,.2f}"
        else:
            display_value = f"{calc.final_value:.4f}"
        st.metric(label=f"Computed result ({calc.unit})", value=display_value)
    else:
        st.warning("Layer 2 could not compute a value. See diagnostics below.")

    if calc.formula_summary:
        st.markdown(f"**Derivation:** {calc.formula_summary}")

    if calc.paths:
        st.markdown("**Path logs**")
        for index, path in enumerate(calc.paths, start=1):
            route = " -> ".join(path.nodes)
            with st.expander(f"Path {index}: {route}"):
                st.write(f"Formula: {path.formula}")
                if path.constrained_value is not None:
                    st.write(f"Path value: {path.constrained_value}")
                if path.notes:
                    for note in path.notes:
                        st.write(f"- {note}")

    if calc.cycles_detected:
        st.markdown("**Cycles detected**")
        for cycle in calc.cycles_detected:
            st.write(" -> ".join(cycle))

    if calc.warnings:
        st.markdown("**Warnings**")
        for warning in calc.warnings:
            st.write(f"- {warning}")

    with st.expander("Raw Layer 2 calculation JSON"):
        st.code(json.dumps(calc.model_dump(), indent=2), language="json")


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


def render_graph_visualization(
    payload: GraphPayload, calc: CalculationResult | None = None
) -> None:
    """Render an interactive network graph using pyvis.

    Highlights the computed path(s) from the Layer 2 calculation when available.
    """

    if not payload.nodes and not payload.edges:
        st.info("Upload SEC filing .htm or .txt files and run the pipeline to populate the map.")
        return

    try:
        from pyvis.network import Network
    except ImportError:
        st.warning("Install `pyvis` to enable the interactive graph view.")
        render_graph_preview(payload)
        return

    highlighted_edges: set[tuple[str, str]] = set()
    highlighted_nodes: set[str] = set()
    if calc is not None:
        for path in calc.paths:
            highlighted_nodes.update(path.nodes)
            for hop in path.hops:
                highlighted_edges.add((hop.source, hop.target))

    network = Network(
        height="520px",
        width="100%",
        directed=True,
        bgcolor="#0e1117",
        font_color="#fafafa",
        cdn_resources="remote",
    )
    network.barnes_hut()

    declared = {node.id for node in payload.nodes}
    for edge in payload.edges:
        declared.add(edge.source)
        declared.add(edge.target)

    node_types = {node.id: node.entity_type for node in payload.nodes}
    for node_id in declared:
        entity_type = node_types.get(node_id, "Subsidiary")
        color = _ENTITY_COLORS.get(entity_type, "#10b981")
        is_highlighted = node_id in highlighted_nodes
        network.add_node(
            node_id,
            label=node_id,
            title=f"{node_id} ({entity_type})",
            color="#ef4444" if is_highlighted else color,
            borderWidth=3 if is_highlighted else 1,
        )

    for edge in payload.edges:
        label_parts = []
        if edge.ownership_percentage is not None:
            label_parts.append(f"own {edge.ownership_percentage:.2f}")
        if edge.voting_power_percentage is not None:
            label_parts.append(f"vote {edge.voting_power_percentage:.2f}")
        if edge.liability_exposure_usd is not None:
            label_parts.append(f"liab ${edge.liability_exposure_usd:,.0f}")
        if edge.guarantee_cap_usd is not None:
            label_parts.append(f"cap ${edge.guarantee_cap_usd:,.0f}")

        is_highlighted = (edge.source, edge.target) in highlighted_edges
        network.add_edge(
            edge.source,
            edge.target,
            title=" | ".join(label_parts) if label_parts else None,
            label=label_parts[0] if label_parts else None,
            color="#ef4444" if is_highlighted else "#6b7280",
            width=4 if is_highlighted else 1,
        )

    try:
        with tempfile.TemporaryDirectory() as tmp_dir:
            html_path = Path(tmp_dir) / "graph.html"
            network.save_graph(str(html_path))
            html = html_path.read_text(encoding="utf-8")
        components.html(html, height=540, scrolling=False)
    except Exception:
        render_graph_preview(payload)
