"""Deterministic tests for the rule-agnostic Layer 2 engine (no LLM required)."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.engine import GovernanceGraphEngine
from core.routing import build_instruction
from core.sample_data import load_sample_payload
from models import CorporateEdge, CorporateNode, GraphPayload, TraversalInstruction


def _engine(payload: GraphPayload) -> GovernanceGraphEngine:
    engine = GovernanceGraphEngine()
    engine.load_graph_from_payload(payload)
    return engine


def test_multiply_and_cascade_single_path():
    payload = GraphPayload(
        nodes=[
            CorporateNode(id="A", entity_type="Parent"),
            CorporateNode(id="B"),
            CorporateNode(id="C"),
        ],
        edges=[
            CorporateEdge(source="A", target="B", ownership_percentage=0.8),
            CorporateEdge(source="B", target="C", ownership_percentage=0.5),
        ],
    )
    engine = _engine(payload)
    instruction = TraversalInstruction(
        start_node="A",
        target_node="C",
        edge_weight_to_track="ownership_percentage",
        aggregation_method="multiply_and_cascade",
    )
    result = engine.execute_rule_agnostic_cascade(instruction)

    assert result.computed is True
    assert result.final_value == pytest.approx(0.4)
    assert result.unit == "fraction (0.0-1.0)"
    assert len(result.paths) == 1


def test_sum_all_across_multiple_paths():
    payload = GraphPayload(
        nodes=[CorporateNode(id=n) for n in ("A", "B", "C", "D")],
        edges=[
            CorporateEdge(source="A", target="B", liability_exposure_usd=10.0),
            CorporateEdge(source="B", target="D", liability_exposure_usd=5.0),
            CorporateEdge(source="A", target="C", liability_exposure_usd=20.0),
            CorporateEdge(source="C", target="D", liability_exposure_usd=2.0),
        ],
    )
    engine = _engine(payload)
    instruction = TraversalInstruction(
        start_node="A",
        target_node="D",
        edge_weight_to_track="liability_exposure_usd",
        aggregation_method="sum_all",
    )
    result = engine.execute_rule_agnostic_cascade(instruction)

    # Path A->B->D = 15, path A->C->D = 22, summed across paths = 37.
    assert result.computed is True
    assert result.final_value == pytest.approx(37.0)
    assert len(result.paths) == 2


def test_guarantee_cap_constrains_path_value():
    payload = GraphPayload(
        nodes=[CorporateNode(id=n) for n in ("A", "B", "C")],
        edges=[
            CorporateEdge(
                source="A", target="B", liability_exposure_usd=30.0, guarantee_cap_usd=25.0
            ),
            CorporateEdge(
                source="B", target="C", liability_exposure_usd=30.0, guarantee_cap_usd=8.0
            ),
        ],
    )
    engine = _engine(payload)
    instruction = TraversalInstruction(
        start_node="A",
        target_node="C",
        edge_weight_to_track="liability_exposure_usd",
        edge_constraint="guarantee_cap_usd",
        aggregation_method="sum_all",
    )
    result = engine.execute_rule_agnostic_cascade(instruction)

    # Raw sum = 60, capped by min(25, 8) = 8.
    assert result.final_value == pytest.approx(8.0)
    assert result.paths[0].raw_value == pytest.approx(60.0)
    assert result.paths[0].constrained_value == pytest.approx(8.0)


def test_min_bottleneck():
    payload = GraphPayload(
        nodes=[CorporateNode(id=n) for n in ("A", "B", "C")],
        edges=[
            CorporateEdge(source="A", target="B", liability_exposure_usd=10.0),
            CorporateEdge(source="B", target="C", liability_exposure_usd=3.0),
        ],
    )
    engine = _engine(payload)
    instruction = TraversalInstruction(
        start_node="A",
        target_node="C",
        edge_weight_to_track="liability_exposure_usd",
        aggregation_method="min_bottleneck",
    )
    result = engine.execute_rule_agnostic_cascade(instruction)
    assert result.final_value == pytest.approx(3.0)


def test_missing_path_returns_not_computed():
    payload = GraphPayload(
        nodes=[CorporateNode(id=n) for n in ("A", "B")],
        edges=[CorporateEdge(source="A", target="B", liability_exposure_usd=10.0)],
    )
    engine = _engine(payload)
    instruction = TraversalInstruction(
        start_node="B",
        target_node="A",
        edge_weight_to_track="liability_exposure_usd",
        aggregation_method="sum_all",
    )
    result = engine.execute_rule_agnostic_cascade(instruction)
    assert result.computed is False
    assert result.final_value is None
    assert result.warnings


def test_duplicate_edges_are_merged():
    payload = GraphPayload(
        nodes=[CorporateNode(id="A"), CorporateNode(id="B")],
        edges=[
            CorporateEdge(source="A", target="B", liability_exposure_usd=10.0),
            CorporateEdge(source="A", target="B", guarantee_cap_usd=5.0),
        ],
    )
    engine = _engine(payload)
    assert engine.graph.number_of_edges() == 1
    edge = engine.graph.edges["A", "B"]
    assert edge["liability_exposure_usd"] == 10.0
    assert edge["guarantee_cap_usd"] == 5.0


def test_cycle_detection_reported():
    payload = GraphPayload(
        nodes=[CorporateNode(id="A"), CorporateNode(id="B")],
        edges=[
            CorporateEdge(source="A", target="B", liability_exposure_usd=10.0),
            CorporateEdge(source="B", target="A", liability_exposure_usd=10.0),
        ],
    )
    engine = _engine(payload)
    instruction = TraversalInstruction(
        start_node="A",
        target_node="B",
        edge_weight_to_track="liability_exposure_usd",
        aggregation_method="sum_all",
    )
    result = engine.execute_rule_agnostic_cascade(instruction)
    assert result.cycles_detected is not None


def test_routing_and_sample_fixture_end_to_end():
    payload = load_sample_payload()
    engine = _engine(payload)
    instruction = build_instruction("Financial Liability Cascade", engine.graph)
    assert instruction is not None
    assert instruction.edge_weight_to_track == "liability_exposure_usd"
    assert instruction.edge_constraint == "guarantee_cap_usd"

    result = engine.execute_rule_agnostic_cascade(instruction)
    assert result.computed is True
    assert result.final_value is not None
