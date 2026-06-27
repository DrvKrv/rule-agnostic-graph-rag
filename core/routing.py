"""Query routing: turn a domain + graph into a deterministic TraversalInstruction.

The graph engine (Layer 2) is rule-agnostic, so *something* must translate a
business question into the concrete metric/aggregation to run. For the proof of
concept this mapping is a small, explicit table rather than an LLM call, which
keeps the system fully testable and runnable without an API key. A routing LLM
can later populate the same ``TraversalInstruction`` fields.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from models import TraversalInstruction

# Domain -> (weight field, optional constraint field, aggregation method).
DOMAIN_RULES: dict[str, tuple[str, Optional[str], str]] = {
    "Financial Liability Cascade": (
        "liability_exposure_usd",
        "guarantee_cap_usd",
        "sum_all",
    ),
    "Voting Power Structure": (
        "voting_power_percentage",
        None,
        "multiply_and_cascade",
    ),
    "Tax Leakage Tracing": (
        "liability_exposure_usd",
        None,
        "sum_all",
    ),
}

DEFAULT_DOMAIN = "Financial Liability Cascade"


def _pick_root(graph: nx.DiGraph) -> Optional[str]:
    """Best-guess upstream entity: prefer a Parent with no incoming edges."""
    roots = [n for n in graph.nodes if graph.in_degree(n) == 0]
    if not roots:
        return next(iter(graph.nodes), None)
    parents = [n for n in roots if graph.nodes[n].get("entity_type") == "Parent"]
    candidates = parents or roots
    # Deterministic and meaningful: the root reaching the most other entities.
    return max(candidates, key=lambda n: len(nx.descendants(graph, n)))


def _pick_leaf(graph: nx.DiGraph, start: Optional[str]) -> Optional[str]:
    """Best-guess downstream entity: a reachable sink furthest from the root."""
    if start is None:
        leaves = [n for n in graph.nodes if graph.out_degree(n) == 0]
        return leaves[0] if leaves else None

    reachable = nx.descendants(graph, start)
    sinks = [n for n in reachable if graph.out_degree(n) == 0]
    candidates = sinks or list(reachable)
    if not candidates:
        return None
    # Furthest reachable sink makes for the most illustrative cascade.
    return max(candidates, key=lambda n: len(nx.ancestors(graph, n)))


def build_instruction(
    domain: str,
    graph: nx.DiGraph,
    start_node: Optional[str] = None,
    target_node: Optional[str] = None,
) -> Optional[TraversalInstruction]:
    """Construct a TraversalInstruction for ``domain`` over ``graph``.

    ``start_node`` / ``target_node`` override automatic selection when provided.
    Returns ``None`` when the graph is too small to traverse.
    """

    if graph.number_of_nodes() == 0:
        return None

    weight_field, constraint_field, aggregation = DOMAIN_RULES.get(
        domain, DOMAIN_RULES[DEFAULT_DOMAIN]
    )

    start = start_node or _pick_root(graph)
    target = target_node or _pick_leaf(graph, start)

    if not start or not target or start == target:
        return None

    return TraversalInstruction(
        start_node=start,
        target_node=target,
        edge_weight_to_track=weight_field,
        edge_constraint=constraint_field,
        aggregation_method=aggregation,
    )
