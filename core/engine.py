"""Layer 2: deterministic, rule-agnostic graph calculation core.

This module is intentionally free of any LLM / network calls. It loads the
``GraphPayload`` produced by Layer 1 into a NetworkX ``DiGraph`` and computes
cascading corporate metrics based on a parameterised ``TraversalInstruction``.

The engine is "rule-agnostic": it has no knowledge of "parent guarantees",
"voting control" or "tax leakage" as special cases. Instead it reads whichever
edge attribute the instruction names (``edge_weight_to_track``), applies an
optional ceiling (``edge_constraint``) and aggregates path values with the
requested ``aggregation_method``. New corporate mechanisms are expressed as new
instructions and edge attributes, not as new code paths here.
"""

from __future__ import annotations

from typing import Optional

import networkx as nx

from models import (
    CalculationResult,
    CorporateEdge,
    GraphPayload,
    PathHop,
    TraversalInstruction,
    TraversalPath,
)

# Aggregation methods whose neutral metric is a fraction rather than a dollar amount.
_FRACTION_WEIGHTS = {"ownership_percentage", "voting_power_percentage"}


def _format_number(value: Optional[float], is_fraction: bool) -> str:
    if value is None:
        return "n/a"
    if is_fraction:
        return f"{value:.4f}"
    return f"{value:,.2f}"


class GovernanceGraphEngine:
    """Loads a corporate graph and runs deterministic cascade computations."""

    def __init__(self) -> None:
        self.graph = nx.DiGraph()

    # ------------------------------------------------------------------
    # Graph construction
    # ------------------------------------------------------------------
    def load_graph_from_payload(self, payload: GraphPayload) -> None:
        """Load a (possibly multi-chunk, duplicate-heavy) payload into the graph.

        Layer 1 emits stateless, per-chunk payloads, so the same entity or
        relationship can appear many times. We merge duplicates deterministically:
        node attributes are filled where missing, and parallel edges between the
        same pair are combined attribute-by-attribute (non-null wins; on conflict
        the larger magnitude is kept and a note is recorded on the edge).
        """

        self.graph.clear()
        self._merge_warnings: list[str] = []

        # Build a case-insensitive canonical-name map across every entity that
        # appears as a node or an edge endpoint. Stateless per-chunk extraction
        # routinely emits the same company under different casing (e.g.
        # "Freedom VCM, Inc." and "FREEDOM VCM, INC."); without canonicalisation
        # these fragment into separate nodes and break path traversal.
        self._canonical: dict[str, str] = {}
        endpoint_names: list[str] = [node.id for node in payload.nodes]
        for edge in payload.edges:
            endpoint_names.extend([edge.source, edge.target])
        for name in endpoint_names:
            self._register_canonical(name)

        for node in payload.nodes:
            attrs = node.model_dump()
            attrs["id"] = self._canonical_name(node.id)
            self._upsert_node(self._canonical_name(node.id), attrs)

        for edge in payload.edges:
            self._upsert_edge(edge)

    @staticmethod
    def _canonical_key(name: str) -> str:
        return " ".join((name or "").split()).casefold()

    def _register_canonical(self, name: str) -> None:
        """Pick a stable display name for each case-insensitive entity key.

        Prefer a variant that is not fully upper-cased (real filings use mixed
        case for the canonical legal name and reserve all-caps for headings/
        signature blocks); fall back to the first variant seen.
        """
        if not name:
            return
        key = self._canonical_key(name)
        if not key:
            return
        current = self._canonical.get(key)
        if current is None:
            self._canonical[key] = name
            return
        if current.isupper() and not name.isupper():
            self._canonical[key] = name

    def _canonical_name(self, name: str) -> str:
        return self._canonical.get(self._canonical_key(name), name)

    def _upsert_node(self, node_id: str, attrs: dict) -> None:
        if not node_id:
            return
        if node_id in self.graph:
            existing = self.graph.nodes[node_id]
            for key, value in attrs.items():
                if existing.get(key) in (None, "Subsidiary") and value not in (None,):
                    existing[key] = value
        else:
            self.graph.add_node(node_id, **attrs)

    def _upsert_edge(self, edge: CorporateEdge) -> None:
        source = self._canonical_name(edge.source)
        target = self._canonical_name(edge.target)
        if not source or not target:
            return

        # Ensure endpoints exist even if Layer 1 only referenced them via an edge.
        for endpoint in (source, target):
            if endpoint not in self.graph:
                self.graph.add_node(endpoint, id=endpoint, entity_type="Subsidiary")

        new_attrs = edge.model_dump(exclude={"source", "target"})

        if self.graph.has_edge(source, target):
            existing = self.graph.edges[source, target]
            for key, value in new_attrs.items():
                if value is None:
                    continue
                current = existing.get(key)
                if current is None:
                    existing[key] = value
                elif abs(value) > abs(current):
                    self._merge_warnings.append(
                        f"Conflicting '{key}' on edge {source} -> {target}: "
                        f"kept {value} over {current}."
                    )
                    existing[key] = value
        else:
            self.graph.add_edge(source, target, **new_attrs)

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------
    def detect_cycles(self) -> Optional[list]:
        """Return the first cycle as a list of (u, v, direction) edges, or None."""
        try:
            return nx.find_cycle(self.graph, orientation="original")
        except nx.NetworkXNoCycle:
            return None

    def _cycles_as_node_lists(self) -> Optional[list[list[str]]]:
        cycle = self.detect_cycles()
        if not cycle:
            return None
        nodes = [edge[0] for edge in cycle]
        if cycle:
            nodes.append(cycle[-1][1])
        return [nodes]

    # ------------------------------------------------------------------
    # Core rule-agnostic cascade
    # ------------------------------------------------------------------
    def execute_rule_agnostic_cascade(
        self, instructions: TraversalInstruction
    ) -> CalculationResult:
        """Compute a deterministic metric from ``start_node`` to ``target_node``.

        The method is generic over the tracked attribute and aggregation strategy:

        * ``multiply_and_cascade`` -- product of hop weights along each path, then
          summed across paths (e.g. effective ownership through intermediaries).
        * ``sum_all`` -- sum of hop weights along each path, then summed across
          paths (e.g. total liability exposure routed through subsidiaries).
        * ``min_bottleneck`` -- minimum hop weight along each path, then summed
          across paths (e.g. the limiting capacity on each route).

        An optional ``edge_constraint`` (e.g. ``guarantee_cap_usd``) caps the value
        contributed by each path. All arithmetic is recorded as path logs and a
        plain-language formula for audit purposes.
        """

        weight_field = instructions.edge_weight_to_track
        constraint_field = instructions.edge_constraint
        method = instructions.aggregation_method
        is_fraction = weight_field in _FRACTION_WEIGHTS

        warnings: list[str] = list(getattr(self, "_merge_warnings", []))
        cycles = self._cycles_as_node_lists()
        if cycles:
            warnings.append(
                "Cycle(s) detected; the graph is not a pure DAG. Cascade math may "
                "double-count along cyclic routes."
            )

        start = instructions.start_node
        target = instructions.target_node

        for label, node in (("start_node", start), ("target_node", target)):
            if node not in self.graph:
                warnings.append(f"{label} '{node}' is not present in the extracted graph.")

        unit = "fraction (0.0-1.0)" if is_fraction else "USD"

        if start not in self.graph or target not in self.graph:
            return CalculationResult(
                computed=False,
                instruction=instructions,
                aggregation_method=method,
                paths=[],
                final_value=None,
                unit=unit,
                formula_summary="No computation: start or target entity missing from graph.",
                cycles_detected=cycles,
                warnings=warnings,
            )

        try:
            raw_paths = list(
                nx.all_simple_paths(self.graph, source=start, target=target)
            )
        except nx.NodeNotFound:
            raw_paths = []

        if not raw_paths:
            warnings.append(
                f"No directed path connects '{start}' to '{target}'. "
                "They may be unrelated or the edge direction may be reversed."
            )
            return CalculationResult(
                computed=False,
                instruction=instructions,
                aggregation_method=method,
                paths=[],
                final_value=None,
                unit=unit,
                formula_summary="No directed path between the requested entities.",
                cycles_detected=cycles,
                warnings=warnings,
            )

        resolved_paths: list[TraversalPath] = []
        path_contributions: list[float] = []

        for node_sequence in raw_paths:
            resolved = self._evaluate_path(
                node_sequence, weight_field, constraint_field, method, is_fraction
            )
            resolved_paths.append(resolved)
            if resolved.constrained_value is not None:
                path_contributions.append(resolved.constrained_value)

        final_value, summary = self._aggregate_paths(
            path_contributions, method, is_fraction, len(resolved_paths)
        )

        if not path_contributions:
            warnings.append(
                "Paths exist but none carried the requested metric; values were null on every hop."
            )

        return CalculationResult(
            computed=final_value is not None,
            instruction=instructions,
            aggregation_method=method,
            paths=resolved_paths,
            final_value=final_value,
            unit=unit,
            formula_summary=summary,
            cycles_detected=cycles,
            warnings=warnings,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    def _evaluate_path(
        self,
        node_sequence: list[str],
        weight_field: str,
        constraint_field: Optional[str],
        method: str,
        is_fraction: bool,
    ) -> TraversalPath:
        hops: list[PathHop] = []
        weights: list[float] = []
        constraints: list[float] = []
        notes: list[str] = []

        for source, target in zip(node_sequence, node_sequence[1:]):
            edge_data = self.graph.edges[source, target]
            weight_value = edge_data.get(weight_field)
            constraint_value = (
                edge_data.get(constraint_field) if constraint_field else None
            )

            hops.append(
                PathHop(
                    source=source,
                    target=target,
                    weight_field=weight_field,
                    weight_value=weight_value,
                    constraint_field=constraint_field,
                    constraint_value=constraint_value,
                )
            )

            if weight_value is None and constraint_value is not None:
                # The tracked metric is absent but a constraint (e.g. a guarantee
                # cap) is present. In corporate filings a guarantee cap with no
                # separately stated liability IS the bounded exposure for that
                # hop, so use it as a proxy weight rather than discarding the hop.
                weights.append(constraint_value)
                notes.append(
                    f"Edge {source} -> {target} has no '{weight_field}'; used "
                    f"'{constraint_field}' value {_format_number(constraint_value, is_fraction)} "
                    "as an exposure proxy."
                )
            elif weight_value is None:
                notes.append(
                    f"Edge {source} -> {target} has no '{weight_field}' value."
                )
            else:
                weights.append(weight_value)
            if constraint_value is not None:
                constraints.append(constraint_value)

        raw_value, formula = self._aggregate_single_path(
            weights, method, is_fraction
        )

        constrained_value = raw_value
        if raw_value is not None and constraints:
            cap = min(constraints)
            if cap < raw_value:
                constrained_value = cap
                formula += (
                    f" -> capped at {_format_number(cap, is_fraction)} "
                    f"by {constraint_field}"
                )

        return TraversalPath(
            nodes=list(node_sequence),
            hops=hops,
            raw_value=raw_value,
            constrained_value=constrained_value,
            formula=formula,
            notes=notes,
        )

    @staticmethod
    def _aggregate_single_path(
        weights: list[float], method: str, is_fraction: bool
    ) -> tuple[Optional[float], str]:
        if not weights:
            return None, "no metric values on this path"

        rendered = [_format_number(w, is_fraction) for w in weights]

        if method == "multiply_and_cascade":
            value = 1.0
            for weight in weights:
                value *= weight
            return value, " * ".join(rendered) + f" = {_format_number(value, is_fraction)}"

        if method == "sum_all":
            value = sum(weights)
            return value, " + ".join(rendered) + f" = {_format_number(value, is_fraction)}"

        if method == "min_bottleneck":
            value = min(weights)
            return value, f"min({', '.join(rendered)}) = {_format_number(value, is_fraction)}"

        return None, f"unknown aggregation method '{method}'"

    @staticmethod
    def _aggregate_paths(
        contributions: list[float],
        method: str,
        is_fraction: bool,
        path_count: int,
    ) -> tuple[Optional[float], str]:
        if not contributions:
            return None, "No path carried the requested metric; nothing to aggregate."

        rendered = [_format_number(c, is_fraction) for c in contributions]

        # Across multiple paths we accumulate exposure/ownership additively, since
        # distinct routes represent distinct contributions to the target metric.
        total = sum(contributions)
        if path_count == 1:
            return total, f"Single path value = {_format_number(total, is_fraction)}."

        return (
            total,
            f"Aggregated {path_count} path(s) by summation: "
            + " + ".join(rendered)
            + f" = {_format_number(total, is_fraction)}.",
        )
