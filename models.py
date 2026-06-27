from pydantic import BaseModel, Field
from typing import List, Optional, Literal

class CorporateNode(BaseModel):
    id: str = Field(..., description="Unique legal name of the corporate entity.")
    jurisdiction: Optional[str] = Field(None, description="State or country of registration (e.g., Delaware).")
    entity_type: Literal["Parent", "Subsidiary", "Shell", "Operating Company"] = "Subsidiary"

class CorporateEdge(BaseModel):
    source: str = Field(..., description="The parent, guarantor, or upstream entity name.")
    target: str = Field(..., description="The subsidiary, debtor, or downstream entity name.")
    ownership_percentage: Optional[float] = Field(None, description="The raw equity stake percentage (0.0 to 1.0).")
    voting_power_percentage: Optional[float] = Field(None, description="The legal voting power percentage (0.0 to 1.0).")
    liability_exposure_usd: Optional[float] = Field(None, description="Direct dollar amount of debt or liability.")
    guarantee_cap_usd: Optional[float] = Field(None, description="The maximum legal limit of a parent guarantee liability.")

class GraphPayload(BaseModel):
    nodes: List[CorporateNode]
    edges: List[CorporateEdge]


class ExtractionSegment(BaseModel):
    chunk_index: int = Field(..., description="Zero-based chunk index within the uploaded document corpus.")
    source_documents: List[str] = Field(
        ...,
        description="Filenames represented in this chunk.",
    )
    token_start: int = Field(..., description="Inclusive token offset for the chunk within the corpus.")
    token_end: int = Field(..., description="Exclusive token offset for the chunk within the corpus.")
    payload: GraphPayload = Field(
        ...,
        description="Clean JSON extraction payload for this chunk, matching CorporateNode and CorporateEdge schemas.",
    )


class ExtractionResult(BaseModel):
    graph: GraphPayload = Field(..., description="Structured entity-relationship graph extracted from the documents.")
    extraction_summary: str = Field(
        ...,
        description="Brief summary of extracted entities, relationships, and any ambiguities or missing data.",
    )
    documents_processed: List[str] = Field(
        ...,
        description="Filenames of the source documents represented in this extraction.",
    )
    segments: List[ExtractionSegment] = Field(
        default_factory=list,
        description="Per-chunk clean JSON extraction payloads produced by Layer 1.",
    )


class SynthesisResponse(BaseModel):
    answer: str = Field(..., description="Audit-ready natural language response to the user query.")
    entities_referenced: List[str] = Field(
        default_factory=list,
        description="Corporate entities cited in the answer.",
    )
    assumptions: List[str] = Field(
        default_factory=list,
        description="Explicit assumptions made because the graph computation layer was not run.",
    )
    data_gaps: List[str] = Field(
        default_factory=list,
        description="Missing or unclear information that limits the analysis.",
    )


class TraversalInstruction(BaseModel):
    start_node: str
    target_node: str
    edge_weight_to_track: Literal["ownership_percentage", "voting_power_percentage", "liability_exposure_usd"]
    edge_constraint: Optional[Literal["guarantee_cap_usd"]] = None
    aggregation_method: Literal["multiply_and_cascade", "sum_all", "min_bottleneck"]


class QueryRoute(BaseModel):
    """Stage 1 routing output: which mechanism and entities a query targets."""

    mechanism: Literal[
        "Financial Liability Cascade",
        "Voting Power Structure",
        "Tax Leakage Tracing",
    ] = Field(
        ...,
        description="The corporate mechanism the question is about, used to pick the metric and aggregation.",
    )
    start_node: Optional[str] = Field(
        None,
        description="Upstream/controlling entity to traverse FROM, matched to an existing graph node id, or null if unclear.",
    )
    target_node: Optional[str] = Field(
        None,
        description="Downstream/debtor entity to traverse TO, matched to an existing graph node id, or null if unclear.",
    )
    reasoning: str = Field(
        "",
        description="Brief justification for the chosen mechanism and entities.",
    )


class PathHop(BaseModel):
    """A single directed edge traversed within a path."""

    source: str = Field(..., description="Upstream entity for this hop.")
    target: str = Field(..., description="Downstream entity for this hop.")
    weight_field: str = Field(..., description="Edge attribute tracked for this hop.")
    weight_value: Optional[float] = Field(
        None, description="Value of the tracked attribute on this edge, if present."
    )
    constraint_field: Optional[str] = Field(
        None, description="Edge attribute applied as a ceiling/constraint, if any."
    )
    constraint_value: Optional[float] = Field(
        None, description="Value of the constraint attribute on this edge, if present."
    )


class TraversalPath(BaseModel):
    """One fully resolved route from start to target, with its computed contribution."""

    nodes: List[str] = Field(..., description="Ordered entity names visited along the path.")
    hops: List[PathHop] = Field(default_factory=list, description="Per-edge detail for the path.")
    raw_value: Optional[float] = Field(
        None, description="Aggregated value before any constraint cap was applied."
    )
    constrained_value: Optional[float] = Field(
        None, description="Aggregated value after applying the edge constraint cap, if any."
    )
    formula: str = Field(
        "", description="Human-readable arithmetic showing how the path value was derived."
    )
    notes: List[str] = Field(
        default_factory=list, description="Per-path warnings such as missing weights."
    )


class CalculationResult(BaseModel):
    """Deterministic Layer 2 output handed to the synthesis layer."""

    computed: bool = Field(
        ..., description="True when the deterministic graph engine produced a result."
    )
    instruction: TraversalInstruction = Field(
        ..., description="The traversal instruction that was executed."
    )
    aggregation_method: str = Field(..., description="Aggregation method applied across paths.")
    paths: List[TraversalPath] = Field(
        default_factory=list, description="All resolved start-to-target paths."
    )
    final_value: Optional[float] = Field(
        None, description="Final computed metric across all paths after aggregation."
    )
    unit: str = Field(
        "", description="Unit of the final value, e.g. 'USD' or 'fraction (0.0-1.0)'."
    )
    formula_summary: str = Field(
        "", description="Plain-language description of how the final value was derived."
    )
    cycles_detected: Optional[List[List[str]]] = Field(
        None, description="Cycles found in the graph that may invalidate DAG assumptions."
    )
    warnings: List[str] = Field(
        default_factory=list, description="Structural or data-quality warnings."
    )