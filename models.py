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