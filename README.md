# Rule-Agnostic Graph-RAG Engine for Corporate Governance

A Streamlit proof of concept for an "LLM-Graph-LLM Sandwich" pipeline that parses SEC EDGAR filing text into strict corporate entity/relationship JSON and prepares that data for deterministic graph analysis.

Standard vector RAG struggles with relational, multi-hop corporate governance questions, such as nested ownership liability or parent-subsidiary guarantee exposure. This project separates probabilistic text extraction from deterministic graph computation so legal/financial calculations can be performed by Python and NetworkX rather than by an LLM.

## Current Repository Status

All three layers are implemented and wired into the runtime pipeline:

* **Layer 1** extracts entities/relationships from `.htm` and `.txt` SEC filings into strict JSON.
* **Layer 2** is a deterministic, rule-agnostic NetworkX engine that merges the extracted graph, validates structure (cycles), and computes cascading metrics from a parameterised `TraversalInstruction`.
* **Layer 3** synthesizes an audit-ready answer that explains the Layer 2 computed figure using its path logs and formulas.

A bundled **Demo Mode** runs Layer 2 on a multi-tiered fixture graph with no API key and no uploads, so the deterministic math can be validated in isolation.

## Key Features

* SEC filing ingestion for `.htm` and `.txt` files only. PDF uploads are not accepted.
* Robust text extraction: `.htm` and markup-laden `.txt` "complete submission" files are both cleaned through BeautifulSoup (scripts, styles, HTML/XBRL noise removed) before chunking.
* Up to 10 uploaded SEC filing files per run.
* HTML cleanup with BeautifulSoup to remove scripts, styles, layout markup, and Inline XBRL/HTML noise before chunking.
* Token-window chunking: 4,000-token chunks with 500-token overlap via `tiktoken`.
* GPT-5-series structured extraction using the OpenAI Responses API.
* Strict Pydantic contracts for `CorporateNode`, `CorporateEdge`, `GraphPayload`, `ExtractionSegment`, and `ExtractionResult`.
* GPT-5-series synthesis for audit-style natural language output.
* Deterministic NetworkX graph engine for Layer 2 cascade computation.
* Interactive `pyvis` network visualization with computed-path highlighting.

## System Architecture

The intended pipeline operates in three decoupled layers.

0. **Stage 1 Routing (GPT-5)**
   * `core/llm_layers.py::route_query()` reads the natural-language question and the list of entities present in the extracted graph, then returns a `QueryRoute` (mechanism + start/target entities) constrained to real node names.
   * The result is converted into a `TraversalInstruction` via `core/routing.py`. Manual sidebar overrides take priority, and deterministic auto-selection is used whenever the router is unsure or unavailable.

1. **Layer 1: Map-Reduce Extraction**
   * Input: `.htm` or `.txt` SEC filing files.
   * HTML files are parsed as raw filing text with BeautifulSoup, not as visual page layouts.
   * Text is segmented into 4,000-token windows with 500-token overlap.
   * Each chunk is independently sent to GPT-5.5 via the OpenAI Responses API.
   * Each chunk returns a clean JSON `GraphPayload` containing `CorporateNode` and `CorporateEdge` records.

2. **Layer 2: NetworkX Calculation Core**
   * Implemented in `core/engine.py` (`GovernanceGraphEngine`).
   * Loads the extracted `GraphPayload` into a directed graph, merges duplicate nodes/edges from multi-chunk extraction, detects cycles, enumerates all simple paths between two entities, and computes deterministic metrics.
   * Rule-agnostic: it reads whichever edge attribute the `TraversalInstruction` names (`edge_weight_to_track`), applies an optional ceiling (`edge_constraint`, e.g. `guarantee_cap_usd`), and aggregates with `multiply_and_cascade`, `sum_all`, or `min_bottleneck`.
   * Non-LLM and stateless with respect to document ingestion.
   * `core/routing.py` translates a routing domain + graph into a concrete `TraversalInstruction`. Start/target entities come from the Stage 1 GPT-5 router, manual UI overrides, or deterministic auto-selection (in that priority order).

3. **Layer 3: Synthesis & Display**
   * GPT-5.5 synthesis that treats the Layer 2 computed result, path logs, and formulas as authoritative ground truth and explains exactly how the figure was derived.
   * The UI displays extraction tables, an interactive `pyvis` network graph (with the computed path highlighted), the Layer 2 computation with path logs/warnings, raw JSON, and the synthesis output.

## Getting Started

### Prerequisites

* Python 3.11+
* OpenAI API Key (configured in environment)

### Installation & Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/DrvKrv/rule-agnostic-graph-rag.git
   cd rule-agnostic-graph-rag
   ```

2. Set up a virtual environment and install dependencies:
    ```bash
    python3 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt
    ```

3. Configure environment variables. Create a `.env` file in the root directory:
    ```bash
    OPENAI_API_KEY=api_key
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## Proof of Concept (PoC) Milestone

The current milestone targets parent guarantees of subsidiary debt using `.txt` and `.htm` material agreements or subsidiary exhibits from SEC EDGAR. Layer 1 extracts local single-hop relationships such as parent ownership, debtor/guarantor relationships, liability exposure, and guarantee caps. Layer 2 computes cascaded liability exposure across the graph, capping each path by its guarantee limit.

## Technical Stack

* Language: `Python`
* Frontend: `Streamlit`
* Filing Parsing: `BeautifulSoup`
* Tokenization: `tiktoken`
* LLM Orchestration: OpenAI API, GPT-5 series only (extraction, fast query routing, and synthesis)
* Structured Outputs: OpenAI Responses API + `Pydantic`
* Synthesis: GPT-5 series
* Graph Mathematics: `networkx` (deterministic Layer 2)
* Graph Visualization: `pyvis`

## Demo Mode (no API key required)

Layer 2 is fully deterministic, so you can validate the graph math without OpenAI access:

1. Run `streamlit run app.py`.
2. In the sidebar, choose a Routing Domain and click **Run Layer 2 Demo (no API key)**.
3. The app loads a bundled multi-tiered guarantee graph (`data/sample_graph.json`), runs the cascade, highlights the computed path in the network view, and shows path logs, formulas, and warnings.

You can also exercise the engine programmatically:

```python
from core.sample_data import load_sample_payload
from core.engine import GovernanceGraphEngine
from core.routing import build_instruction

engine = GovernanceGraphEngine()
engine.load_graph_from_payload(load_sample_payload())
instruction = build_instruction("Financial Liability Cascade", engine.graph)
result = engine.execute_rule_agnostic_cascade(instruction)
print(result.final_value, result.unit, result.formula_summary)
```

## Running Tests

Deterministic Layer 2 tests require no API key:

```bash
python -m pytest tests/
```

## Architecture Notes for Contributors

The data contracts in `models.py` are the source of truth between layers:

* `CorporateNode` / `CorporateEdge` describe graph nodes and directed edges (the edge attribute vocabulary is fixed).
* `GraphPayload` is what Layer 1 hands to Layer 2.
* `TraversalInstruction` tells the rule-agnostic engine which attribute to track, which constraint to cap with, and how to aggregate.
* `CalculationResult` / `TraversalPath` / `PathHop` are the structured Layer 2 output (path logs, formulas, warnings, final totals) consumed by Layer 3.
* `QueryRoute` is the Stage 1 routing output (mechanism + start/target entities) extracted from the user's natural-language question.

Layer 2 is intentionally generic: new corporate mechanisms are added as new edge attributes plus new routing entries in `core/routing.py`, not as new branches inside `GovernanceGraphEngine`.

## Contributors

* Kaylum Truong ([@DrvKrv](https://github.com/DrvKrv))
    * Role: Core Backend Architecture, API Orchestration, & Deterministic Validation Gates

* Alice Yang ([@kep1r](https://github.com/kep1r))
    * Role: System Scoping, Compliance Workflow Mapping, & Interface Logic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.