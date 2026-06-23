# Rule-Agnostic Graph-RAG Engine for Corporate Governance

A Streamlit proof of concept for an "LLM-Graph-LLM Sandwich" pipeline that parses SEC EDGAR filing text into strict corporate entity/relationship JSON and prepares that data for deterministic graph analysis.

Standard vector RAG struggles with relational, multi-hop corporate governance questions, such as nested ownership liability or parent-subsidiary guarantee exposure. This project separates probabilistic text extraction from deterministic graph computation so legal/financial calculations can be performed by Python and NetworkX rather than by an LLM.

## Current Repository Status

The repository currently implements Layer 1 extraction and Layer 3 synthesis. Layer 2 exists as a NetworkX skeleton and is intentionally not wired into the runtime pipeline yet.

## Key Features

* SEC filing ingestion for `.htm` and `.txt` files only. PDF uploads are not accepted.
* Up to 10 uploaded SEC filing files per run.
* HTML cleanup with BeautifulSoup to remove scripts, styles, layout markup, and Inline XBRL/HTML noise before chunking.
* Token-window chunking: 4,000-token chunks with 500-token overlap via `tiktoken`.
* GPT-5-series structured extraction using the OpenAI Responses API.
* Strict Pydantic contracts for `CorporateNode`, `CorporateEdge`, `GraphPayload`, `ExtractionSegment`, and `ExtractionResult`.
* GPT-5-series synthesis for audit-style natural language output.
* NetworkX graph engine skeleton reserved for deterministic Layer 2 development.

## System Architecture

The intended pipeline operates in three decoupled layers.

1. **Layer 1: Map-Reduce Extraction**
   * Input: `.htm` or `.txt` SEC filing files.
   * HTML files are parsed as raw filing text with BeautifulSoup, not as visual page layouts.
   * Text is segmented into 4,000-token windows with 500-token overlap.
   * Each chunk is independently sent to GPT-5.5 via the OpenAI Responses API.
   * Each chunk returns a clean JSON `GraphPayload` containing `CorporateNode` and `CorporateEdge` records.

2. **Layer 2: NetworkX Calculation Core**
   * Current status: skeleton only in `core/engine.py`.
   * Intended role: load the extracted `GraphPayload` into a directed graph, validate DAG structure, detect cycles, traverse paths, and compute deterministic corporate exposure metrics.
   * This layer should remain non-LLM and stateless with respect to document ingestion.

3. **Layer 3: Synthesis & Display**
   * Current status: GPT-5.5 synthesis over extracted graph payloads.
   * Future role: consume Layer 2 calculation results and path logs, then produce an audit-ready explanation.
   * Current UI displays extraction tables, raw graph JSON, segment JSON, and synthesis output.

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

The current milestone targets parent guarantees of subsidiary debt using `.txt` and `.htm` material agreements or subsidiary exhibits from SEC EDGAR. Layer 1 extracts local single-hop relationships such as parent ownership, debtor/guarantor relationships, liability exposure, and guarantee caps. Layer 2 will later compute cascaded liability exposure across the graph.

## Technical Stack

* Language: `Python`
* Frontend: `Streamlit`
* Filing Parsing: `BeautifulSoup`
* Tokenization: `tiktoken`
* LLM Orchestration: OpenAI API, GPT-5 series only
* Structured Outputs: OpenAI Responses API + `Pydantic`
* Synthesis: GPT-5 series
* Graph Mathematics: `networkx` skeleton for Layer 2

## Layer 2 Development Notes

If you are building Layer 2, start with these files:

* `models.py`: defines the contracts Layer 2 consumes.
  * `CorporateNode` and `CorporateEdge` describe graph nodes and directed edges.
  * `GraphPayload` is the object Layer 1 gives to Layer 2.
  * `TraversalInstruction` is the current starter schema for telling Layer 2 what metric/rule to run.
* `core/engine.py`: the intended home for deterministic NetworkX graph processing.
  * `load_graph_from_payload()` already loads `GraphPayload` into a `networkx.DiGraph`.
  * `detect_cycles()` is the starting point for DAG validation.
  * `execute_rule_agnostic_cascade()` is currently the main unimplemented Layer 2 function.
* `core/llm_layers.py`: Layer 1 returns `ExtractionResult.graph` and `ExtractionResult.segments`.
  * `ExtractionResult.graph` is a flattened preview payload from all chunks.
  * `ExtractionResult.segments` preserves individual chunk-level payloads.
* `app.py`: currently sends `extraction_result.graph` directly to Layer 3. Once Layer 2 is built, insert the graph engine call between extraction and synthesis.

Layer 3 currently expects a `GraphPayload`, source filenames, and an extraction summary. Once Layer 2 exists, Layer 3 should instead receive structured calculation output: path logs, formulas applied, warnings, and final computed exposure totals.

## Contributors

* Kaylum Truong ([@DrvKrv](https://github.com/DrvKrv))
    * Role: Core Backend Architecture, API Orchestration, & Deterministic Validation Gates

* Alice Yang ([@kep1r](https://github.com/kep1r))
    * Role: System Scoping, Compliance Workflow Mapping, & Interface Logic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.