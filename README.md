# Rule-Agnostic Graph-RAG Engine for Corporate Governance

A high-performance, deterministic "LLM-Graph-LLM Sandwich" pipeline designed to parse complex, unstructured corporate finance and legal documents into Directed Acyclic Graphs (DAGs) for cascading risk, liability, and exposure analysis.

Standard Vector RAG struggles with relational, multi-tiered corporate governance queries (e.g., tracking nested ownership liabilities or parent-subsidiary debt guarantees). This engine bypasses semantic vector distance lookups in favor of strict, mathematical graph traversal to compute financial and legal exposure deterministically.

## Key Features

* Rule-Agnostic Graph Core: The underlying network infrastructure acts as a generic mathematical calculator, accepting dynamic corporate mechanism rules and mapping them across network edges.
* Strict Schema Enforcement: Powered by `instructor` and `Pydantic` to force high-capability LLMs into strict JSON data contracts, guaranteeing the graph core never receives malformed syntactical data.
* Deterministic Computation: Utilizes `NetworkX` for explicit path routing, depth-first searches, and reverse topological sorts to calculate cascading metrics from bottom-level nodes up to parent entities.
* Interactive Visualization: Streamlit workspace featuring dynamic, in-browser network graph rendering via Pyvis/Streamlit-Agraph for audit-ready transparency.

## System Architecture

The pipeline operates in a decoupled, three-stage execution loop.

1. Extraction & Routing Interface: Parses natural language queries or raw text (e.g., SEC EDGAR filings), extracting entities (Nodes) and relationships (Edges).
2. NetworkX Calculation Core: A pure Python backend managing the DAG. It executes recursive mathematical formulas across edge attributes (e.g., ownership percentages, guarantee caps) to compute cascading risks.
3. Synthesis & Visualization Layer: Passes the structured path logs and calculated mathematical outputs to a synthesis LLM to compile an audit-ready response, while simultaneously rendering the network UI.

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

3. Configure Environment Variables. Create a .env file in the root directory:
    ```bash
    OPENAI_API_KEY=api_key
    ```

4. Run the application:
    ```bash
    streamlit run app.py
    ```

## Proof of Concept (PoC) Milestone

The initial milestone targets Parent Guarantees of Subsidiary Debt using raw .txt and .htm material credit agreements extracted from the SEC EDGAR Database. The core engine extracts the Principal Debtor, Guarantor, and Total Exposure, cascading these liabilities across multi-tiered holding configurations.

## Technical Stack

* Language: `Python`
* Graph Mathematics: `networkx`
* LLM Orchestration: OpenAI API (`gpt-4o` / `gpt-4o-mini`)
* Type Enforcement: `instructor` + `Pydantic`
* Frontend UI & Rendering: `Streamlit` + `pyvis` / `streamlit-agraph`

## Contributors

* Kaylum Truong ([@DrvKrv](https://github.com/DrvKrv))
    * Role: Core Backend Architecture, API Orchestration, & Deterministic Validation Gates

* Alice Yang ([@kep1r](https://github.com/kep1r))
    * Role: System Scoping, Compliance Workflow Mapping, & Interface Logic

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.