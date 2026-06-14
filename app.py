import streamlit as st
import networkx as nx
from core.engine import GovernanceGraphEngine

st.set_page_config(page_title="Graph-RAG Corporate Governance Engine", layout="wide")

st.title("Rule-Agnostic Graph-RAG Engine")
st.subheader("Corporate Governance & Liability Cascade Analyzer")

with st.sidebar:
    st.header("1. Document Ingestion")
    uploaded_file = st.file_uploader("Upload SEC EDGAR Filing (.txt, .htm, .pdf)", type=["txt", "htm", "html", "pdf"])
    
    st.header("2. Execution Parameters")
    query_topic = st.selectbox("Routing Domain Override", ["Financial Liability Cascade", "Voting Power Structure", "Tax Leakage Tracing"])
    
    st.header("3. API Gateway Configuration")
    api_key = st.text_input("OpenAI API Key Token", type="password")

col1, col2 = st.columns([2, 1])

with col1:
    st.write("### Network Topology View")
    st.info("System state: Awaiting graph generation payload. Upload target SEC filings to initialize.")
    
with col2:
    st.write("### Runtime State & Token Tracing")
    st.text_area("Extraction LLM Routing Matrix", value="[IDLE] Awaiting document ingestion...", height=120, disabled=True)
    st.text_area("NetworkX Math Traversal Path", value="[IDLE] Awaiting algorithmic execution call...", height=120, disabled=True)

st.write("---")
st.write("### Natural Language Reasoning Interface")
user_query = st.text_input(
    "Query the corporate architecture:", 
    placeholder="e.g., If Subsidiary C defaults, what is Parent A's total exposure under our current guarantee caps?"
)

if st.button("Execute Computational Routing"):
    if not uploaded_file:
        st.warning("Execution halted: Graph database state is empty. Please feed valid SEC documents into the ingestion layer.")
    else:
        st.spinner("Executing extraction pipelines and localized NetworkX calculations...")