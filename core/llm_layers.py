import os

import instructor
from openai import OpenAI

from models import ExtractionResult, GraphPayload, SynthesisResponse

DOMAIN_FOCUS = {
    "Financial Liability Cascade": (
        "Focus on debt obligations, parent guarantees, subsidiary liabilities, "
        "guarantee caps, and exposure amounts."
    ),
    "Voting Power Structure": (
        "Focus on ownership percentages, voting rights, control relationships, "
        "and parent-subsidiary governance links."
    ),
    "Tax Leakage Tracing": (
        "Focus on intercompany flows, tax jurisdictions, transfer pricing entities, "
        "and structural tax exposure paths."
    ),
}

EXTRACTION_MODEL = "gpt-4o-mini"
SYNTHESIS_MODEL = "gpt-4o"


def resolve_api_key(override: str | None) -> str:
    api_key = (override or "").strip() or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise ValueError(
            "OpenAI API key is required. Set OPENAI_API_KEY or enter a key in the sidebar."
        )
    return api_key


def _instructor_client(api_key: str):
    return instructor.from_openai(OpenAI(api_key=api_key))


def extract_graph_from_documents(
    document_corpus: str,
    source_documents: list[str],
    query_topic: str,
    api_key: str,
) -> ExtractionResult:
    domain_focus = DOMAIN_FOCUS.get(query_topic, DOMAIN_FOCUS["Financial Liability Cascade"])
    client = _instructor_client(api_key)

    result = client.chat.completions.create(
        model=EXTRACTION_MODEL,
        response_model=ExtractionResult,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a corporate governance extraction engine. Parse SEC EDGAR-style "
                    "filings and credit agreements into a strict entity-relationship graph. "
                    "Use exact legal entity names when available. Normalize duplicate entities "
                    "across documents. Express percentages as decimals between 0.0 and 1.0. "
                    "Express dollar amounts as plain floats without currency symbols. "
                    "If a value is not stated, leave the field null rather than guessing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Routing domain: {query_topic}\n"
                    f"Domain focus: {domain_focus}\n"
                    f"Source documents: {', '.join(source_documents)}\n\n"
                    "Extract all corporate nodes and directed relationships from the documents below.\n\n"
                    f"{document_corpus}"
                ),
            },
        ],
    )
    return result.model_copy(update={"documents_processed": source_documents})


def synthesize_answer(
    user_query: str,
    query_topic: str,
    graph_payload: GraphPayload,
    source_documents: list[str],
    extraction_summary: str,
    api_key: str,
) -> SynthesisResponse:
    domain_focus = DOMAIN_FOCUS.get(query_topic, DOMAIN_FOCUS["Financial Liability Cascade"])
    client = _instructor_client(api_key)
    graph_json = graph_payload.model_dump_json(indent=2)

    return client.chat.completions.create(
        model=SYNTHESIS_MODEL,
        response_model=SynthesisResponse,
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a corporate governance analyst producing audit-ready responses. "
                    "Answer using only the extracted graph data provided. The deterministic graph "
                    "computation layer has NOT been run, so do not invent cascaded totals or "
                    "computed path metrics. Clearly state assumptions, missing data, and limitations. "
                    "When quantitative cascade math would normally be required, explain what can "
                    "and cannot be concluded from the extracted relationships alone."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"User query: {user_query}\n"
                    f"Routing domain: {query_topic}\n"
                    f"Domain focus: {domain_focus}\n"
                    f"Source documents: {', '.join(source_documents)}\n"
                    f"Extraction summary: {extraction_summary}\n\n"
                    f"Extracted graph payload:\n{graph_json}"
                ),
            },
        ],
    )
