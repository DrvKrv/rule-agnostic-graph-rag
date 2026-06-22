import os

import instructor
from openai import OpenAI

from core.document_loader import segment_text_by_tokens
from models import ExtractionResult, ExtractionSegment, GraphPayload, SynthesisResponse

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

EXTRACTION_MODEL = "gpt-5.5"
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


def _responses_client(api_key: str) -> OpenAI:
    return OpenAI(api_key=api_key)


def _extract_chunk_payload(
    client: OpenAI,
    chunk_text: str,
    source_documents: list[str],
    query_topic: str,
    domain_focus: str,
) -> GraphPayload:
    response = client.responses.parse(
        model=EXTRACTION_MODEL,
        input=[
            {
                "role": "system",
                "content": (
                    "You are a stateless corporate governance relationship extraction engine. "
                    "Read only the provided raw text/markdown chunk. Return clean JSON that "
                    "strictly matches the GraphPayload schema, whose nodes must match "
                    "CorporateNode and whose edges must match CorporateEdge. Do not infer "
                    "relationships from outside this chunk. Do not perform graph traversal, "
                    "deduplication, cascade math, ownership math, or NetworkX-style processing. "
                    "Use exact legal entity names when available. Express percentages as "
                    "decimals between 0.0 and 1.0 and dollar amounts as plain floats. Leave "
                    "unknown fields null rather than guessing."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"Routing domain: {query_topic}\n"
                    f"Domain focus: {domain_focus}\n"
                    f"Source documents in corpus: {', '.join(source_documents)}\n\n"
                    "Extract only corporate nodes and directed corporate relationships that "
                    "are explicitly supported by this raw text/markdown chunk.\n\n"
                    f"{chunk_text}"
                ),
            },
        ],
        text_format=GraphPayload,
    )
    if response.output_parsed is None:
        raise ValueError("Layer 1 extraction returned no structured payload for a chunk.")
    return response.output_parsed


def _flatten_segment_payloads(segments: list[ExtractionSegment]) -> GraphPayload:
    nodes = []
    edges = []
    for segment in segments:
        nodes.extend(segment.payload.nodes)
        edges.extend(segment.payload.edges)
    return GraphPayload(nodes=nodes, edges=edges)


def extract_graph_from_documents(
    document_corpus: str,
    source_documents: list[str],
    query_topic: str,
    api_key: str,
) -> ExtractionResult:
    domain_focus = DOMAIN_FOCUS.get(query_topic, DOMAIN_FOCUS["Financial Liability Cascade"])
    client = _responses_client(api_key)
    chunks = segment_text_by_tokens(document_corpus)
    if not chunks:
        raise ValueError("No extractable text chunks were produced from the uploaded PDFs.")

    segments = []
    for chunk in chunks:
        payload = _extract_chunk_payload(
            client=client,
            chunk_text=chunk["text"],
            source_documents=source_documents,
            query_topic=query_topic,
            domain_focus=domain_focus,
        )
        segments.append(
            ExtractionSegment(
                chunk_index=chunk["chunk_index"],
                source_documents=source_documents,
                token_start=chunk["token_start"],
                token_end=chunk["token_end"],
                payload=payload,
            )
        )

    graph = _flatten_segment_payloads(segments)
    extraction_summary = (
        f"Layer 1 produced {len(segments)} stateless extraction segment(s) from "
        f"{len(source_documents)} PDF(s) using {EXTRACTION_MODEL}. Segment payloads are "
        "literal chunk-level JSON extractions; Layer 2 graph processing was not run."
    )
    return ExtractionResult(
        graph=graph,
        extraction_summary=extraction_summary,
        documents_processed=source_documents,
        segments=segments,
    )


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
