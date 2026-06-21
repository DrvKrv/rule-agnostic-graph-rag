from io import BytesIO

from pypdf import PdfReader

MAX_PDFS = 10
MAX_TOTAL_CHARS = 120_000


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    pages = []
    for page in reader.pages:
        pages.append(page.extract_text() or "")
    body = "\n".join(pages).strip()
    return f"--- Document: {filename} ---\n{body}"


def build_document_corpus(files: list[tuple[str, bytes]]) -> tuple[str, list[str]]:
    if not files:
        raise ValueError("At least one PDF is required.")
    if len(files) > MAX_PDFS:
        raise ValueError(f"A maximum of {MAX_PDFS} PDFs is allowed.")

    sections: list[str] = []
    filenames: list[str] = []
    for filename, file_bytes in files:
        filenames.append(filename)
        sections.append(extract_text_from_pdf(file_bytes, filename))

    corpus = "\n\n".join(sections)
    if len(corpus) > MAX_TOTAL_CHARS:
        per_doc_budget = MAX_TOTAL_CHARS // len(sections)
        truncated_sections = []
        for section in sections:
            if len(section) <= per_doc_budget:
                truncated_sections.append(section)
            else:
                truncated_sections.append(
                    section[:per_doc_budget]
                    + f"\n\n[Truncated: document exceeded {per_doc_budget:,} character budget]"
                )
        corpus = "\n\n".join(truncated_sections)

    if not corpus.strip():
        raise ValueError("No extractable text found in the uploaded PDFs.")

    return corpus, filenames
