from io import BytesIO

from pypdf import PdfReader
import tiktoken

MAX_PDFS = 10
CHUNK_TOKEN_SIZE = 4_000
CHUNK_TOKEN_OVERLAP = 500
TOKEN_ENCODING = "cl100k_base"


def _tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKEN_ENCODING)


def extract_text_from_pdf(file_bytes: bytes, filename: str) -> str:
    reader = PdfReader(BytesIO(file_bytes))
    body = "\n\n".join(page.extract_text() or "" for page in reader.pages).strip()
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
    if not corpus.strip():
        raise ValueError("No extractable text found in the uploaded PDFs.")

    return corpus, filenames


def segment_text_by_tokens(
    text: str,
    chunk_token_size: int = CHUNK_TOKEN_SIZE,
    chunk_token_overlap: int = CHUNK_TOKEN_OVERLAP,
) -> list[dict]:
    if chunk_token_size <= 0:
        raise ValueError("Chunk token size must be greater than zero.")
    if chunk_token_overlap < 0:
        raise ValueError("Chunk token overlap cannot be negative.")
    if chunk_token_overlap >= chunk_token_size:
        raise ValueError("Chunk token overlap must be smaller than chunk token size.")

    encoding = _tokenizer()
    tokens = encoding.encode(text)
    if not tokens:
        return []

    chunks = []
    start = 0
    chunk_index = 0
    step = chunk_token_size - chunk_token_overlap

    while start < len(tokens):
        end = min(start + chunk_token_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(
            {
                "chunk_index": chunk_index,
                "token_start": start,
                "token_end": end,
                "text": encoding.decode(chunk_tokens),
            }
        )
        if end == len(tokens):
            break
        start += step
        chunk_index += 1

    return chunks
