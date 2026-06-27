from pathlib import Path

from bs4 import BeautifulSoup
import tiktoken

MAX_UPLOAD_FILES = 10
ACCEPTED_EXTENSIONS = {".htm", ".txt"}
CHUNK_TOKEN_SIZE = 4_000
CHUNK_TOKEN_OVERLAP = 500
TOKEN_ENCODING = "cl100k_base"


def _tokenizer() -> tiktoken.Encoding:
    return tiktoken.get_encoding(TOKEN_ENCODING)


def _decode_document(file_bytes: bytes) -> str:
    for encoding in ("utf-8", "latin-1"):
        try:
            return file_bytes.decode(encoding)
        except UnicodeDecodeError:
            continue
    return file_bytes.decode("utf-8", errors="replace")


def _normalize_lines(text: str) -> str:
    lines = (line.strip() for line in text.splitlines())
    return "\n".join(line for line in lines if line)


def _extract_text_from_html(raw_text: str) -> str:
    soup = BeautifulSoup(raw_text, "html.parser")
    for element in soup(["script", "style", "noscript", "svg"]):
        element.decompose()
    return _normalize_lines(soup.get_text(separator="\n"))


def _looks_like_markup(raw_text: str) -> bool:
    """Heuristic: detect SEC complete-submission .txt files that embed HTML/SGML/XBRL."""
    sample = raw_text[:10_000].lower()
    markers = ("<sec-document", "<html", "<table", "<xbrl", "<document", "<type>")
    if any(marker in sample for marker in markers):
        return True
    # Fall back to tag density across the document.
    return raw_text.count("<") > 50


def _extract_text_from_txt(raw_text: str) -> str:
    # SEC "complete submission" .txt files are concatenated SGML + HTML + XBRL, not
    # plain text. Strip the markup so the extractor sees readable filing prose.
    if _looks_like_markup(raw_text):
        return _extract_text_from_html(raw_text)
    return _normalize_lines(raw_text)


def extract_text_from_document(file_bytes: bytes, filename: str) -> str:
    extension = Path(filename).suffix.lower()
    if extension not in ACCEPTED_EXTENSIONS:
        accepted = ", ".join(sorted(ACCEPTED_EXTENSIONS))
        raise ValueError(f"Unsupported file type for {filename}. Accepted types: {accepted}.")

    raw_text = _decode_document(file_bytes)
    if extension == ".htm":
        body = _extract_text_from_html(raw_text)
    else:
        body = _extract_text_from_txt(raw_text)

    return f"--- Document: {filename} ---\n{body}"


def build_document_corpus(files: list[tuple[str, bytes]]) -> tuple[str, list[str]]:
    if not files:
        raise ValueError("At least one .htm or .txt file is required.")
    if len(files) > MAX_UPLOAD_FILES:
        raise ValueError(f"A maximum of {MAX_UPLOAD_FILES} files is allowed.")

    sections: list[str] = []
    filenames: list[str] = []
    for filename, file_bytes in files:
        filenames.append(filename)
        sections.append(extract_text_from_document(file_bytes, filename))

    corpus = "\n\n".join(sections)
    if not corpus.strip():
        raise ValueError("No extractable text found in the uploaded SEC filing files.")

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
