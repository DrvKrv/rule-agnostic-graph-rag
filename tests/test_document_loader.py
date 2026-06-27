"""Document decoding / extraction tests (no LLM required)."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.document_loader import _decode_document, extract_text_from_document


def test_decode_prefers_cp1252_smart_quotes():
    # 0x93 / 0x94 are left/right double quotes in Windows-1252 (cp1252), but
    # C1 control characters in latin-1. SEC EDGAR filings are cp1252, so the
    # decoder must turn these into real curly quotes, not control chars.
    raw = b"He is the \x93Parent\x94 entity."
    decoded = _decode_document(raw)
    assert "\u201c" in decoded  # left double quotation mark
    assert "\u201d" in decoded  # right double quotation mark
    assert "\x93" not in decoded
    assert "\x94" not in decoded


def test_decode_utf8_still_wins_when_valid():
    raw = "Café Société".encode("utf-8")
    assert _decode_document(raw) == "Café Société"


def test_extract_html_preserves_curly_quotes():
    html = b"<html><body><p>The \x93Guarantor\x94 agrees.</p></body></html>"
    text = extract_text_from_document(html, "filing.htm")
    assert "\u201cGuarantor\u201d" in text
    assert "\x93" not in text
