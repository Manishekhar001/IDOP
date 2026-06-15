#!/usr/bin/env python3
"""
Create a minimal valid PDF with extractable text for testing PDF upload.
Generates a proper PDF structure with xref table for pypdf to parse.
"""

import os
import sys


def create_pdf(filepath: str, text: str | None = None):
    """Create a PDF with substantial multi-paragraph text."""
    if text is None:
        text = (
            "IDOP Intelligent Data Operations Platform. "
            "This document provides a comprehensive overview of the IDOP platform capabilities. "
            "The platform enables users to query databases using natural language, "
            "perform data mutations with approval workflows, and leverage RAG-powered "
            "document understanding.\n\n"
            "Key Features and Capabilities. "
            "IDOP supports multiple feature pipelines including SQL query generation, "
            "database mutation execution, and RAG-based document retrieval. "
            "Each feature pipeline includes dedicated sub-components for validation, "
            "approval, and execution. The platform uses LangGraph state machines "
            "for orchestrating multi-step agent workflows.\n\n"
            "System Architecture Overview. "
            "The system is built on a FastAPI backend with Qdrant vector store "
            "for document embeddings and PostgreSQL for structured data and LangGraph checkpoints. "
            "Documents are processed using pypdf for lightweight text extraction, "
            "then chunked and embedded for semantic search.\n\n"
            "Security and Approval Mechanisms. "
            "All SQL queries and database mutations go through an approval gate "
            "that generates cryptographic tokens for secure execution. "
            "Business rules are enforced through configurable rule validators "
            "that check field types, ranges, and allowed values before any data modification. "
            "The LLM judge provides a final audit layer for mutation approval."
        )

    pdf_parts = []
    pdf_parts.append(b"%PDF-1.4")

    # Object 1: Catalog
    pdf_parts.append(b"1 0 obj")
    pdf_parts.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    pdf_parts.append(b"endobj")

    # Object 2: Pages
    pdf_parts.append(b"2 0 obj")
    pdf_parts.append(b"<< /Type /Pages /Kids [3 0 R] /Count 1 >>")
    pdf_parts.append(b"endobj")

    # Object 3: Page with multiple text operations
    text_lines = text.replace("\n\n", "\n").split("\n")
    stream_lines = [b"BT"]
    y_pos = 720
    for i, line in enumerate(text_lines):
        line = line.strip()
        if not line:
            continue
        y_pos = 720 - (i * 30)
        if y_pos < 50:
            break
        stream_lines.append(b"/F1 14 Tf")
        stream_lines.append(b"50 " + str(y_pos).encode() + b" Td")
        # Escape special characters in PDF string
        safe = line.encode("latin-1", errors="replace")
        safe = safe.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)")
        stream_lines.append(b"(" + safe + b") Tj")
    stream_lines.append(b"ET")

    stream_data = b"\n".join(stream_lines)
    stream_length = len(stream_data)

    pdf_parts.append(b"3 0 obj")
    pdf_parts.append(b"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792]")
    pdf_parts.append(b"   /Contents 4 0 R /Resources << /Font << /F1 5 0 R >> >> >>")
    pdf_parts.append(b"endobj")

    # Object 4: Content stream
    pdf_parts.append(b"4 0 obj")
    pdf_parts.append(b"<< /Length " + str(stream_length).encode() + b" >>")
    pdf_parts.append(b"stream")
    pdf_parts.append(stream_data)
    pdf_parts.append(b"endstream")
    pdf_parts.append(b"endobj")

    # Object 5: Font
    pdf_parts.append(b"5 0 obj")
    pdf_parts.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    pdf_parts.append(b"endobj")

    # Cross-reference table
    pdf_parts.append(b"xref")
    pdf_parts.append(b"0 6")
    pdf_parts.append(b"0000000000 65535 f ")

    offset = 0
    offsets = [0]
    for part in pdf_parts:
        offset += len(part) + 1
        offsets.append(offset)

    for i in range(1, 6):
        entry = str(offsets[i]).zfill(10) + " 00000 n "
        pdf_parts.append(entry.encode())

    xref_offset = offset
    pdf_parts.append(b"trailer")
    pdf_parts.append(b"<< /Size 6 /Root 1 0 R >>")
    pdf_parts.append(b"startxref")
    pdf_parts.append(str(xref_offset).encode())
    pdf_parts.append(b"%%EOF")

    with open(filepath, "wb") as f:
        for part in pdf_parts:
            f.write(part + b"\n")

    file_size = os.path.getsize(filepath)
    print(f"PDF created: {filepath} ({file_size} bytes, {len(text)} chars)")
    return filepath


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "/tmp/test_pypdf.pdf"
    create_pdf(out)
