#!/usr/bin/env python3
"""
Direct Upload Benchmark Documents
==================================
Uploads benchmark policy documents directly to Qdrant using internal Python APIs,
bypassing the need for a running FastAPI server.

Usage:
    python scripts/direct_upload_benchmark_docs.py
    python scripts/direct_upload_benchmark_docs.py --run-ablation
    python scripts/direct_upload_benchmark_docs.py --run-ablation --subset 5
"""

import argparse
import asyncio
import sys
from pathlib import Path

from langchain_core.documents import Document

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = PROJECT_ROOT / "benchmark_docs"


# ═══════════════════════════════════════════════════════════════════════════════
# Document Content
# ═══════════════════════════════════════════════════════════════════════════════

# Document content is loaded from the benchmark_docs/ directory to ensure
# consistency between the upload script and what's indexed in Qdrant.
# If you update benchmark_docs/*.txt, these will pick up the changes.


_DOCS_DIR = Path(__file__).parent.parent / "benchmark_docs"


def _load_doc(filename: str) -> str:
    path = _DOCS_DIR / filename
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


REFUND_POLICY_DOC = _load_doc("refund_policy.txt")
EMPLOYEE_HANDBOOK_DOC = _load_doc("employee_handbook.txt")
PLATFORM_OPERATIONS_DOC = _load_doc("platform_operations.txt")
REFUND_POLICY_2025_DOC = _load_doc("refund_policy_2025_superseded.txt")
EMPLOYEE_HANDBOOK_2025_DOC = _load_doc("employee_handbook_2025_superseded.txt")
REGIONAL_POLICY_DOC = _load_doc("regional_policy.txt")
INTERNAL_MEMOS_DOC = _load_doc("internal_memos.txt")


async def upload_document_direct(
    vector_store,
    filename: str,
    content: str,
) -> dict:
    """Process and upload a document string directly to Qdrant."""
    from app.core.document_processor import DocumentProcessor

    processor = DocumentProcessor()

    # Create a Document from the raw text
    doc = Document(
        page_content=content,
        metadata={"source": filename, "doc_type": "benchmark"},
    )

    # Split into chunks
    chunks = processor.split_documents([doc])
    print(f"  Chunked {filename}: {len(chunks)} chunks")

    # Embed
    texts = [c.page_content for c in chunks]
    embeddings = vector_store.embeddings.embed_documents(texts)
    print(f"  Embedded {filename}: {len(embeddings)} vectors")

    # Index in Qdrant (with SHA-256 dedup)
    doc_ids = vector_store.add_documents_with_embeddings(chunks, embeddings)
    print(f"  Indexed {filename}: {len(doc_ids)} IDs (dedup applied)")

    return {
        "filename": filename,
        "chunks_created": len(chunks),
        "document_ids": doc_ids,
    }


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


async def main_async():
    parser = argparse.ArgumentParser(
        description="Upload benchmark documents directly to Qdrant (no server needed)"
    )
    parser.add_argument(
        "--run-ablation",
        action="store_true",
        help="Also run the ablation study after uploading",
    )
    parser.add_argument(
        "--subset", "-n", type=int, default=None, help="Run ablation on a subset"
    )
    args = parser.parse_args()

    # Set up event loop for Windows
    import asyncio

    if sys.platform == "win32":
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

    print("=" * 60)
    print("  Direct Upload: Benchmark Documents -> Qdrant")
    print("=" * 60)

    # Initialize vector store
    from app.core.vector_store import VectorStoreService

    print("\n  [1/3] Initializing VectorStoreService (Qdrant + OpenAI)...")
    vector_store = VectorStoreService()
    print("  [OK] VectorStoreService ready")

    # Check current collection state
    info = vector_store.get_collection_info()
    print(
        f"  Current Qdrant collection: '{info['name']}' — {info['points_count']} points"
    )

    # Upload documents
    print("\n  [2/3] Uploading benchmark documents...\n")

    documents = [
        ("refund_policy.txt", REFUND_POLICY_DOC),
        ("employee_handbook.txt", EMPLOYEE_HANDBOOK_DOC),
        ("platform_operations.txt", PLATFORM_OPERATIONS_DOC),
    ]

    results = []
    for filename, content in documents:
        print(f"  ── {filename} ──")
        try:
            result = await upload_document_direct(vector_store, filename, content)
            results.append(result)
            print(f"  ✓ {filename}: {result['chunks_created']} chunks indexed\n")
        except Exception as e:
            print(f"  ✗ {filename}: ERROR — {type(e).__name__}: {e}\n")

    # Summary
    print("=" * 60)
    total_chunks = sum(r.get("chunks_created", 0) for r in results)
    successes = sum(1 for r in results if "chunks_created" in r)
    print(f"  Upload complete: {successes}/{len(documents)} documents indexed")
    print(f"  Total chunks added: {total_chunks}")

    # Show final Qdrant state
    info = vector_store.get_collection_info()
    print(f"  Qdrant collection now has: {info['points_count']} points")
    print("=" * 60)

    # Optionally run ablation study
    if args.run_ablation:
        print(f"\n{'=' * 60}")
        print("  Running ablation study...")
        print(f"{'=' * 60}\n")

        from scripts.eval_ragas import main as run_ablation

        # Override sys.argv for the ablation script
        import sys as _sys

        _sys.argv = ["eval_ragas.py", "--no-ragas"]
        if args.subset:
            _sys.argv.extend(["--subset", str(args.subset)])

        run_ablation()

    print("\n  Done!")


def main():
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
