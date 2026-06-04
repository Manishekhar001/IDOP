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
import hashlib
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = PROJECT_ROOT / "benchmark_docs"


# ═══════════════════════════════════════════════════════════════════════════════
# Document Content
# ═══════════════════════════════════════════════════════════════════════════════

REFUND_POLICY_DOC = """IDOP COMPANY REFUND AND CUSTOMER SUPPORT POLICY
===============================================

Last Updated: January 2026
Effective Date: February 1, 2026

1. STANDARD REFUND WINDOW
Retail products are eligible for a full refund within 30 days of purchase with receipt.
All refund requests must include the original receipt or proof of purchase.

2. INTERNATIONAL RETURN SHIPPING
For international orders, customers must cover return shipping costs unless the product arrived damaged.
Damaged product claims must be supported with photographic evidence submitted within 48 hours of delivery.

3. CLEARANCE AND FINAL SALE ITEMS
All clearance items are marked as final sale and are strictly non-refundable.
This policy applies to all clearance merchandise regardless of condition or purchase date.

4. DIGITAL CREDIT PROCESSING
Approved digital credits are processed within 24 to 48 business hours of verification.
Digital credits are issued as store credit and must be used within 90 days.

5. BANK TRANSFER REFUNDS
Direct bank transfer refunds typically take 5 to 7 business days to post to the account.
International bank transfers may take up to 10-14 business days depending on the destination country.

6. GIFT CARDS
Gift cards cannot be returned, exchanged, or redeemed for cash refunds under any circumstances.
Gift card balances are non-transferable and expire 24 months from the date of issuance.

7. RESTOCKING FEES ON OPENED ELECTRONICS
Opened electronics are subject to a 15% restocking fee if returned within the 30-day window.
Electronics must be returned with all original accessories, packaging, and manuals to qualify for any refund.

8. LOST IN TRANSIT CLAIMS
Lost in transit claims must be filed within 14 days of the expected delivery date.
Upon verification, the company will issue a replacement or full refund.
Claims filed after 14 days may be denied at the carrier's discretion.

9. SHIPPING FEE REFUNDS
Original shipping and handling fees are non-refundable for voluntary customer returns.
Shipping fees are only refundable if the return is due to a company error (wrong item shipped, damaged product).

10. PROMOTIONAL BUNDLE RETURNS
Promotional bundle items must be returned together to receive a full refund; partial returns are rejected.
If a customer returns only part of a promotional bundle, the entire bundle refund is forfeited.
"""

EMPLOYEE_HANDBOOK_DOC = """IDOP EMPLOYEE HANDBOOK — CORPORATE POLICIES
===============================================

Last Updated: January 2026
Effective Date: February 1, 2026

1. OFFICE CORE HOURS
Core collaboration hours are from 10:00 AM to 3:00 PM EST daily.
All employees are expected to be available during core hours regardless of their flexible schedule.

2. HEALTH WELLNESS STIPEND
Wellness stipends of up to $500 are claimed by submitting receipts through the HR Expense portal before December 1st.
Eligible expenses include gym memberships, fitness classes, mental health services, and wellness app subscriptions.
Receipts must be dated within the current calendar year to be eligible.

3. EMPLOYEE REFERRAL BONUS
The standard referral bonus is $2,000, paid in two installments after the new hire completes 90 and 180 days.
First installment of $1,000 is paid at 90 days. Second installment of $1,000 is paid at 180 days.
Referral bonuses apply only to full-time hires; contract worker referrals are ineligible for standard HR referral payouts.

4. TRAVEL FLIGHT BOOKING POLICY
All business flights under 6 hours must be booked in economy class. Business class requires VP approval.
Flights over 6 hours do not have class restrictions, but standard flights under 6 hours require economy class unless approved by VP.

5. SICK LEAVE POLICY
Consecutive sick leaves of 3 or more days require a valid medical certificate submitted to HR.
Short-term sick leave (1-2 days) can be self-certified without documentation.

6. EQUIPMENT RETURN ON DEPARTURE
Departing employees must return all company-owned hardware within 5 business days of their final day.
Equipment includes laptops, monitors, keyboards, phones, and any other company-issued devices.
Failure to return equipment may result in deduction from final paycheck.

7. CORPORATE GIFT POLICY
Employees cannot accept corporate gifts exceeding a nominal value of $100 without compliance sign-off.
Gifts from vendors or partners exceeding $100 must be reported to the compliance department within 5 business days.

8. TUITION REIMBURSEMENT
The company reimburses up to $5,250 annually for pre-approved, job-related graduate coursework with grade B or higher.
Courses must be pre-approved by the employee's manager and HR before enrollment.
Reimbursement is contingent upon successful completion with a grade of B or above.

9. PARENTAL LEAVE
The platform provides 12 weeks of fully paid parental leave for primary and secondary caregivers after 1 year of service.
Parental leave must be taken within 12 months of the birth or adoption of a child.

10. CONTRACT WORKER BENEFITS
Contract workers are generally ineligible for corporate group dental benefits unless explicitly detailed in their agreement.
Contract workers receive only the benefits specified in their independent contractor agreement.
"""

PLATFORM_OPERATIONS_DOC = """IDOP PLATFORM OPERATIONS GUIDE
===============================================

Last Updated: January 2026

1. MUTATION ROW LIMITS
The platform enforces a strict threshold limit of 1000 rows per spreadsheet mutation upload to protect database memory.
Uploads exceeding 1000 rows will be rejected with an error message.

2. DEPARTMENT ENUMS
Allowed corporate department enums are 'HR', 'Engineering', 'Sales', and 'Finance'.
Any department values outside these four will be rejected by the mutation validator.

3. BUSINESS RULES CONFIGURATION
Validation constraints are declared in the local 'business_rules/rules.json' configuration file.
This file contains all business rules for salary limits, department mappings, and data validation.

4. SALARY CAPS
According to rules.json, the maximum permitted salary for junior tiers is capped at $120,000.
Senior and executive roles have separate salary bands defined in the rules.json file.

5. MUTATION APPROVAL PROCESS
All validated mutations require human-in-the-loop validation using a secure token at the '/mutation/approve' route.
The approval gate generates a cryptographically signed single-use token for each pending mutation.

6. EMPTY CELL HANDLING
Empty or null cells are rejected with validation errors if the rule declares the column as required/non-null.
Optional columns with empty cells are allowed and will be stored as NULL in the database.

7. CHECKPOINT INTERVAL
The LangGraph state checkpointer commits checkpoints instantly upon transition of every active processing node.
This ensures zero data loss even if the service experiences an unexpected restart.

8. POSTGRES CHECKPOINTER
IDOP uses an internal PostgreSQL Docker container for checkpoint states and user facts stores.
The container runs postgres:16 with persistent volume storage.

9. VOYAGE AI RERANKING
Semantic query reranking is offloaded to the Voyage AI Rerank-2.5 Cross-Encoder API.
Voyage AI reranking provides cross-encoder precision for improved context relevance.

10. VECTOR DIMENSIONS
Qdrant utilizes 1536-dimensional dense vectors generated by OpenAI text-embedding-3-small.
The sparse vectors use BM25 keyword tokenization for complementary exact-match retrieval.
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Document processor helpers (mirrors app.core.document_processor logic)
# ═══════════════════════════════════════════════════════════════════════════════

from langchain_core.documents import Document


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
    parser.add_argument("--run-ablation", action="store_true", help="Also run the ablation study after uploading")
    parser.add_argument("--subset", "-n", type=int, default=None, help="Run ablation on a subset")
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
    from app.core.embeddings import EmbeddingsService

    print("\n  [1/3] Initializing VectorStoreService (Qdrant + OpenAI)...")
    vector_store = VectorStoreService()
    print("  [OK] VectorStoreService ready")

    # Check current collection state
    info = vector_store.get_collection_info()
    print(f"  Current Qdrant collection: '{info['name']}' — {info['points_count']} points")

    # Upload documents
    print(f"\n  [2/3] Uploading benchmark documents...\n")

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
