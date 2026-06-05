#!/usr/bin/env python3
"""
Seed Benchmark Documents & Run Ablation Study
=============================================
Creates policy documents (.txt files) containing the ground truth data for the
50-question benchmark, uploads them to Qdrant via the FastAPI /documents/upload
endpoint, then runs the ablation study.

Usage:
    python scripts/seed_benchmark_docs.py          # Upload docs only
    python scripts/seed_benchmark_docs.py --run     # Upload docs + run ablation
    python scripts/seed_benchmark_docs.py --run --subset 10  # Subset for testing
"""

import argparse
import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.absolute()
sys.path.insert(0, str(PROJECT_ROOT))

DOCS_DIR = PROJECT_ROOT / "benchmark_docs"
DOCS_DIR.mkdir(exist_ok=True)

# ═══════════════════════════════════════════════════════════════════════════════
# Document Content — Written to mirror the benchmark ground truths
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
# Create document files
# ═══════════════════════════════════════════════════════════════════════════════

DOCUMENTS = [
    ("refund_policy.txt", REFUND_POLICY_DOC),
    ("employee_handbook.txt", EMPLOYEE_HANDBOOK_DOC),
    ("platform_operations.txt", PLATFORM_OPERATIONS_DOC),
]


def create_document_files():
    """Write the benchmark document content to .txt files."""
    created = []
    for filename, content in DOCUMENTS:
        path = DOCS_DIR / filename
        path.write_text(content.strip(), encoding="utf-8")
        size_kb = path.stat().st_size / 1024
        created.append((filename, size_kb))
        print(f"  [OK] Created {filename} ({size_kb:.1f} KB)")
    return created


# ═══════════════════════════════════════════════════════════════════════════════
# Upload documents via FastAPI
# ═══════════════════════════════════════════════════════════════════════════════


async def upload_document(api_base: str, filepath: Path) -> dict:
    """Upload a single document via the /documents/upload endpoint."""
    import httpx

    async with httpx.AsyncClient(timeout=120.0) as client:
        with open(filepath, "rb") as f:
            files = {"file": (filepath.name, f, "text/plain")}
            resp = await client.post(f"{api_base}/documents/upload", files=files)
            if resp.status_code == 200:
                data = resp.json()
                print(
                    f"  [OK] {filepath.name}: {data['chunks_created']} chunks, "
                    f"{'CACHE HIT' if data.get('cache_hit') else 'indexed fresh'}"
                )
                return data
            else:
                print(
                    f"  [FAIL] {filepath.name}: HTTP {resp.status_code} — {resp.text[:200]}"
                )
                return {"error": resp.text, "filename": filepath.name}


async def upload_all_documents(api_base: str):
    """Upload all benchmark documents."""
    print(f"\n{'='*60}")
    print("  Uploading benchmark documents to Qdrant...")
    print(f"  API: {api_base}")
    print(f"{'='*60}")

    results = []
    for filename, _ in DOCUMENTS:
        filepath = DOCS_DIR / filename
        result = await upload_document(api_base, filepath)
        results.append(result)

    # Print summary
    print(f"\n{'='*60}")
    total_chunks = sum(r.get("chunks_created", 0) for r in results if "error" not in r)
    successes = sum(1 for r in results if "error" not in r)
    failures = sum(1 for r in results if "error" in r)
    print(f"  Upload complete: {successes} succeeded, {failures} failed")
    print(f"  Total chunks indexed: {total_chunks}")
    print(f"{'='*60}")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Start FastAPI server
# ═══════════════════════════════════════════════════════════════════════════════


def start_server(port: int = 8088) -> subprocess.Popen:
    """Start the FastAPI server in the background."""
    print(f"\n  Starting FastAPI server on port {port}...")
    env = os.environ.copy()
    env["LOG_LEVEL"] = "WARNING"

    proc = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "uvicorn",
            "app.main:app",
            "--host",
            "127.0.0.1",
            f"--port={port}",
        ],
        cwd=str(PROJECT_ROOT),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )

    # Wait for server to be ready
    import httpx

    api_base = f"http://127.0.0.1:{port}"
    for i in range(30):
        try:
            resp = httpx.get(f"{api_base}/health", timeout=2.0)
            if resp.status_code == 200:
                print(f"  [OK] Server ready (attempt {i+1})")
                return proc
        except (httpx.ConnectError, httpx.TimeoutException):
            pass
        time.sleep(1)
    print("  [WARN] Server may not be ready — proceeding anyway")
    return proc


# ═══════════════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════════════


def parse_args(argv=None):
    parser = argparse.ArgumentParser(
        description="Create benchmark documents and upload them to Qdrant",
    )
    parser.add_argument(
        "--run", action="store_true", help="Also run the ablation study after uploading"
    )
    parser.add_argument(
        "--subset",
        "-n",
        type=int,
        default=None,
        help="Run ablation on a subset of N questions",
    )
    parser.add_argument(
        "--port", type=int, default=8088, help="Port for the FastAPI server"
    )
    parser.add_argument(
        "--no-ragas",
        action="store_true",
        help="Skip RAGAS library, use in-house evaluator",
    )
    return parser.parse_args(argv)


def main():
    args = parse_args()
    port = args.port
    api_base = f"http://127.0.0.1:{port}"

    # Step 1: Create document files
    print("\n  Creating benchmark document files...")
    create_document_files()

    # Step 2: Start server
    server_proc = start_server(port)

    try:
        # Step 3: Upload documents
        asyncio.run(upload_all_documents(api_base))

        # Step 4: Optionally run ablation study
        if args.run:
            print(f"\n{'='*60}")
            print("  Running ablation study...")
            print(f"{'='*60}")

            cmd = [sys.executable, "-m", "scripts.eval_ragas", "--no-ragas"]
            if args.subset:
                cmd.extend(["--subset", str(args.subset)])

            subprocess.run(cmd, cwd=str(PROJECT_ROOT))
    finally:
        # Cleanup: stop server
        print("\n  Shutting down server...")
        server_proc.terminate()
        try:
            server_proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server_proc.kill()

    print("\n  Done!")


if __name__ == "__main__":
    main()
