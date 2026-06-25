#!/usr/bin/env python3
"""Verify fastembed BM25 memory footprint.

Measures peak RSS memory usage when importing fastembed, loading the
Qdrant/bm25 model, and running inference.  Designed to confirm that
the BM25 model (statistical IDF lookup, not a neural forward pass) is
lightweight enough for deployment on t2.micro (1 GB RAM).

Usage:
    python scripts/verify_fastembed_memory.py
"""

import os
import sys
from pathlib import Path

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def get_rss_mb() -> float:
    """Return current process RSS in MB (cross-platform)."""
    try:
        import psutil

        return psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        pass

    # Fallback: platform-specific
    if sys.platform == "linux":
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / 1024  # kB → MB
    elif sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class PROCESS_MEMORY_COUNTERS(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        counters = PROCESS_MEMORY_COUNTERS()
        counters.cb = ctypes.sizeof(PROCESS_MEMORY_COUNTERS)
        kernel32 = ctypes.windll.kernel32
        handle = kernel32.GetCurrentProcess()
        psapi = ctypes.windll.psapi
        psapi.GetProcessMemoryInfo(handle, ctypes.byref(counters), counters.cb)
        return counters.WorkingSetSize / (1024 * 1024)

    return -1.0  # Unknown


def main():
    print("=" * 60)
    print("  fastembed BM25 Memory Footprint Verification")
    print("=" * 60)

    rss_baseline = get_rss_mb()
    print(f"\n1. Baseline RSS (Python + stdlib): {rss_baseline:.1f} MB")

    # --- Import fastembed ---
    print("\n2. Importing fastembed...")
    from fastembed import SparseTextEmbedding  # noqa: E402

    rss_after_import = get_rss_mb()
    print(f"   RSS after import:              {rss_after_import:.1f} MB")
    print(f"   Delta (import):                +{rss_after_import - rss_baseline:.1f} MB")

    # --- Load model ---
    print("\n3. Loading Qdrant/bm25 model...")
    model = SparseTextEmbedding(model_name="Qdrant/bm25")
    rss_after_model = get_rss_mb()
    print(f"   RSS after model load:          {rss_after_model:.1f} MB")
    print(f"   Delta (model load):            +{rss_after_model - rss_after_import:.1f} MB")

    # --- Run inference ---
    print("\n4. Running sample inference...")
    sample_texts = [
        "IDOP is an enterprise-grade intelligent data operations platform.",
        "It combines NL-to-SQL, document mutations, and advanced RAG.",
        "Hybrid search uses dense and sparse vectors with RRF fusion.",
    ]
    results = list(model.embed(sample_texts))
    rss_after_inference = get_rss_mb()
    print(f"   RSS after inference:           {rss_after_inference:.1f} MB")
    print(f"   Delta (inference):             +{rss_after_inference - rss_after_model:.1f} MB")

    # --- Summary ---
    total_delta = rss_after_inference - rss_baseline
    print("\n" + "=" * 60)
    print(f"  TOTAL MEMORY DELTA:  +{total_delta:.1f} MB")
    print(f"  PEAK RSS:            {rss_after_inference:.1f} MB")
    print("=" * 60)

    if total_delta < 100:
        print("\n[PASS] fastembed BM25 is lightweight (< 100 MB delta)")
        print("   Safe for t2.micro (1 GB RAM) deployment.")
    elif total_delta < 200:
        print(f"\n[WARNING] fastembed BM25 uses {total_delta:.0f} MB")
        print("   Should be fine on t2.micro but monitor in production.")
    else:
        print(f"\n[CONCERN] fastembed BM25 uses {total_delta:.0f} MB")
        print("   May be tight on t2.micro (1 GB RAM). Investigate alternatives.")

    # Print sparse vector sample
    print("\n--- Sample sparse vector ---")
    sv = results[0]
    print(f"   Indices count: {len(sv.indices)}")
    print(f"   First 5 indices: {sv.indices[:5].tolist()}")
    print(f"   First 5 values:  {sv.values[:5].tolist()}")

    # Verify determinism
    print("\n--- Quick determinism check ---")
    results2 = list(model.embed(sample_texts))
    match = all(
        (r1.indices == r2.indices).all() and (r1.values == r2.values).all()
        for r1, r2 in zip(results, results2)
    )
    print(f"   Same model, same input -> same output: {'YES' if match else 'NO'}")

    return 0 if total_delta < 200 else 1


if __name__ == "__main__":
    sys.exit(main())
