"""
Feature 2 (mutation) approval gate — re-exports the shared ApprovalGate singleton.

Kept as a thin re-export for backward compatibility so existing imports
(``from app.core.feature2_mutation.approval_gate import mutation_approval_gate``)
continue to work.
"""

from app.core.approval_gate import mutation_approval_gate  # noqa: F401
