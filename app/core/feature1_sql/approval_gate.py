"""
Feature 1 (SQL) approval gate — re-exports the shared ApprovalGate singleton.

Kept as a thin re-export for backward compatibility so existing imports
(``from app.core.feature1_sql.approval_gate import approval_gate``) continue to work.
"""

from app.core.approval_gate import approval_gate  # noqa: F401
