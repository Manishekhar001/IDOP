"""
File Reference Validator.

Scans the entire codebase and validates that every file reference
(Python imports, hardcoded file paths, documentation links) resolves
to an existing file on disk.

Exit codes:
    0 - All references valid
    1 - One or more broken references found

Usage:
    python scripts/validate_file_references.py
    python scripts/validate_file_references.py --verbose
"""

import ast
import io
import os
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

# --- Paths -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# File extensions to scan
PYTHON_EXTS = {".py"}
DOC_EXTS = {".md", ".html", ".rst"}
# Directories to skip entirely
SKIP_DIRS = {
    ".git",
    "__pycache__",
    ".venv",
    "venv",
    "node_modules",
    ".eggs",
    ".mypy_cache",
    ".pytest_cache",
}

# Project-internal top-level packages - only validate imports from these
PROJECT_PACKAGES = {"app", "tests", "scripts", "docs"}

# Pattern to find file-path strings in Python source
FILE_PATH_PATTERN = re.compile(
    r"""["']                                 # opening quote
        ((?:app|tests|scripts|docs|business_rules|\.github)
        /[\w\-./]+\.(?:py|md|yml|yaml|json|cfg|ini|txt|html))  # extension
        ["']                                 # closing quote
    """,
    re.VERBOSE,
)


def is_python_file(path: Path) -> bool:
    """Check if a path is a Python file (excluding __init__.py)."""
    return path.suffix in PYTHON_EXTS and path.name != "__init__.py"


def get_all_project_files() -> Set[Path]:
    """Return a set of all tracked project file paths (relative to PROJECT_ROOT)."""
    files: Set[Path] = set()
    for root, dirs, filenames in os.walk(PROJECT_ROOT):
        # Skip ignored/virtual directories
        dirs[:] = [d for d in dirs if d not in SKIP_DIRS and not d.startswith(".")]
        for fn in filenames:
            full = Path(root) / fn
            rel = full.relative_to(PROJECT_ROOT)
            files.add(rel)
    return files


def _module_exists(module: str, all_files: Set[Path]) -> bool:
    """Check if a dotted module path corresponds to an existing file."""
    parts = module.split(".")
    # Candidate: module.py
    if len(parts) >= 1:
        p = Path(*parts[:-1], f"{parts[-1]}.py")
        if p in all_files:
            return True
    # Candidate: module/__init__.py
    p2 = Path(*parts, "__init__.py")
    if p2 in all_files:
        return True
    return False


def check_python_imports(
    file_paths: List[Path], all_files: Set[Path], verbose: bool
) -> List[str]:
    """Parse every Python file and validate its import statements.

    Only validates imports from project-internal packages (app, tests, scripts, docs).
    Third-party and stdlib imports are skipped.
    """
    errors: List[str] = []

    for file_rel in file_paths:
        if not is_python_file(file_rel) and file_rel.name != "__init__.py":
            continue

        file_abs = PROJECT_ROOT / file_rel
        try:
            with open(file_abs, encoding="utf-8") as f:
                source = f.read()
        except Exception as e:
            errors.append(f"  Cannot read {file_rel}: {e}")
            continue

        try:
            tree = ast.parse(source, filename=str(file_abs))
        except SyntaxError as e:
            errors.append(f"  Syntax error in {file_rel}: {e}")
            continue

        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    mod = alias.name
                    top = mod.split(".")[0]
                    # Only validate project-internal imports
                    if top not in PROJECT_PACKAGES:
                        continue
                    if not _module_exists(mod, all_files):
                        errors.append(
                            f"  {file_rel}: import {mod} - module not found"
                        )

            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    name = alias.name

                    if node.level:
                        # Relative import - resolve relative to the source file's package
                        pkg_parts = list(file_rel.parts[:-1])  # Remove filename
                        # Walk up `node.level` levels (1 = same dir, 2 = parent, etc.)
                        up = node.level - 1
                        if up > 0 and len(pkg_parts) >= up:
                            pkg_parts = pkg_parts[:-up]
                        elif up > 0:
                            continue  # Can't resolve beyond project root

                        if module:
                            full_mod = ".".join(pkg_parts + module.split("."))
                        else:
                            full_mod = ".".join(pkg_parts)

                        # Also check with the imported name as a submodule
                        if not _module_exists(full_mod, all_files) and not _module_exists(
                            full_mod + "." + name, all_files
                        ):
                            rel_str = "." * node.level + module
                            errors.append(
                                f"  {file_rel}: from {rel_str} import {name} - module not found"
                            )
                    else:
                        # Absolute import - only validate project-internal modules
                        if not module:
                            continue
                        top = module.split(".")[0]
                        if top not in PROJECT_PACKAGES:
                            continue
                        if not _module_exists(module, all_files) and not _module_exists(
                            module + "." + name, all_files
                        ):
                            errors.append(
                                f"  {file_rel}: from {module} import {name} - module not found"
                            )

    return errors


def check_hardcoded_paths(
    file_paths: List[Path], all_files: Set[Path], verbose: bool
) -> List[str]:
    """Find hardcoded file path strings in Python files and check they exist.

    Catches patterns like:
      - File lists in test files (e.g. ROUTE_FILES = [..., "app/api/routes/chat.py"])
      - Config references to other files
      - Any string literal matching a project file path
    """
    errors: List[str] = []

    for file_rel in file_paths:
        if file_rel.suffix not in PYTHON_EXTS:
            continue

        file_abs = PROJECT_ROOT / file_rel
        try:
            with open(file_abs, encoding="utf-8") as f:
                source = f.read()
        except Exception:
            continue

        for match in FILE_PATH_PATTERN.finditer(source):
            path_str = match.group(1)
            p = Path(path_str.replace("/", os.sep).replace("\\", os.sep))
            if not p.is_absolute() and p not in all_files:
                errors.append(
                    f"  {file_rel}: string path \"{path_str}\" - file not found"
                )

    return errors


def check_doc_references(
    doc_files: List[Path], all_files: Set[Path], verbose: bool
) -> List[str]:
    """Scan documentation files for broken file references."""
    errors: List[str] = []

    # Pattern to find file references in markdown/docs
    doc_pattern = re.compile(
        r"""(?:`|\()                    # backtick or opening paren
            ((?:
                app/|tests/|scripts/|docs/|\.github/
            )
            [\w\-./]+\.\w+)
            (?:`|\))                   # closing backtick or paren
        """,
        re.VERBOSE,
    )

    for file_rel in doc_files:
        file_abs = PROJECT_ROOT / file_rel
        try:
            with open(file_abs, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        seen: Set[str] = set()
        for match in doc_pattern.finditer(content):
            path_str = match.group(1).rstrip(")`")
            if path_str in seen:
                continue
            seen.add(path_str)

            p = Path(path_str.replace("/", os.sep))
            if p not in all_files:
                errors.append(f"  {file_rel}: references \"{path_str}\" - file not found")

    return errors


def check_workflow_references(
    workflow_files: List[Path], all_files: Set[Path], verbose: bool
) -> List[str]:
    """Check file references in GitHub Actions workflow files."""
    errors: List[str] = []

    # Pattern for scripts/ references in YAML
    script_re = re.compile(r"""(?:python\s+)?scripts/[\w\-./]+\.py""")

    for file_rel in workflow_files:
        file_abs = PROJECT_ROOT / file_rel
        try:
            with open(file_abs, encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue

        seen: Set[str] = set()
        for match in script_re.finditer(content):
            path_str = match.group(0).replace("python ", "")
            if path_str in seen:
                continue
            seen.add(path_str)

            p = Path(path_str.replace("/", os.sep))
            if p not in all_files:
                errors.append(
                    f"  {file_rel}: references \"{path_str}\" - script not found"
                )

    return errors


def main():
    verbose = "--verbose" in sys.argv or "-v" in sys.argv

    # Configure stdout for UTF-8 on Windows
    if sys.platform == "win32":
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

    print("=" * 60)
    print("  File Reference Validator")
    print("=" * 60)
    print()

    # --- Gather all files ----------------------------------------------------
    print("Scanning project files...", end=" ")
    sys.stdout.flush()
    all_files = get_all_project_files()
    print(f"found {len(all_files)} files.")
    if verbose:
        for f in sorted(all_files):
            print(f"    {f}")
    print()

    # Categorize files
    python_files = [f for f in all_files if f.suffix in PYTHON_EXTS]
    doc_files = [f for f in all_files if f.suffix in DOC_EXTS]
    workflow_files = [f for f in all_files if ".github/" in str(f).replace(os.sep, "/")]

    all_errors: List[str] = []

    # --- Check 1: Python imports -------------------------------------------
    print("[1/4] Checking Python imports...")
    import_errors = check_python_imports(python_files, all_files, verbose)
    all_errors.extend(import_errors)
    if not import_errors:
        print("   [OK] All imports resolve correctly.")
    else:
        print(f"   [FAIL] {len(import_errors)} broken import(s) found:")
        for e in import_errors:
            print(e)
    print()

    # --- Check 2: Hardcoded file path strings ------------------------------
    print("[2/4] Checking hardcoded file path strings...")
    path_errors = check_hardcoded_paths(python_files, all_files, verbose)
    all_errors.extend(path_errors)
    if not path_errors:
        print("   [OK] All file path strings resolve correctly.")
    else:
        print(f"   [FAIL] {len(path_errors)} broken path(s) found:")
        for e in path_errors:
            print(e)
    print()

    # --- Check 3: Documentation references ---------------------------------
    print("[3/4] Checking documentation file references...")
    doc_errors = check_doc_references(doc_files, all_files, verbose)
    all_errors.extend(doc_errors)
    if not doc_errors:
        print("   [OK] All documentation references resolve correctly.")
    else:
        print(f"   [FAIL] {len(doc_errors)} broken doc reference(s) found:")
        for e in doc_errors:
            print(e)
    print()

    # --- Check 4: Workflow script references --------------------------------
    print("[4/4] Checking workflow script references...")
    workflow_errors = check_workflow_references(workflow_files, all_files, verbose)
    all_errors.extend(workflow_errors)
    if not workflow_errors:
        print("   [OK] All workflow references resolve correctly.")
    else:
        print(f"   [FAIL] {len(workflow_errors)} broken workflow reference(s) found:")
        for e in workflow_errors:
            print(e)
    print()

    # --- Summary ------------------------------------------------------------
    print("=" * 60)
    total = len(all_errors)
    if total == 0:
        print("  [OK] ALL FILE REFERENCES VALID - no broken references found.")
    else:
        print(f"  [FAIL] {total} BROKEN REFERENCE(S) FOUND:")
        for e in all_errors:
            print(f"     {e}")
    print("=" * 60)

    return 1 if total > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
