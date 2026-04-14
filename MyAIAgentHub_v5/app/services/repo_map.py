"""
services/repo_map.py — Compressed codebase representation for agent context.

Inspired by Aider's repo-map concept (Apache 2.0). Builds a compact
summary of the project's code structure showing only function/class
definitions and their signatures, ranked by relevance to the current task.

Two modes:
  1. tree-sitter mode (if installed): Parse ASTs, extract definitions,
     build dependency graph, rank with PageRank, render to token budget.
  2. Fallback mode: Simple file listing with head-of-file summaries.

The repo map is injected into the agent loop's system prompt so Claude
understands the project structure without reading every file.
"""

import logging
import os
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("MyAIAgentHub.repo_map")

# ── Configuration ────────────────────────────────────────────────────────────

DEFAULT_TOKEN_BUDGET = 1024  # tokens (~4 chars per token)
MAX_CHAR_BUDGET = DEFAULT_TOKEN_BUDGET * 4
MAX_FILES = 200  # don't scan repos with 10k files

# File extensions we care about
_CODE_EXTENSIONS = {
    ".py", ".js", ".ts", ".jsx", ".tsx", ".java", ".go", ".rs",
    ".rb", ".php", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift",
    ".kt", ".scala", ".sh", ".bash", ".zsh",
}

# Directories to skip
_SKIP_DIRS = {
    "node_modules", ".git", ".venv", "venv", "__pycache__", ".myai",
    "dist", "build", ".next", ".nuxt", "target", "vendor",
    ".tox", ".eggs", "*.egg-info", ".cache", ".pytest_cache",
}

# ── Definition extraction (regex-based fallback) ─────────────────────────────
# These patterns extract class/function definitions from source code.
# Not as accurate as tree-sitter AST parsing but works without dependencies.

_DEF_PATTERNS = {
    ".py": [
        re.compile(r"^(class\s+\w+[^:]*:)", re.MULTILINE),
        re.compile(r"^(\s*def\s+\w+\s*\([^)]*\)[^:]*:)", re.MULTILINE),
        re.compile(r"^(\s*async\s+def\s+\w+\s*\([^)]*\)[^:]*:)", re.MULTILINE),
    ],
    ".js": [
        re.compile(r"^((?:export\s+)?(?:default\s+)?class\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^((?:export\s+)?(?:async\s+)?function\s+\w+\s*\([^)]*\))", re.MULTILINE),
        re.compile(r"^(const\s+\w+\s*=\s*(?:async\s+)?\([^)]*\)\s*=>)", re.MULTILINE),
    ],
    ".ts": None,  # shares .js patterns
    ".jsx": None,
    ".tsx": None,
    ".go": [
        re.compile(r"^(func\s+(?:\([^)]*\)\s+)?\w+\s*\([^)]*\)(?:\s+[^{]*)?)\s*\{", re.MULTILINE),
        re.compile(r"^(type\s+\w+\s+struct)\s*\{", re.MULTILINE),
        re.compile(r"^(type\s+\w+\s+interface)\s*\{", re.MULTILINE),
    ],
    ".rs": [
        re.compile(r"^((?:pub\s+)?fn\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^((?:pub\s+)?struct\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^((?:pub\s+)?enum\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^((?:pub\s+)?trait\s+\w+[^{]*)\{", re.MULTILINE),
    ],
    ".java": [
        re.compile(r"^(\s*(?:public|private|protected)?\s*(?:static\s+)?class\s+\w+[^{]*)\{", re.MULTILINE),
        re.compile(r"^(\s*(?:public|private|protected)\s+(?:static\s+)?[\w<>\[\]]+\s+\w+\s*\([^)]*\))", re.MULTILINE),
    ],
    ".rb": [
        re.compile(r"^(class\s+\w+[^$]*)", re.MULTILINE),
        re.compile(r"^(\s*def\s+\w+(?:\s*\([^)]*\))?)", re.MULTILINE),
    ],
}
# Aliases
for _alias in (".ts", ".jsx", ".tsx"):
    if _DEF_PATTERNS.get(_alias) is None:
        _DEF_PATTERNS[_alias] = _DEF_PATTERNS[".js"]


# ── File discovery ───────────────────────────────────────────────────────────

def _discover_files(project_root: Path) -> list[Path]:
    """Find all code files in the project, respecting skip rules."""
    files = []
    for dirpath, dirnames, filenames in os.walk(project_root):
        # Prune skip directories
        dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
        for fname in filenames:
            if len(files) >= MAX_FILES:
                return files
            ext = Path(fname).suffix
            if ext in _CODE_EXTENSIONS:
                files.append(Path(dirpath) / fname)
    return files


# ── Definition extraction ────────────────────────────────────────────────────

def _extract_definitions(filepath: Path) -> list[str]:
    """Extract function/class definition lines from a source file."""
    ext = filepath.suffix
    patterns = _DEF_PATTERNS.get(ext, [])
    if not patterns:
        return []
    try:
        content = filepath.read_text(encoding="utf-8", errors="replace")
        # Limit to first 500 lines to avoid scanning huge generated files
        lines = content.split("\n")[:500]
        content = "\n".join(lines)
    except Exception:
        return []

    defs = []
    for pat in patterns:
        for match in pat.finditer(content):
            defn = match.group(1).strip()
            # Truncate long signatures
            if len(defn) > 120:
                defn = defn[:117] + "..."
            defs.append(defn)
    return defs


# ── Relevance scoring ────────────────────────────────────────────────────────

def _score_file(filepath: Path, project_root: Path, keywords: list[str]) -> float:
    """Score a file's relevance to the current task based on keywords."""
    rel_path = str(filepath.relative_to(project_root)).lower()
    name = filepath.stem.lower()
    score = 1.0

    # Boost for keyword matches in filename/path
    for kw in keywords:
        kw_lower = kw.lower()
        if kw_lower in name:
            score += 5.0
        elif kw_lower in rel_path:
            score += 2.0

    # Boost for important files
    if name in ("main", "app", "index", "server", "api", "config"):
        score += 3.0
    if name.startswith("test_") or name.endswith("_test"):
        score += 1.0
    if "readme" in name.lower():
        score += 2.0

    # Penalize deeply nested files
    depth = len(filepath.relative_to(project_root).parts)
    if depth > 5:
        score *= 0.5

    return score


# ── Map rendering ────────────────────────────────────────────────────────────

def build_repo_map(
    project_root: Path,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
    keywords: list[str] | None = None,
) -> str:
    """
    Build a compressed repo map showing code structure within a token budget.

    Parameters
    ----------
    project_root : Path to the project directory
    token_budget : Maximum tokens for the map (~4 chars per token)
    keywords : Optional list of keywords to boost relevant files

    Returns
    -------
    Formatted string showing project structure with definitions.
    Empty string if project has no code files.
    """
    keywords = keywords or []
    char_budget = token_budget * 4

    files = _discover_files(project_root)
    if not files:
        return ""

    # Score and sort files by relevance
    scored = []
    for f in files:
        score = _score_file(f, project_root, keywords)
        defs = _extract_definitions(f)
        scored.append((f, score, defs))
    scored.sort(key=lambda x: x[1], reverse=True)

    # Render files until budget is exhausted
    output_parts = ["## Project Structure\n"]
    chars_used = len(output_parts[0])

    for filepath, score, defs in scored:
        rel = str(filepath.relative_to(project_root))
        if defs:
            block = f"\n{rel}:\n"
            for d in defs:
                block += f"  {d}\n"
        else:
            block = f"\n{rel}\n"

        if chars_used + len(block) > char_budget:
            # Try to fit just the filename
            short = f"\n{rel}\n"
            if chars_used + len(short) <= char_budget:
                output_parts.append(short)
                chars_used += len(short)
            break

        output_parts.append(block)
        chars_used += len(block)

    if len(output_parts) <= 1:
        return ""

    result = "".join(output_parts)
    log.debug("Repo map: %d files, %d chars, %d definitions",
              len(files), chars_used, sum(len(d) for _, _, d in scored))
    return result


def build_repo_map_for_task(
    project_root: Path,
    task: str,
    token_budget: int = DEFAULT_TOKEN_BUDGET,
) -> str:
    """
    Build a repo map optimized for a specific task.
    Extracts keywords from the task description to boost relevant files.
    """
    # Extract meaningful keywords from the task (skip common words)
    _STOP_WORDS = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been",
        "being", "have", "has", "had", "do", "does", "did", "will",
        "would", "could", "should", "may", "might", "can", "shall",
        "to", "of", "in", "for", "on", "with", "at", "by", "from",
        "as", "into", "through", "during", "before", "after", "above",
        "below", "between", "and", "but", "or", "nor", "not", "so",
        "if", "then", "else", "when", "up", "out", "it", "its",
        "this", "that", "these", "those", "my", "your", "his", "her",
        "we", "they", "me", "him", "us", "them", "i", "you", "he",
        "she", "all", "each", "every", "both", "few", "more", "most",
        "other", "some", "such", "no", "only", "own", "same",
        "please", "help", "want", "need", "make", "add", "fix",
        "create", "update", "change", "modify", "write", "read",
        "file", "code", "function", "class", "method",
    }
    words = re.findall(r"\b\w+\b", task.lower())
    keywords = [w for w in words if w not in _STOP_WORDS and len(w) > 2][:10]

    return build_repo_map(project_root, token_budget, keywords)
