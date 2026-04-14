"""
services/project_memory.py — CLAUDE.md / project memory loader.

Loads persistent project knowledge from markdown files and injects it
into agent system prompts. Supports hierarchical loading and @imports.

Load order (later overrides earlier):
  1. ~/.myai/CLAUDE.md        (user-level defaults)
  2. {project_root}/CLAUDE.md (project-level rules)
  3. {project_root}/.myai/MEMORY.md (auto-generated long-term memory)

Token-efficient base rules (from drona23/claude-token-efficient, MIT)
are prepended automatically to reduce output verbosity by ~60%.
"""

import logging
import re
from pathlib import Path
from typing import Optional

log = logging.getLogger("MyAIAgentHub.project_memory")

# ── Token-efficient base rules ───────────────────────────────────────────────
# Sourced from drona23/claude-token-efficient patterns (MIT licensed).
# These reduce Claude output verbosity by ~63% with zero accuracy loss.

_BASE_RULES = """## Response Rules
- No sycophantic openers ("Sure!", "Great question!", "I'd be happy to help")
- No closing fluff ("Hope this helps!", "Let me know if you need anything")
- Do not restate the user's question before answering
- No unsolicited suggestions or over-engineered alternatives
- Read files before modifying — never edit blind
- Prefer targeted edits over full-file rewrites
- Do not re-read files already in context unless they changed
- Use ASCII-only formatting (no em dashes, smart quotes, decorative Unicode)
- State findings first, methodology after — lead with the answer
- When returning structured data, use JSON or markdown tables
- User instructions always override these rules"""


# ── Import pattern ───────────────────────────────────────────────────────────

_IMPORT_RE = re.compile(r"^@(\S+)\s*$", re.MULTILINE)


def _resolve_imports(content: str, base_dir: Path, depth: int = 0) -> str:
    """Resolve @path imports in CLAUDE.md content. Max depth 3 to prevent loops."""
    if depth > 3:
        return content

    def _replace(match):
        rel_path = match.group(1)
        target = (base_dir / rel_path).resolve()
        if target.exists() and target.is_file():
            try:
                imported = target.read_text(encoding="utf-8", errors="replace")
                # Recursively resolve imports in the imported file
                return _resolve_imports(imported, target.parent, depth + 1)
            except Exception as exc:
                log.debug("Import failed for %s: %s", rel_path, exc)
                return f"[Import failed: {rel_path}]"
        return f"[File not found: {rel_path}]"

    return _IMPORT_RE.sub(_replace, content)


# ── Main loader ──────────────────────────────────────────────────────────────

def load_project_memory(project_root: Path) -> str:
    """
    Load and merge all project memory files.

    Returns a single string suitable for prepending to system prompts.
    Includes base token-efficiency rules + all loaded memory files.
    Returns empty string if no memory files exist.
    """
    sections: list[str] = []
    loaded_files: list[str] = []

    # 1. User-level defaults
    user_claude = Path.home() / ".myai" / "CLAUDE.md"
    if user_claude.exists():
        try:
            content = user_claude.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                content = _resolve_imports(content, user_claude.parent)
                sections.append(f"## User Defaults\n{content}")
                loaded_files.append(str(user_claude))
        except Exception as exc:
            log.debug("Failed to load user CLAUDE.md: %s", exc)

    # 2. Project-level CLAUDE.md
    project_claude = project_root / "CLAUDE.md"
    if project_claude.exists():
        try:
            content = project_claude.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                content = _resolve_imports(content, project_root)
                sections.append(f"## Project Rules\n{content}")
                loaded_files.append(str(project_claude))
        except Exception as exc:
            log.debug("Failed to load project CLAUDE.md: %s", exc)

    # 3. Auto-generated long-term memory
    memory_md = project_root / ".myai" / "MEMORY.md"
    if memory_md.exists():
        try:
            content = memory_md.read_text(encoding="utf-8", errors="replace").strip()
            if content:
                sections.append(f"## Project Memory\n{content}")
                loaded_files.append(str(memory_md))
        except Exception as exc:
            log.debug("Failed to load MEMORY.md: %s", exc)

    if not sections:
        # Return just base rules even with no project files
        return f"# Agent Guidelines\n\n{_BASE_RULES}"

    if loaded_files:
        log.info("Loaded project memory from: %s", ", ".join(loaded_files))

    # Combine: base rules first (cacheable prefix), then project-specific
    combined = f"# Agent Guidelines\n\n{_BASE_RULES}\n\n" + "\n\n".join(sections)

    return combined


def generate_starter_claude_md(project_root: Path) -> Optional[Path]:
    """
    Generate a starter CLAUDE.md by scanning the project structure.
    Returns the path if created, None if file already exists.
    """
    target = project_root / "CLAUDE.md"
    if target.exists():
        return None

    # Scan project for key signals
    lines = ["# Project Rules for AI Agents\n"]

    # Detect language/framework
    if (project_root / "package.json").exists():
        lines.append("## Stack\n- Node.js / JavaScript/TypeScript project")
        if (project_root / "tsconfig.json").exists():
            lines.append("- TypeScript enabled")
        lines.append("")
    elif (project_root / "pyproject.toml").exists() or (project_root / "setup.py").exists():
        lines.append("## Stack\n- Python project")
        if (project_root / "pyproject.toml").exists():
            lines.append("- Uses pyproject.toml for configuration")
        lines.append("")
    elif (project_root / "Cargo.toml").exists():
        lines.append("## Stack\n- Rust project\n")
    elif (project_root / "go.mod").exists():
        lines.append("## Stack\n- Go project\n")

    # Detect test framework
    if (project_root / "pytest.ini").exists() or (project_root / "pyproject.toml").exists():
        lines.append("## Testing\n- Run tests: `python -m pytest`\n")
    elif (project_root / "package.json").exists():
        lines.append("## Testing\n- Run tests: `npm test`\n")

    # Add common rules
    lines.append("## Code Style")
    lines.append("- Follow existing patterns in the codebase")
    lines.append("- Keep functions focused and small")
    lines.append("- Write descriptive commit messages")
    lines.append("")
    lines.append("## Important")
    lines.append("- Always read files before modifying them")
    lines.append("- Run tests after making changes")
    lines.append("- Never commit secrets or API keys")

    content = "\n".join(lines) + "\n"
    target.write_text(content, encoding="utf-8")
    log.info("Generated starter CLAUDE.md at %s", target)
    return target
