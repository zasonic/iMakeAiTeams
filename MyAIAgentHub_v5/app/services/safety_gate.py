"""
services/safety_gate.py

Dangerous command and content detection for multi-agent workflows.

Scans agent-generated content (especially from task_scheduler) for patterns
that could cause harm if executed.  Returns a verdict with risk level
and reason so the orchestrator or UI can warn/block/confirm.

Verdicts:
    SAFE    — no dangerous patterns detected
    WARN    — medium-risk pattern found, recommend user confirmation
    BLOCK   — high-risk pattern found, refuse without explicit override
"""

import logging
import re
import unicodedata
from dataclasses import dataclass
from enum import Enum

log = logging.getLogger("MyAIEnv.safety_gate")


class RiskLevel(str, Enum):
    SAFE  = "safe"
    WARN  = "warn"
    BLOCK = "block"


@dataclass
class SafetyVerdict:
    level:   RiskLevel
    reason:  str = ""
    pattern: str = ""   # the pattern that matched (for logging)


# ── Dangerous shell command patterns ─────────────────────────────────────────

# HIGH risk — these can destroy data or compromise the system
_DANGEROUS_BASH_PREFIXES: list[tuple[str, str]] = [
    ("rm -rf /",              "Recursive delete from root"),
    ("rm -rf ~",              "Recursive delete of home directory"),
    ("rm -rf .",              "Recursive delete of current directory"),
    ("rm -r /",               "Recursive delete from root"),
    ("rmdir /s",              "Windows recursive directory delete"),
    ("del /f /s /q",          "Windows force-delete all files"),
    ("format ",               "Disk format command"),
    ("mkfs.",                 "Filesystem format command"),
    ("dd if=",                "Raw disk write (dd)"),
    ("> /dev/sda",            "Direct write to disk device"),
    ("chmod -R 777 /",        "Remove all file permissions recursively from root"),
    (":(){ :|:&};:",          "Fork bomb"),
    (":(){:|:&};:",           "Fork bomb (no spaces)"),
    ("shutdown",              "System shutdown"),
    ("reboot",                "System reboot"),
    ("init 0",                "System halt"),
    ("halt",                  "System halt"),
    ("poweroff",              "System poweroff"),
    ("killall",               "Kill all processes"),
    ("pkill -9",              "Force kill processes"),
]

# MEDIUM risk — code execution that could be exploited
_CODE_EXECUTION_PATTERNS: list[tuple[str, str]] = [
    ("eval ",                 "Dynamic code evaluation"),
    ("exec(",                 "Dynamic code execution"),
    ("python -c",             "Inline Python execution"),
    ("python3 -c",            "Inline Python execution"),
    ("node -e",               "Inline Node.js execution"),
    ("ruby -e",               "Inline Ruby execution"),
    ("perl -e",               "Inline Perl execution"),
    ("| sh",                  "Pipe to shell"),
    ("| bash",                "Pipe to bash"),
    ("| zsh",                 "Pipe to zsh"),
    ("| powershell",          "Pipe to PowerShell"),
    ("| pwsh",                "Pipe to PowerShell Core"),
    ("curl | sh",             "Download and execute"),
    ("wget | sh",             "Download and execute"),
    ("Invoke-Expression",     "PowerShell dynamic execution"),
    ("iex ",                  "PowerShell IEX shorthand"),
    ("Start-Process",         "PowerShell process spawn"),
    ("Add-Type",              "PowerShell .NET code injection"),
    ("os.system(",            "Python os.system call"),
    ("subprocess.call(",      "Python subprocess call"),
    ("subprocess.Popen(",     "Python subprocess Popen"),
    ("__import__(",           "Python dynamic import"),
]

# MEDIUM risk — network exfiltration
_EXFILTRATION_PATTERNS: list[tuple[str, str]] = [
    ("curl -X POST",          "HTTP POST (potential data exfiltration)"),
    ("curl --data",           "HTTP POST with data"),
    ("wget --post",           "HTTP POST via wget"),
    ("nc -e",                 "Netcat reverse shell"),
    ("ncat -e",               "Ncat reverse shell"),
    ("/dev/tcp/",             "Bash TCP connection"),
    ("ssh ",                  "SSH connection"),
    ("scp ",                  "SCP file transfer"),
    ("rsync ",                "Rsync file transfer"),
    ("ftp ",                  "FTP connection"),
]

# HIGH risk — credential/key access
_CREDENTIAL_PATTERNS: list[tuple[str, str]] = [
    (".ssh/id_rsa",           "SSH private key access"),
    (".ssh/id_ed25519",       "SSH private key access"),
    (".aws/credentials",      "AWS credentials access"),
    (".env",                  "Environment file access"),
    ("API_KEY",               "API key reference"),
    ("SECRET_KEY",            "Secret key reference"),
    ("PASSWORD",              "Password reference"),
    ("PRIVATE_KEY",           "Private key reference"),
    ("/etc/shadow",           "System password file"),
    ("/etc/passwd",           "System user file"),
    ("keychain",              "Keychain access"),
]


# ── Text normalization (defeats obfuscation tricks) ──────────────────────────

def _normalize(text: str) -> str:
    """Normalize unicode tricks, quote stripping, variable expansion markers."""
    # Normalize unicode to ASCII equivalents
    text = unicodedata.normalize("NFKD", text)
    # Remove zero-width chars, soft hyphens, etc.
    text = re.sub(r'[\u200b-\u200f\u2028-\u202f\u00ad\ufeff]', '', text)
    # Collapse quote-splitting tricks: r''m -> rm, r""m -> rm
    text = re.sub(r"['\"]", "", text)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


# ── HIGH-risk regex patterns (survive obfuscation) ───────────────────────────

_HIGH_RISK_REGEXES: list[tuple[str, str]] = [
    (r'\brm\b.*-[rR].*[/~.]',         "Recursive delete"),
    (r'\bdd\b.*\bif=',                 "Raw disk write"),
    (r'\bmkfs\b',                       "Filesystem format"),
    (r'\bfind\b.*-delete',             "Find with delete"),
    (r'>\s*/dev/[sh]d',               "Direct device write"),
    (r'chmod\s+.*777\s+/',             "Recursive permission change"),
    (r':\(\)\s*\{.*\|.*&\s*\}\s*;',   "Fork bomb"),
    (r'\b(shutdown|reboot|halt|poweroff|init\s+0)\b', "System power command"),
    (r'\b(killall|pkill\s+-9)\b',      "Mass process kill"),
    (r'\bformat\s+[a-zA-Z]:',         "Windows disk format"),
    (r'\brmdir\s+/s\b',               "Windows recursive delete"),
    (r'\bdel\s+/f\s+/s\s+/q\b',       "Windows force-delete"),
]


# ── Main scan function ───────────────────────────────────────────────────────

def scan_command(command: str) -> SafetyVerdict:
    """
    Scan a single shell command string for dangerous patterns.
    Uses text normalization to defeat quote-splitting and unicode obfuscation.
    Returns a SafetyVerdict.
    """
    if not command or not command.strip():
        return SafetyVerdict(RiskLevel.SAFE)

    normalized = _normalize(command.lower())

    # HIGH risk: regex patterns that survive obfuscation
    for pattern, reason in _HIGH_RISK_REGEXES:
        if re.search(pattern, normalized):
            return SafetyVerdict(RiskLevel.BLOCK, reason, pattern)

    # HIGH risk: substring patterns (credential access)
    for pattern, reason in _CREDENTIAL_PATTERNS:
        if pattern.lower() in normalized:
            return SafetyVerdict(RiskLevel.BLOCK, reason, pattern)

    # Also run original substring checks against normalized text
    for pattern, reason in _DANGEROUS_BASH_PREFIXES:
        if pattern.lower() in normalized:
            return SafetyVerdict(RiskLevel.BLOCK, reason, pattern)

    # MEDIUM risk checks
    for pattern, reason in _CODE_EXECUTION_PATTERNS:
        if pattern.lower() in normalized:
            return SafetyVerdict(RiskLevel.WARN, reason, pattern)

    for pattern, reason in _EXFILTRATION_PATTERNS:
        if pattern.lower() in normalized:
            return SafetyVerdict(RiskLevel.WARN, reason, pattern)

    return SafetyVerdict(RiskLevel.SAFE)


def scan_content(content: str) -> SafetyVerdict:
    """
    Scan agent-generated content (which may contain embedded commands,
    code blocks, or instructions) for dangerous patterns.

    Extracts code blocks and scans each one, plus scans the full text
    for credential patterns.
    """
    if not content or not content.strip():
        return SafetyVerdict(RiskLevel.SAFE)

    # Extract code blocks
    code_blocks = re.findall(r"```(?:\w+)?\s*\n([\s\S]*?)```", content)

    # Also check inline backtick commands
    inline_commands = re.findall(r"`([^`]{5,})`", content)

    worst = SafetyVerdict(RiskLevel.SAFE)

    # Scan each code block
    for block in code_blocks + inline_commands:
        for line in block.split("\n"):
            verdict = scan_command(line.strip())
            if _severity(verdict.level) > _severity(worst.level):
                worst = verdict

    # Scan full text for credential patterns (use normalized text)
    normalized = _normalize(content.lower())
    for pattern, reason in _CREDENTIAL_PATTERNS:
        if pattern.lower() in normalized:
            v = SafetyVerdict(RiskLevel.WARN,
                              f"Content references {reason}", pattern)
            if _severity(v.level) > _severity(worst.level):
                worst = v

    return worst


def scan_workflow_task(task_name: str, agent_role: str,
                       input_data: str, output_data: str) -> SafetyVerdict:
    """
    Scan a workflow task's input and output for dangerous content.
    """
    worst = SafetyVerdict(RiskLevel.SAFE)

    for text in [input_data, output_data]:
        if text:
            v = scan_content(text)
            if _severity(v.level) > _severity(worst.level):
                worst = v

    if worst.level != RiskLevel.SAFE:
        log.warning("Safety gate flagged task '%s' (role=%s): %s — %s",
                    task_name, agent_role, worst.level.value, worst.reason)

    return worst


# ── Helpers ───────────────────────────────────────────────────────────────────

def _severity(level: RiskLevel) -> int:
    return {RiskLevel.SAFE: 0, RiskLevel.WARN: 1, RiskLevel.BLOCK: 2}[level]


def get_all_patterns() -> dict:
    """Return all patterns for display in the Settings UI."""
    return {
        "dangerous_commands": [
            {"pattern": p, "reason": r, "risk": "block"}
            for p, r in _DANGEROUS_BASH_PREFIXES
        ],
        "code_execution": [
            {"pattern": p, "reason": r, "risk": "warn"}
            for p, r in _CODE_EXECUTION_PATTERNS
        ],
        "exfiltration": [
            {"pattern": p, "reason": r, "risk": "warn"}
            for p, r in _EXFILTRATION_PATTERNS
        ],
        "credentials": [
            {"pattern": p, "reason": r, "risk": "block"}
            for p, r in _CREDENTIAL_PATTERNS
        ],
    }
