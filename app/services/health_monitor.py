"""
services/health_monitor.py — Pre-flight health checks.

Run check_all() at app startup or on demand.
Returns a list of {component, status, message} dicts.

Status values: "pass" | "warn" | "fail"
"""

import shutil
import subprocess
import concurrent.futures
from pathlib import Path


def _check(component: str, fn) -> dict:
    """Run a single check and normalise the result."""
    try:
        result = fn()
        return {"component": component, **result}
    except Exception as exc:
        return {"component": component, "status": "fail", "message": str(exc)}


def check_anthropic_api(api_key: str) -> dict:
    """Send a minimal Anthropic API call to verify connectivity."""
    if not api_key or not api_key.strip():
        return {"status": "fail", "message": "No API key configured. Add it in Settings."}
    try:
        from anthropic import Anthropic
        client = Anthropic(api_key=api_key)
        client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=5,
            messages=[{"role": "user", "content": "ping"}],
        )
        return {"status": "pass", "message": "Anthropic API reachable."}
    except Exception as exc:
        name = type(exc).__name__
        if "Authentication" in name:
            return {"status": "fail", "message": "API key is invalid. Check Settings."}
        if "Connection" in name or "Timeout" in name:
            return {"status": "fail", "message": "Cannot reach Anthropic API. Check your internet connection."}
        return {"status": "warn", "message": f"API check inconclusive: {exc}"}


def check_local_models(ollama_url: str, lmstudio_url: str) -> dict:
    """Check if at least one local model backend is reachable."""
    import requests
    results = []
    for name, url, endpoint in [
        ("Ollama", ollama_url, "/api/tags"),
        ("LM Studio", lmstudio_url, "/v1/models"),
    ]:
        try:
            r = requests.get(url + endpoint, timeout=2)
            if r.status_code == 200:
                results.append(name)
        except Exception:
            pass

    if results:
        return {"status": "pass", "message": f"Local model backend(s) available: {', '.join(results)}"}
    return {
        "status": "warn",
        "message": "No local model backends reachable. Ollama or LM Studio enables free local inference.",
    }


def check_rag_index(app_root: str) -> dict:
    """Check RAG index chunk count via ChromaDB (the active document store)."""
    try:
        from services import semantic_search
        if not semantic_search.is_available():
            return {"status": "warn",
                    "message": "Semantic search is not initialised — RAG unavailable. "
                               "Install chromadb and sentence-transformers if needed."}
        count = semantic_search.document_count()
        if count > 0:
            return {"status": "pass", "message": f"RAG index loaded with {count} document chunks in ChromaDB."}
        return {"status": "warn",
                "message": "RAG index is empty — no documents indexed yet. Add docs in the Documents view."}
    except Exception as exc:
        return {"status": "warn", "message": f"Could not check RAG index: {exc}"}


def check_git() -> dict:
    try:
        r = subprocess.run(["git", "--version"], capture_output=True, timeout=5)
        if r.returncode == 0:
            return {"status": "pass", "message": r.stdout.decode().strip()[:60]}
        return {"status": "warn", "message": "Git found but returned non-zero exit."}
    except FileNotFoundError:
        return {"status": "warn", "message": "Git not installed — optional."}
    except Exception as exc:
        return {"status": "warn", "message": str(exc)}


def check_disk_space(output_dir: str, min_gb: float = 2.0) -> dict:
    try:
        p = Path(output_dir)
        p.mkdir(parents=True, exist_ok=True)
        usage = shutil.disk_usage(str(p))
        free_gb = usage.free / (1024 ** 3)
        if free_gb >= min_gb:
            return {"status": "pass", "message": f"{free_gb:.1f} GB free."}
        if free_gb >= 0.5:
            return {"status": "warn", "message": f"Only {free_gb:.1f} GB free. Consider freeing disk space."}
        return {"status": "fail", "message": f"Only {free_gb:.1f} GB free. App may fail."}
    except Exception as exc:
        return {"status": "warn", "message": f"Could not check disk space: {exc}"}


def check_memory_health() -> dict:
    """Check if memory fact extraction is working reliably."""
    try:
        from services.memory import _extract_attempts, _extract_failures
        if _extract_attempts < 10:
            return {"status": "pass",
                    "message": "Not enough data yet to assess memory health"}
        rate = _extract_failures / _extract_attempts
        if rate > 0.5:
            return {"status": "warn",
                    "message": f"Fact extraction failing {rate:.0%} of the time — "
                               "consider using a larger local model (13B+)"}
        return {"status": "pass",
                "message": f"Extraction success rate: {1-rate:.0%}"}
    except Exception:
        return {"status": "pass", "message": "Memory health check skipped"}


def check_all(
    api_key: str = "",
    app_root: str = ".",
    ollama_url: str = "http://localhost:11434",
    lmstudio_url: str = "http://localhost:1234",
    skip_api: bool = False,
) -> list[dict]:
    """
    Run all health checks and return a list of results.
    Checks run in parallel for speed.
    """
    checks = {
        "Disk space":     lambda: check_disk_space(app_root),
        "Local models":   lambda: check_local_models(ollama_url, lmstudio_url),
        "RAG index":      lambda: check_rag_index(app_root),
        "Memory":         check_memory_health,
        "Git":            check_git,
    }
    if not skip_api and api_key:
        checks["Anthropic API"] = lambda: check_anthropic_api(api_key)

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = {pool.submit(fn): name for name, fn in checks.items()}
        for future in concurrent.futures.as_completed(futures):
            name = futures[future]
            try:
                result = future.result(timeout=15)
                results.append({"component": name, **result})
            except Exception as exc:
                results.append({"component": name, "status": "warn", "message": str(exc)})

    order = {"fail": 0, "warn": 1, "pass": 2}
    results.sort(key=lambda r: order.get(r.get("status", "warn"), 1))
    return results


def has_blocking_failures(results: list[dict]) -> bool:
    """True if any check returned 'fail'."""
    return any(r.get("status") == "fail" for r in results)
