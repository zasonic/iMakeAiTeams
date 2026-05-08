"""build-scripts/fetch_bundled_assets.py — populate branding/sidecar-bundle/.

Phase 9 build step:

  1. Resolve the **latest** llama.cpp release (per the operator's PR-3 choice
     to keep the binary current rather than pinning a tag).
  2. Download the Windows CPU x64 server zip, extract llama-server.exe + the
     shipped DLLs into branding/sidecar-bundle/llama-server/.
  3. Hit the Hugging Face metadata endpoint for each catalog model and write
     branding/sidecar-bundle/bundled_models.json with sha256 + size_bytes.

The runtime (services/bundled_server.py) reads bundled_models.json to
validate downloads, and resolves the binary via paths.bundled_server_binary().

Usage (from repo root):
    python build-scripts/fetch_bundled_assets.py
"""

from __future__ import annotations

import io
import json
import shutil
import sys
import urllib.parse
import zipfile
from pathlib import Path
from typing import Iterable

import urllib.request


REPO_ROOT = Path(__file__).resolve().parent.parent
BUNDLE_DIR = REPO_ROOT / "branding" / "sidecar-bundle"
LLAMA_DIR = BUNDLE_DIR / "llama-server"
CATALOG_PATH = BUNDLE_DIR / "bundled_models.json"

LLAMACPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
HF_TREE_API = "https://huggingface.co/api/models/{repo}/tree/main"

# Catalog of models the wizard's Quick Start can offer. Mirror the ids in
# backend/services/bundled_server.py:_DEFAULT_MODELS — that file owns the
# runtime defaults; this script just fills sha256/size at build time.
MODELS = [
    {
        "model_id": "Qwen3-4B-Instruct-Q4_K_M",
        "repo":     "Qwen/Qwen3-4B-Instruct-GGUF",
        "filename": "Qwen3-4B-Instruct-Q4_K_M.gguf",
    },
]


def _http_get(url: str, *, accept: str = "application/json") -> bytes:
    req = urllib.request.Request(url, headers={"Accept": accept,
                                                "User-Agent": "iMakeAiTeams-build"})
    with urllib.request.urlopen(req, timeout=30) as resp:  # noqa: S310
        return resp.read()


def _resolve_llamacpp_zip_url() -> tuple[str, str]:
    """Return (release_tag, asset_download_url) for the Windows CPU x64 build."""
    body = _http_get(LLAMACPP_RELEASES_API)
    data = json.loads(body)
    tag = data["tag_name"]
    # llama.cpp publishes per-release archives with names like
    # "llama-<tag>-bin-win-cpu-x64.zip" or "llama-<tag>-bin-win-x64.zip"; the
    # exact pattern has shifted over time. Match the first archive whose name
    # mentions both 'win' and ('cpu' or x64) and ends with .zip.
    candidates = []
    for asset in data.get("assets", []):
        name = asset.get("name", "").lower()
        if "win" in name and name.endswith(".zip") and ("cpu" in name or "x64" in name):
            candidates.append(asset["browser_download_url"])
    if not candidates:
        raise RuntimeError(f"no Windows CPU asset found in release {tag}")
    # Prefer one that explicitly says cpu — avoid pulling a CUDA-only build.
    cpu_first = [u for u in candidates if "cpu" in u.lower()]
    return tag, (cpu_first[0] if cpu_first else candidates[0])


def _extract_llama_server(zip_bytes: bytes, out_dir: Path) -> list[str]:
    """Extract llama-server.exe + every DLL into ``out_dir`` (flat layout)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name.lower()
            if not name:
                continue
            if name == "llama-server.exe" or name.endswith(".dll"):
                target = out_dir / Path(member.filename).name
                with zf.open(member) as src, open(target, "wb") as dst:
                    shutil.copyfileobj(src, dst)
                extracted.append(target.name)
    return extracted


def _hf_lookup(repo: str, filename: str) -> tuple[str, int]:
    body = _http_get(HF_TREE_API.format(repo=urllib.parse.quote(repo, safe="/")))
    items = json.loads(body)
    for entry in items:
        if not isinstance(entry, dict) or entry.get("path") != filename:
            continue
        lfs = entry.get("lfs") or {}
        sha = lfs.get("sha256") or lfs.get("oid") or ""
        size = lfs.get("size") or entry.get("size") or 0
        if not sha or not size:
            raise RuntimeError(f"HF metadata for {repo}/{filename} missing sha256/size")
        return str(sha), int(size)
    raise RuntimeError(f"file {filename} not found in {repo}")


def _populate_catalog(models: Iterable[dict]) -> dict[str, dict]:
    catalog: dict[str, dict] = {}
    for m in models:
        sha, size = _hf_lookup(m["repo"], m["filename"])
        catalog[m["model_id"]] = {
            "repo":                 m["repo"],
            "filename":             m["filename"],
            "expected_sha256":      sha,
            "expected_size_bytes":  size,
        }
    return catalog


def main() -> int:
    print("[fetch_bundled_assets] resolving latest llama.cpp release…")
    try:
        tag, zip_url = _resolve_llamacpp_zip_url()
    except Exception as exc:
        print(f"[fetch_bundled_assets] could not resolve release: {exc}", file=sys.stderr)
        return 1
    print(f"[fetch_bundled_assets] release tag {tag}, asset {zip_url}")

    BUNDLE_DIR.mkdir(parents=True, exist_ok=True)
    print("[fetch_bundled_assets] downloading llama-server zip…")
    try:
        zip_bytes = _http_get(zip_url, accept="application/octet-stream")
    except Exception as exc:
        print(f"[fetch_bundled_assets] download failed: {exc}", file=sys.stderr)
        return 1

    if LLAMA_DIR.exists():
        shutil.rmtree(LLAMA_DIR)
    extracted = _extract_llama_server(zip_bytes, LLAMA_DIR)
    print(f"[fetch_bundled_assets] extracted {len(extracted)} files into {LLAMA_DIR}")
    if "llama-server.exe" not in extracted:
        print("[fetch_bundled_assets] WARNING: llama-server.exe not found in zip", file=sys.stderr)
        return 1

    print("[fetch_bundled_assets] resolving HF metadata for catalog models…")
    try:
        catalog = _populate_catalog(MODELS)
    except Exception as exc:
        print(f"[fetch_bundled_assets] HF metadata lookup failed: {exc}", file=sys.stderr)
        return 1

    catalog["_meta"] = {"llama_release_tag": tag}
    CATALOG_PATH.write_text(json.dumps(catalog, indent=2), encoding="utf-8")
    print(f"[fetch_bundled_assets] wrote catalog → {CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
