"""build-scripts/fetch_bundled_assets.py — populate branding/sidecar-bundle/.

Phase 9 build step (extended in PR 17):

  1. Resolve the **latest** llama.cpp release (per the operator's PR-3 choice
     to keep the binary current rather than pinning a tag).
  2. Download the Windows CPU x64 server zip, extract llama-server.exe + the
     shipped DLLs into branding/sidecar-bundle/llama-server/.
  3. Hit the Hugging Face metadata endpoint for each catalog model and write
     branding/sidecar-bundle/bundled_models.json with sha256 + size_bytes.
  4. PR 17: resolve the latest Whisper.cpp + Piper Windows releases, extract
     whisper-cli.exe / piper.exe + their DLLs into the sidecar bundle, and
     write branding/sidecar-bundle/voice_assets.json with the per-model sha256
     + size for first-run download verification.

The runtime (services/bundled_server.py + services/voice.py) reads the
catalogs to validate downloads, and resolves binaries via
paths.bundled_server_binary() / paths.whisper_binary() / paths.piper_binary().

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
WHISPER_DIR = BUNDLE_DIR / "whisper"
PIPER_DIR = BUNDLE_DIR / "piper"
CATALOG_PATH = BUNDLE_DIR / "bundled_models.json"
VOICE_CATALOG_PATH = BUNDLE_DIR / "voice_assets.json"

LLAMACPP_RELEASES_API = "https://api.github.com/repos/ggml-org/llama.cpp/releases/latest"
WHISPER_RELEASES_API = "https://api.github.com/repos/ggerganov/whisper.cpp/releases/latest"
PIPER_RELEASES_API = "https://api.github.com/repos/rhasspy/piper/releases/latest"
HF_TREE_API = "https://huggingface.co/api/models/{repo}/tree/main"

# PR 17: HuggingFace bin URL for Whisper.cpp models. Piper voices live on the
# rhasspy GitHub release pages but the per-voice download is a single .tar.gz
# wrapping the .onnx + .json — VoiceService streams those at runtime; the
# build catalog only records expected sha256/size for verification.
WHISPER_HF_TREE = "https://huggingface.co/api/models/ggerganov/whisper.cpp/tree/main"

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

# PR 17: voice asset catalog. STT models are Whisper.cpp .bin files hosted on
# the ggerganov/whisper.cpp HF repo. TTS voices are Piper .onnx + .json
# pairs hosted on the rhasspy/piper-voices HF repo. We bundle one default of
# each so the wizard's quick path can complete without browsing a model list.
WHISPER_MODELS = [
    {
        "model_id": "whisper-base.en",
        "filename": "ggml-base.en.bin",
    },
]

# Each Piper voice ships as `<voice>.onnx` + `<voice>.onnx.json`. We mirror
# the rhasspy/piper-voices HF layout: en/en_US/amy/medium/<files>.
PIPER_VOICES = [
    {
        "voice_id": "en_US-amy-medium",
        "model_filename": "en_US-amy-medium.onnx",
        "config_filename": "en_US-amy-medium.onnx.json",
        "hf_subpath": "en/en_US/amy/medium",
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


# ── PR 17: voice binary fetchers (Whisper.cpp + Piper) ──────────────────────


def _resolve_voice_zip_url(api_url: str, *, name_filter: str) -> tuple[str, str]:
    """Return (release_tag, asset_download_url) for a Windows zip in the
    referenced GitHub release. ``name_filter`` is matched as a substring
    against the asset file name (case-insensitive)."""
    body = _http_get(api_url)
    data = json.loads(body)
    tag = data.get("tag_name", "")
    for asset in data.get("assets", []):
        name = asset.get("name", "").lower()
        if name.endswith(".zip") and name_filter in name and "win" in name:
            return tag, asset["browser_download_url"]
    raise RuntimeError(f"no Windows asset matching {name_filter!r} found in {api_url}")


def _extract_named(zip_bytes: bytes, out_dir: Path,
                   *, primary: str, also_dlls: bool = True) -> list[str]:
    """Extract ``primary`` and (optionally) every .dll into ``out_dir`` as a
    flat layout. Returns the list of file basenames written."""
    out_dir.mkdir(parents=True, exist_ok=True)
    extracted: list[str] = []
    primary_lc = primary.lower()
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        for member in zf.infolist():
            name = Path(member.filename).name.lower()
            if not name:
                continue
            keep = name == primary_lc or (also_dlls and name.endswith(".dll"))
            if not keep:
                continue
            target = out_dir / Path(member.filename).name
            with zf.open(member) as src, open(target, "wb") as dst:
                shutil.copyfileobj(src, dst)
            extracted.append(target.name)
    return extracted


def _populate_voice_catalog() -> dict[str, dict]:
    """Resolve sha256 + size for every catalog entry from Hugging Face."""
    catalog: dict[str, dict] = {"stt": {}, "tts": {}}

    for w in WHISPER_MODELS:
        sha, size = _hf_lookup("ggerganov/whisper.cpp", w["filename"])
        catalog["stt"][w["model_id"]] = {
            "filename":            w["filename"],
            "repo":                "ggerganov/whisper.cpp",
            "expected_sha256":     sha,
            "expected_size_bytes": size,
        }

    for v in PIPER_VOICES:
        # Piper voices live on rhasspy/piper-voices; the HF tree API takes a
        # nested path so we look up each file individually.
        model_path = f"{v['hf_subpath']}/{v['model_filename']}"
        config_path = f"{v['hf_subpath']}/{v['config_filename']}"
        try:
            model_sha, model_size = _hf_lookup("rhasspy/piper-voices", model_path)
        except RuntimeError as exc:
            print(f"[fetch_bundled_assets] piper voice lookup failed: {exc}",
                  file=sys.stderr)
            raise
        try:
            cfg_sha, cfg_size = _hf_lookup("rhasspy/piper-voices", config_path)
        except RuntimeError as exc:
            print(f"[fetch_bundled_assets] piper config lookup failed: {exc}",
                  file=sys.stderr)
            raise

        catalog["tts"][v["voice_id"]] = {
            "repo":                "rhasspy/piper-voices",
            "hf_subpath":          v["hf_subpath"],
            "model_filename":      v["model_filename"],
            "config_filename":     v["config_filename"],
            "model_sha256":        model_sha,
            "model_size_bytes":    model_size,
            "config_sha256":       cfg_sha,
            "config_size_bytes":   cfg_size,
        }
    return catalog


def _fetch_voice_binaries() -> dict[str, str]:
    """Download the latest Whisper.cpp + Piper Windows zips and stage their
    binaries under branding/sidecar-bundle/. Returns a {component: tag} map
    that's mixed into the voice catalog as _meta."""
    meta: dict[str, str] = {}

    print("[fetch_bundled_assets] resolving latest whisper.cpp release…")
    tag, zip_url = _resolve_voice_zip_url(WHISPER_RELEASES_API, name_filter="bin")
    print(f"[fetch_bundled_assets] whisper.cpp tag {tag}, asset {zip_url}")
    zip_bytes = _http_get(zip_url, accept="application/octet-stream")
    if WHISPER_DIR.exists():
        shutil.rmtree(WHISPER_DIR)
    extracted = _extract_named(zip_bytes, WHISPER_DIR, primary="whisper-cli.exe")
    print(f"[fetch_bundled_assets] extracted {len(extracted)} files into {WHISPER_DIR}")
    if "whisper-cli.exe" not in extracted:
        # Some upstream zips name the CLI 'main.exe' for back-compat; rename.
        cand = WHISPER_DIR / "main.exe"
        if cand.exists():
            cand.rename(WHISPER_DIR / "whisper-cli.exe")
        else:
            raise RuntimeError(
                f"whisper-cli.exe not found in zip (got: {extracted})"
            )
    meta["whisper_release_tag"] = tag

    print("[fetch_bundled_assets] resolving latest piper release…")
    tag, zip_url = _resolve_voice_zip_url(PIPER_RELEASES_API, name_filter="windows")
    print(f"[fetch_bundled_assets] piper tag {tag}, asset {zip_url}")
    zip_bytes = _http_get(zip_url, accept="application/octet-stream")
    if PIPER_DIR.exists():
        shutil.rmtree(PIPER_DIR)
    extracted = _extract_named(zip_bytes, PIPER_DIR, primary="piper.exe")
    print(f"[fetch_bundled_assets] extracted {len(extracted)} files into {PIPER_DIR}")
    if "piper.exe" not in extracted:
        raise RuntimeError(
            f"piper.exe not found in zip (got: {extracted})"
        )
    meta["piper_release_tag"] = tag
    return meta


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

    # ── PR 17: voice binaries + catalog ──────────────────────────────────
    print("[fetch_bundled_assets] fetching voice binaries (Whisper + Piper)…")
    try:
        voice_meta = _fetch_voice_binaries()
    except Exception as exc:
        print(f"[fetch_bundled_assets] voice binary fetch failed: {exc}",
              file=sys.stderr)
        return 1

    print("[fetch_bundled_assets] resolving HF metadata for voice assets…")
    try:
        voice_catalog = _populate_voice_catalog()
    except Exception as exc:
        print(f"[fetch_bundled_assets] voice metadata lookup failed: {exc}",
              file=sys.stderr)
        return 1

    voice_catalog["_meta"] = voice_meta
    VOICE_CATALOG_PATH.write_text(
        json.dumps(voice_catalog, indent=2), encoding="utf-8",
    )
    print(f"[fetch_bundled_assets] wrote voice catalog → {VOICE_CATALOG_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
