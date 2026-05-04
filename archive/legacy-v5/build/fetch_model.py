"""
Pre-download the all-MiniLM-L6-v2 sentence-transformers model into
build/models/all-MiniLM-L6-v2/ so PyInstaller can bundle it as data.

Idempotent — no-op if the target already has config.json. Failure is
fatal: we do not ship a build without the model.

Run before pyinstaller:
    python build/fetch_model.py
"""

from __future__ import annotations

import sys
from pathlib import Path

MODEL_NAME = "all-MiniLM-L6-v2"
TARGET = Path(__file__).resolve().parent / "models" / MODEL_NAME


def main() -> int:
    if (TARGET / "config.json").exists():
        print(f"[fetch_model] already present: {TARGET}")
        return 0

    TARGET.parent.mkdir(parents=True, exist_ok=True)

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print(
            "[fetch_model] sentence-transformers is not installed. "
            "Install it first: pip install -r app/requirements-extensions.txt",
            file=sys.stderr,
        )
        return 2

    print(f"[fetch_model] downloading {MODEL_NAME} -> {TARGET}")
    model = SentenceTransformer(MODEL_NAME)
    model.save(str(TARGET))

    if not (TARGET / "config.json").exists():
        print(
            f"[fetch_model] save completed but {TARGET}/config.json is missing",
            file=sys.stderr,
        )
        return 3

    print(f"[fetch_model] ok: {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
