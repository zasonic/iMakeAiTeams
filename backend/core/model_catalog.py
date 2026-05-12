"""
core/model_catalog.py — Single source of truth for Claude model metadata.

Before this module landed, the model list was duplicated in three places:
``settings.SETTINGS_DEFAULTS["claude_model"]`` (default id),
``chat_orchestrator._DEFAULT_MODEL_PRICES`` (per-family prices), and
``desktop-ui/components/SettingsPanel.tsx`` (free-text input where the
user typed the id by hand). Bug 12 — the savings calculation hardcoding
the Sonnet price — was one symptom of that drift.

The catalog is loaded from ``backend/config/models.json`` and cached
in-process. ``prices_for_model()`` is the single price lookup used by
the orchestrator; it prefers an exact id match and falls back to the
deterministic family substring search (Bug 12 fix) when the id is not
in the catalog. ``get_catalog()`` returns the typed list for the
``GET /api/models/catalog`` route the renderer Settings dropdown reads.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

log = logging.getLogger("iMakeAiTeams.model_catalog")

# Path resolution: ``backend/config/models.json`` relative to this file,
# regardless of where the sidecar was launched from. Tests can override the
# path via ``set_catalog_path_for_testing()``.
_DEFAULT_PATH = Path(__file__).resolve().parent.parent / "config" / "models.json"

_lock = threading.Lock()
_cached: "Optional[Catalog]" = None
_cached_path: Optional[Path] = None


@dataclass(frozen=True)
class ModelEntry:
    """One row in the catalog. Mirrors the JSON schema 1:1."""
    id:                       str
    family:                   str       # "opus" | "sonnet" | "haiku"
    display_name:             str
    input_price_per_mtok:     float
    output_price_per_mtok:    float
    context_window_tokens:    int
    vision:                   bool
    available_via:            tuple[str, ...]

    def to_dict(self) -> dict:
        return {
            "id":                       self.id,
            "family":                   self.family,
            "display_name":             self.display_name,
            "input_price_per_mtok":     self.input_price_per_mtok,
            "output_price_per_mtok":    self.output_price_per_mtok,
            "context_window_tokens":    self.context_window_tokens,
            "vision":                   self.vision,
            "available_via":            list(self.available_via),
        }


@dataclass(frozen=True)
class Catalog:
    """Parsed snapshot of models.json."""
    models:                   tuple[ModelEntry, ...]
    default_claude_id:        str
    family_fallback_prices:   dict[str, tuple[float, float]]

    def find_by_id(self, model_id: str) -> Optional[ModelEntry]:
        for m in self.models:
            if m.id == model_id:
                return m
        return None

    def detect_family(self, model_id: str) -> Optional[str]:
        """Deterministic family substring search (Bug 12 fix).

        Pre-catalog ``_estimate_cost`` iterated a dict and took the first
        substring match, which depended on dict iteration order — a name
        like ``claude-haiku-with-opus-fallback`` could resolve either
        way. Pick families in fixed order opus → sonnet → haiku so the
        result is reproducible.
        """
        m = (model_id or "").lower()
        for candidate in ("opus", "sonnet", "haiku"):
            if candidate in m:
                return candidate
        return None

    def prices_for_model(
        self, model_id: str,
        user_overrides: Optional[dict[str, tuple[float, float]]] = None,
    ) -> tuple[float, float]:
        """Return ``(input_price, output_price)`` per million tokens.

        Resolution order:
          1. user_overrides by family — set via the existing
             ``settings.model_prices`` dict so the user can hot-patch
             pricing without editing the catalog.
          2. exact catalog id match.
          3. family substring fallback against
             ``family_fallback_prices``.
          4. ``(3.0, 15.0)`` — the Sonnet default; matches the
             pre-catalog behaviour so old call sites can't regress.
        """
        family = self.detect_family(model_id)
        if user_overrides and family and family in user_overrides:
            override = user_overrides[family]
            return (float(override[0]), float(override[1]))

        entry = self.find_by_id(model_id)
        if entry is not None:
            return (entry.input_price_per_mtok, entry.output_price_per_mtok)

        if family and family in self.family_fallback_prices:
            return self.family_fallback_prices[family]

        return (3.0, 15.0)


def _parse_catalog(raw: dict, source: Path) -> Catalog:
    models_raw = raw.get("models")
    if not isinstance(models_raw, list):
        raise ValueError(
            f"{source}: top-level 'models' must be a list, got {type(models_raw)!r}"
        )

    models: list[ModelEntry] = []
    for i, row in enumerate(models_raw):
        if not isinstance(row, dict):
            raise ValueError(f"{source}: models[{i}] is not a dict")
        try:
            models.append(ModelEntry(
                id=                       str(row["id"]),
                family=                   str(row["family"]),
                display_name=             str(row["display_name"]),
                input_price_per_mtok=     float(row["input_price_per_mtok"]),
                output_price_per_mtok=    float(row["output_price_per_mtok"]),
                context_window_tokens=    int(row["context_window_tokens"]),
                vision=                   bool(row["vision"]),
                available_via=            tuple(str(v) for v in (row.get("available_via") or [])),
            ))
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{source}: models[{i}] invalid: {exc}") from exc

    default_id = str(raw.get("default_claude_id", ""))
    if default_id and not any(m.id == default_id for m in models):
        log.warning(
            "%s: default_claude_id %r not in models list — falling back to first entry",
            source, default_id,
        )
        default_id = models[0].id if models else ""

    family_fallback: dict[str, tuple[float, float]] = {}
    fb_raw = raw.get("family_fallback_prices") or {}
    for family, prices in fb_raw.items():
        if family.startswith("_") or not isinstance(prices, dict):
            continue
        try:
            family_fallback[family] = (
                float(prices["input_price_per_mtok"]),
                float(prices["output_price_per_mtok"]),
            )
        except (KeyError, TypeError, ValueError):
            log.warning(
                "%s: family_fallback_prices[%r] missing input/output prices — skipped",
                source, family,
            )

    return Catalog(
        models=tuple(models),
        default_claude_id=default_id,
        family_fallback_prices=family_fallback,
    )


def get_catalog(force_reload: bool = False) -> Catalog:
    """Return the parsed catalog. Cached after the first call.

    ``force_reload=True`` re-reads the JSON file — used by tests and the
    eventual /api/models/catalog/reload admin route if we ever need it.
    """
    global _cached
    with _lock:
        if _cached is not None and not force_reload:
            return _cached
        path = _cached_path or _DEFAULT_PATH
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Model catalog not found at {path}. The orchestrator's price "
                "math depends on it — the file must ship with the installer."
            )
        except json.JSONDecodeError as exc:
            raise ValueError(f"{path}: invalid JSON: {exc}") from exc
        _cached = _parse_catalog(raw, path)
        return _cached


def set_catalog_path_for_testing(path: Optional[Path]) -> None:
    """Override the catalog path and clear the cache. Test-only helper."""
    global _cached, _cached_path
    with _lock:
        _cached_path = path
        _cached = None
