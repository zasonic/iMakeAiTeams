"""GET /api/models/catalog — the renderer's view of the Claude model lineup.

Reads ``backend/config/models.json`` through ``core.model_catalog`` and
returns the typed list the Settings dropdown renders. The orchestrator's
price math uses the same catalog, so the renderer can never offer a
model whose cost the backend cannot compute.
"""

from __future__ import annotations

from fastapi import APIRouter

from core.model_catalog import get_catalog

router = APIRouter()


@router.get("/catalog")
async def catalog() -> dict:
    """Return the full Claude model catalog.

    Shape mirrors ``models.json`` so the renderer doesn't need a second
    schema definition:

      {
        "default_claude_id": "claude-sonnet-4-6",
        "models": [
          {id, family, display_name, input_price_per_mtok,
           output_price_per_mtok, context_window_tokens, vision,
           available_via},
          ...
        ]
      }
    """
    cat = get_catalog()
    return {
        "default_claude_id": cat.default_claude_id,
        "models": [m.to_dict() for m in cat.models],
    }
