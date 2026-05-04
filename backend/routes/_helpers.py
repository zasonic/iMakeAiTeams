"""Helpers shared across route modules."""

from __future__ import annotations

from fastapi import Request

from core.api import API


def get_api(request: Request) -> API:
    """Resolve the shared API facade from the FastAPI application state."""
    container = request.app.state.container
    return container.api
