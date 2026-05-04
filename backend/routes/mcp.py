"""MCP routes — wrap core/api/mcp.MCPAPI."""

from __future__ import annotations

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ._helpers import get_api

router = APIRouter()


class IngestFolderIn(BaseModel):
    folder_path: str
    overwrite: bool = False


class EnabledIn(BaseModel):
    server_id: str
    enabled: bool


class SecretIn(BaseModel):
    server_id: str
    key: str
    value: str = ""


@router.get("/servers")
async def list_servers(request: Request) -> dict:
    return get_api(request).list_mcp_servers()


@router.post("/install")
async def install(body: IngestFolderIn, request: Request) -> dict:
    return get_api(request).pick_mcp_server_folder(
        folder_path=body.folder_path, overwrite=body.overwrite,
    )


@router.post("/remove/{server_id}")
async def remove(server_id: str, request: Request) -> dict:
    return get_api(request).remove_mcp_server(server_id)


@router.post("/enabled")
async def set_enabled(body: EnabledIn, request: Request) -> dict:
    return get_api(request).set_mcp_server_enabled(body.server_id, body.enabled)


@router.post("/secrets/set")
async def set_secret(body: SecretIn, request: Request) -> dict:
    return get_api(request).set_mcp_secret(body.server_id, body.key, body.value)


@router.post("/secrets/clear")
async def clear_secret(body: SecretIn, request: Request) -> dict:
    return get_api(request).clear_mcp_secret(body.server_id, body.key)


@router.post("/refresh")
async def refresh(request: Request) -> dict:
    return get_api(request).refresh_mcp_registry()
