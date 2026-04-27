"""Settings routes — wrap core/api/settings.SettingsAPI."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Request
from pydantic import BaseModel

from ._helpers import get_api

router = APIRouter()


class KVIn(BaseModel):
    key: str
    value: Any


class FirstRunIn(BaseModel):
    start_tab: str


class VerifyKeyIn(BaseModel):
    key: str


class StudioModeIn(BaseModel):
    enabled: bool


class PricesIn(BaseModel):
    prices: dict


@router.get("")
async def get_settings(request: Request) -> dict:
    return get_api(request).get_settings()


@router.post("/save")
async def save(body: KVIn, request: Request) -> dict:
    get_api(request).save_setting(body.key, body.value)
    return {"ok": True}


@router.post("/set")
async def set_kv(body: KVIn, request: Request) -> dict:
    return get_api(request).set_setting(body.key, body.value)


@router.get("/get")
async def get_one(request: Request, key: str) -> dict:
    return get_api(request).get_setting(key)


@router.post("/complete_first_run")
async def complete_first_run(body: FirstRunIn, request: Request) -> dict:
    get_api(request).complete_first_run(body.start_tab)
    return {"ok": True}


@router.post("/verify_api_key")
async def verify_api_key(body: VerifyKeyIn, request: Request) -> dict:
    return get_api(request).verify_api_key(body.key)


@router.get("/detect_local")
async def detect_local(request: Request) -> dict:
    return get_api(request).detect_local_setup()


@router.get("/model_prices")
async def get_prices(request: Request) -> dict:
    return get_api(request).get_model_prices()


@router.post("/model_prices")
async def set_prices(body: PricesIn, request: Request) -> dict:
    return get_api(request).set_model_prices(body.prices)


@router.get("/studio_mode")
async def get_studio(request: Request) -> dict:
    return get_api(request).studio_mode_get()


@router.post("/studio_mode")
async def set_studio(body: StudioModeIn, request: Request) -> dict:
    return get_api(request).studio_mode_set(body.enabled)
