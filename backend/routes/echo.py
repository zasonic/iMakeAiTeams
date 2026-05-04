"""POST /api/echo — proof-of-life endpoint that reverses the input text."""

from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class EchoIn(BaseModel):
    text: str


class EchoOut(BaseModel):
    text: str
    reversed: str


@router.post("/echo", response_model=EchoOut)
async def echo(body: EchoIn) -> EchoOut:
    return EchoOut(text=body.text, reversed=body.text[::-1])
