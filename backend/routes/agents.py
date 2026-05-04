"""Agent + team CRUD routes — wrap core/api/agents.AgentsAPI."""

from __future__ import annotations

from typing import Optional

from fastapi import APIRouter, Request
from pydantic import BaseModel, Field

from ._helpers import get_api

router = APIRouter()


class AgentCreateIn(BaseModel):
    name: str
    description: str
    system_prompt: str
    model_preference: str = "auto"
    temperature: float = 0.7
    max_tokens: int = 4096


class AgentUpdateIn(BaseModel):
    agent_id: str
    fields: dict


class AgentDuplicateIn(BaseModel):
    agent_id: str
    new_name: str


class AgentTomIn(BaseModel):
    agent_name: str
    agent_domain: str
    agent_scope: str
    teammates: Optional[list] = None


class AgentProjectRootIn(BaseModel):
    path: str


class TeamCreateIn(BaseModel):
    name: str
    description: str
    coordinator_id: str


class TeamMemberIn(BaseModel):
    team_id: str
    agent_id: str
    role: str = "worker"


@router.get("")
async def list_agents(request: Request) -> list:
    return get_api(request).agent_list()


@router.get("/{agent_id}")
async def get_agent(agent_id: str, request: Request):
    return get_api(request).agent_get(agent_id)


@router.post("/create")
async def create(body: AgentCreateIn, request: Request) -> dict:
    return get_api(request).agent_create(
        body.name, body.description, body.system_prompt,
        body.model_preference, body.temperature, body.max_tokens,
    )


@router.post("/update")
async def update(body: AgentUpdateIn, request: Request) -> dict:
    return get_api(request).agent_update(body.agent_id, body.fields)


@router.post("/duplicate")
async def duplicate(body: AgentDuplicateIn, request: Request) -> dict:
    return get_api(request).agent_duplicate(body.agent_id, body.new_name)


@router.post("/delete/{agent_id}")
async def delete(agent_id: str, request: Request) -> dict:
    return get_api(request).agent_delete(agent_id)


@router.post("/generate_tom")
async def generate_tom(body: AgentTomIn, request: Request) -> dict:
    return get_api(request).agent_generate_tom(
        body.agent_name, body.agent_domain, body.agent_scope, body.teammates,
    )


@router.post("/refresh_team_tom/{team_id}")
async def refresh_team_tom(team_id: str, request: Request) -> dict:
    return get_api(request).agent_refresh_team_tom(team_id)


@router.post("/set_project_root")
async def set_project_root(body: AgentProjectRootIn, request: Request) -> dict:
    return get_api(request).agent_set_project_root(body.path)


# ── Teams ────────────────────────────────────────────────────────────────────


@router.get("/teams/all")
async def list_teams(request: Request) -> list:
    return get_api(request).team_list()


@router.get("/teams/{team_id}")
async def get_team(team_id: str, request: Request):
    return get_api(request).team_get(team_id)


@router.post("/teams/create")
async def team_create(body: TeamCreateIn, request: Request) -> dict:
    return get_api(request).team_create(body.name, body.description, body.coordinator_id)


@router.post("/teams/add_member")
async def team_add_member(body: TeamMemberIn, request: Request) -> dict:
    return get_api(request).team_add_member(body.team_id, body.agent_id, body.role)


@router.post("/teams/remove_member")
async def team_remove_member(body: TeamMemberIn, request: Request) -> dict:
    return get_api(request).team_remove_member(body.team_id, body.agent_id)


@router.post("/teams/delete/{team_id}")
async def team_delete(team_id: str, request: Request) -> dict:
    return get_api(request).team_delete(team_id)
