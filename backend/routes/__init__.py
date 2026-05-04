"""FastAPI route modules for the iMakeAiTeams sidecar.

Each module wraps a domain sub-API from core/api/* and exposes the methods
as POST/GET endpoints. All routes are mounted under /api/<group> by
backend/server.py and gated by the Bearer auth middleware.
"""
