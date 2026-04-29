"""Determines whether the first-run wizard should be shown."""

from core.settings import Settings


def needs_first_run(settings: Settings) -> bool:
    return not settings.get("first_run_complete", False)
