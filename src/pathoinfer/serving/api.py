"""Slide job API: submit a slide, get a job id, poll status, fetch results; runs the engine in
workers. Lands in v0.3.
"""

from typing import Any

from pathoinfer._extras import extra, require


def create_app() -> Any:
    require("fastapi", extra("serve"))
    raise NotImplementedError("lands in v0.3")
