"""Minimal ASGI app for tests/test_launch.py: answers every request with its worker's pid."""

from __future__ import annotations

import os
from typing import Any

if os.environ.get("PID_APP_CRASH") == "1":
    raise RuntimeError("simulated broken bundle: the worker dies while importing the app")


async def app(scope: dict[str, Any], receive: Any, send: Any) -> None:
    if scope["type"] == "lifespan":
        while True:
            message = await receive()
            if message["type"] == "lifespan.startup":
                await send({"type": "lifespan.startup.complete"})
            elif message["type"] == "lifespan.shutdown":
                await send({"type": "lifespan.shutdown.complete"})
                return
    body = str(os.getpid()).encode()
    await send(
        {
            "type": "http.response.start",
            "status": 200,
            "headers": [(b"content-length", str(len(body)).encode())],
        }
    )
    await send({"type": "http.response.body", "body": body})
