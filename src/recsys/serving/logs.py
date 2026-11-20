"""JSON logs with a per-request id, so one request's lines can be found across workers.

Every line is a single JSON object: {"ts", "level", "logger", "msg", "request_id", "pid",
...extra}.
`request_id` comes from the X-Request-ID header when a caller (gateway, load balancer)
sets one, otherwise it is generated; it is echoed back in the response header.
"""

from __future__ import annotations

import contextvars
import json
import logging
import time
from typing import Any

request_id_var: contextvars.ContextVar[str] = contextvars.ContextVar("request_id", default="-")

_RESERVED = set(logging.LogRecord("", 0, "", 0, "", (), None).__dict__) | {"message"}


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created))
            + f".{int(record.msecs):03d}Z",
            "level": record.levelname,
            "logger": record.name,
            "msg": record.getMessage(),
            "request_id": request_id_var.get(),
            "pid": record.process,  # which worker: load balance across workers shows up here
        }
        # structured fields passed as logger.info("...", extra={...})
        payload |= {k: v for k, v in record.__dict__.items() if k not in _RESERVED}
        if record.exc_info:
            payload["exc"] = self.formatException(record.exc_info)
        return json.dumps(payload, default=str)


def configure_logging(level: str = "INFO", fmt: str = "json") -> None:
    handler = logging.StreamHandler()
    handler.setFormatter(
        JsonFormatter()
        if fmt == "json"
        else logging.Formatter("%(asctime)s %(levelname)s %(name)s %(message)s")
    )
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level)
