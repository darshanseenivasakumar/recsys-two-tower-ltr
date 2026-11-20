"""HTTP API.

    RECSYS_BUNDLE_DIR=artifacts/serving/ml-1m REDIS_URL=redis://localhost:6379/0 \
        uvicorn recsys.serving.api:app --port 8000

GET  /recommend?user_id=..&k=10&diversify=true   top-k movies for a user
POST /interaction {user_id, item_id, rating, timestamp?}   record an event (affects the
                                                           next /recommend immediately)
GET  /health                                       liveness + bundle/Redis versions
GET  /monitoring?last=5000                         traffic health over recent requests:
                                                   cold-start rate, latency, exposure, drift

Environment:
  RECSYS_BUNDLE_DIR     serving bundle directory (required)
  REDIS_URL             default redis://localhost:6379/0
  RECSYS_CLOCK          "snapshot" (default; time features as of the bundle cutoff, as
                        evaluated offline) or "wall" (a live system's real clock)
  FEATURE_SAMPLE_RATE   share of personalized requests whose candidate features are logged
                        for drift monitoring (default 0.05)
  RANKER_NUM_THREADS    LightGBM threads per request (default 0 = all cores; set it so
                        workers x threads <= cores when running several workers)
  RANKER_BACKEND        "lightgbm" (default) or "lleaves": the same trees compiled to native
                        code, ~5x faster, served only if bit-identical to LightGBM on the
                        bundle's reference rows (else LightGBM; /health says which)
  RANKER_CACHE_DIR      where compiled rankers are cached (default <bundle>/.ranker-cache;
                        fill it before starting workers: python -m recsys.serving.compiled)
  THREADPOOL_TOKENS     concurrent requests per worker (default 0 = anyio's 40)
  LOG_FORMAT / LOG_LEVEL   "json" (default) or "text"; INFO

Handlers are plain `def`: the work is CPU-bound numpy/LightGBM, so FastAPI runs them in
its threadpool; throughput comes from running several uvicorn worker processes.
"""

from __future__ import annotations

import logging
import os
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

import anyio.to_thread
import numpy as np
import pandas as pd
import torch
from fastapi import FastAPI, HTTPException, Query, Request, Response
from pydantic import BaseModel, Field

from recsys.monitoring.reference import REFERENCE_FILE
from recsys.monitoring.report import summarize_traffic
from recsys.serving.bundle import ServingBundle
from recsys.serving.logs import configure_logging, request_id_var
from recsys.serving.service import (
    MAX_K,
    RecommenderService,
    UnknownItemError,
    UnknownUserError,
)
from recsys.serving.store import EventLog, InteractionStore

logger = logging.getLogger("recsys.api")


class InteractionIn(BaseModel):
    user_id: str
    item_id: str
    rating: float = Field(ge=1, le=5)
    timestamp: int | None = Field(default=None, description="unix seconds; default: the clock")


def _json_safe(value: Any) -> Any:
    """NaN/inf -> None (JSON has no NaN; e.g. a feature that is all-missing has no mean)."""
    if isinstance(value, dict):
        return {k: _json_safe(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_json_safe(v) for v in value]
    if isinstance(value, float | np.floating):
        return float(value) if np.isfinite(value) else None
    if isinstance(value, np.integer):
        return int(value)
    return value


def _service_from_env() -> tuple[RecommenderService, pd.DataFrame | None]:
    bundle_dir = Path(os.environ["RECSYS_BUNDLE_DIR"])
    bundle = ServingBundle.load(bundle_dir)
    store = InteractionStore.from_url(os.environ.get("REDIS_URL", "redis://localhost:6379/0"))
    seeded = store.bundle_version()
    if seeded != bundle.manifest["version"]:
        logger.warning(
            "Redis seeded for a different bundle",
            extra={"redis_bundle": seeded, "serving_bundle": bundle.manifest["version"]},
        )
    clock_mode = os.environ.get("RECSYS_CLOCK", "snapshot")
    if clock_mode not in ("snapshot", "wall"):
        raise ValueError("RECSYS_CLOCK must be 'snapshot' or 'wall'")
    service = RecommenderService(
        bundle,
        store,
        clock=time.time if clock_mode == "wall" else None,
        event_log=EventLog(store.client),
        feature_sample_rate=float(os.environ.get("FEATURE_SAMPLE_RATE", "0.05")),
        ranker_threads=int(os.environ.get("RANKER_NUM_THREADS", "0")),
        ranker_backend=os.environ.get("RANKER_BACKEND", "lightgbm"),
        ranker_cache_dir=Path(d) if (d := os.environ.get("RANKER_CACHE_DIR")) else None,
    )
    ref_path = bundle_dir / REFERENCE_FILE
    reference = pd.read_parquet(ref_path) if ref_path.exists() else None
    logger.info(
        "service ready",
        extra={
            "bundle": bundle.manifest["version"],
            "clock": clock_mode,
            "ranker_backend": service.ranker_backend,
            "drift_reference_rows": None if reference is None else len(reference),
        },
    )
    return service, reference


def create_app(
    service: RecommenderService | None = None, reference: pd.DataFrame | None = None
) -> FastAPI:
    """App factory. Tests pass a ready service; production builds one from env vars."""

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        torch.set_num_threads(1)  # tiny per-request tensors; avoid thread oversubscription
        # Requests in one worker share one GIL; with anyio's default 40 threads they
        # interleave, and each one's latency absorbs its neighbours' Python time.
        if tokens := int(os.environ.get("THREADPOOL_TOKENS", "0")):
            anyio.to_thread.current_default_thread_limiter().total_tokens = tokens
        if service is not None:
            app.state.service, app.state.reference = service, reference
        else:
            configure_logging(
                os.environ.get("LOG_LEVEL", "INFO"), os.environ.get("LOG_FORMAT", "json")
            )
            app.state.service, app.state.reference = _service_from_env()
        yield

    app = FastAPI(title="MovieLens recommender", lifespan=lifespan)

    @app.middleware("http")
    async def access_log(
        request: Request, call_next: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        rid = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        token = request_id_var.set(rid)  # every log line in this request carries it
        start = time.perf_counter()
        try:
            response = await call_next(request)
            response.headers["X-Request-ID"] = rid
            logging.getLogger("recsys.access").info(
                "request",
                extra={
                    "method": request.method,
                    "path": request.url.path,
                    "status": response.status_code,
                    "duration_ms": round((time.perf_counter() - start) * 1e3, 2),
                },
            )
            return response
        finally:
            request_id_var.reset(token)

    @app.get("/health")
    def health(request: Request) -> dict[str, Any]:
        svc: RecommenderService = request.app.state.service
        return {
            "status": "ok",
            "bundle_version": svc.bundle.manifest["version"],
            "redis_bundle_version": svc.store.bundle_version(),
            "redis": svc.store.ping(),
            "ranker_backend": svc.ranker_backend,
        }

    @app.get("/recommend")
    def recommend(
        request: Request,
        user_id: str,
        k: int = Query(10, ge=1, le=MAX_K),
        diversify: bool = True,
    ) -> dict[str, Any]:
        svc: RecommenderService = request.app.state.service
        rec = svc.recommend(user_id, k, diversify)
        logger.info(
            "recommend",
            extra={
                "user_id": rec.user_id,
                "strategy": rec.strategy,
                "n_candidates": rec.n_candidates,
                "timings_ms": {k: round(v, 2) for k, v in rec.timings_ms.items()},
            },
        )
        return {
            "user_id": rec.user_id,
            "strategy": rec.strategy,
            "items": rec.items,
            "n_candidates": rec.n_candidates,
            "timings_ms": {k: round(v, 2) for k, v in rec.timings_ms.items()},
        }

    @app.post("/interaction", status_code=202)
    def interaction(request: Request, event: InteractionIn) -> dict[str, Any]:
        svc: RecommenderService = request.app.state.service
        try:
            n = svc.record(event.user_id, event.item_id, event.rating, event.timestamp)
        except UnknownUserError as e:
            raise HTTPException(404, f"unknown user_id {event.user_id}") from e
        except UnknownItemError as e:
            raise HTTPException(404, f"unknown item_id {event.item_id}") from e
        return {"user_id": event.user_id, "n_interactions": n}

    @app.get("/monitoring")
    def monitoring(request: Request, last: int = Query(5000, ge=1, le=200_000)) -> dict[str, Any]:
        svc: RecommenderService = request.app.state.service
        if svc.event_log is None:
            raise HTTPException(503, "event logging is disabled")
        reference = request.app.state.reference
        summary = summarize_traffic(
            svc.event_log.recommendations(last),
            svc.event_log.features(last) if reference is not None else pd.DataFrame(),
            reference if reference is not None else pd.DataFrame(),
            svc.feature_names,
            np.expm1(svc.bundle.item_table.log_pop),
            float(svc.bundle.manifest["cutoff_unix"]),
        )
        return _json_safe({"bundle_version": svc.bundle.manifest["version"], **summary})

    return app


def __getattr__(name: str) -> Any:
    # `uvicorn recsys.serving.api:app` builds the production app lazily, so importing this
    # module (e.g. in tests) never needs env vars, a bundle on disk, or Redis.
    if name == "app":
        return create_app()
    raise AttributeError(name)
