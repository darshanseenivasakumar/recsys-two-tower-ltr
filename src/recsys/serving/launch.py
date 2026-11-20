"""Run N uvicorn workers, each accepting on its OWN socket (SO_REUSEPORT), same port.

    python -m recsys.serving.launch --app recsys.serving.api:app --port 8000 --workers 4

Why not `uvicorn --workers N`: that binds one listening socket in the parent and every
worker accept()s from it. Linux does not hand those accepts out evenly (a worker already
awake in its event loop tends to win), and with keep-alive clients a connection stays on
the worker that accepted it for its whole life, so the imbalance persists. Under Locust we
measured one worker at its 1-core GIL limit while the other three ran 0.27-0.47 cores.

With SO_REUSEPORT every worker binds its own socket on the port and the kernel picks the
socket for each NEW connection by a hash of its address 4-tuple: an even spread in
expectation, decided by the kernel instead of by which worker woke first. It balances
connections, not requests: a few very busy keep-alive clients can still land together.

The parent only supervises: it forwards SIGTERM/SIGINT for uvicorn's graceful shutdown and
restarts a worker that dies, unless workers keep dying right after starting (a broken
bundle would otherwise restart forever). Linux / macOS only (Windows has no SO_REUSEPORT).
"""

from __future__ import annotations

import argparse
import logging
import multiprocessing as mp
import signal
import socket
import sys
import time
from multiprocessing.process import BaseProcess
from types import FrameType

logger = logging.getLogger("recsys.launch")

FAST_EXIT_S = 15.0  # a worker that dies sooner than this after starting counts as a crash loop
MAX_FAST_EXITS = 3


def reuseport_socket(host: str, port: int, backlog: int) -> socket.socket:
    if not hasattr(socket, "SO_REUSEPORT"):
        raise SystemExit("SO_REUSEPORT is not available on this platform (Linux/macOS only)")
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEPORT, 1)
    sock.bind((host, port))
    sock.listen(backlog)
    return sock


def _worker(app: str, host: str, port: int, backlog: int) -> None:
    import uvicorn

    sock = reuseport_socket(host, port, backlog)  # bound in the child: one socket per worker
    config = uvicorn.Config(app, host=host, port=port, backlog=backlog, access_log=False)
    uvicorn.Server(config).run(sockets=[sock])


def supervise(app: str, host: str, port: int, workers: int, backlog: int = 2048) -> int:
    ctx = mp.get_context("spawn")
    procs: dict[int, tuple[BaseProcess, float]] = {}
    stopping = False
    fast_exits = 0

    def start(slot: int) -> None:
        p = ctx.Process(target=_worker, args=(app, host, port, backlog), name=f"worker-{slot}")
        p.start()
        procs[slot] = (p, time.monotonic())
        logger.info("worker started", extra={"slot": slot, "worker_pid": p.pid})

    def stop(signum: int, _frame: FrameType | None) -> None:
        nonlocal stopping
        stopping = True
        for p, _ in procs.values():
            if p.is_alive():
                p.terminate()  # SIGTERM: uvicorn finishes in-flight requests, then exits

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    for slot in range(workers):
        start(slot)

    while not stopping:
        time.sleep(0.5)
        for slot, (p, started) in list(procs.items()):
            if p.is_alive() or stopping:
                continue
            lived = time.monotonic() - started
            logger.error(
                "worker exited", extra={"slot": slot, "exitcode": p.exitcode, "lived_s": lived}
            )
            fast_exits = fast_exits + 1 if lived < FAST_EXIT_S else 0
            if fast_exits >= MAX_FAST_EXITS:
                logger.error("workers keep dying on startup; giving up")
                stop(signal.SIGTERM, None)
                break
            start(slot)

    for p, _ in procs.values():
        p.join(timeout=30)
        if p.is_alive():
            p.kill()
    return 1 if fast_exits >= MAX_FAST_EXITS else 0


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--app", default="recsys.serving.api:app")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--backlog", type=int, default=2048)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sys.exit(supervise(args.app, args.host, args.port, args.workers, args.backlog))


if __name__ == "__main__":
    main()
