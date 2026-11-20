"""The SO_REUSEPORT launcher: connections spread over every worker, clean shutdown, and no
endless restarts of a worker that can't start. Linux/macOS only (skipped on Windows); the
serving image runs it (see reports/serving_ml-1m.md)."""

from __future__ import annotations

import http.client
import os
import signal
import socket
import subprocess
import sys
import time
from collections import Counter
from pathlib import Path

import pytest

pytestmark = pytest.mark.skipif(
    not hasattr(socket, "SO_REUSEPORT"), reason="SO_REUSEPORT needs Linux/macOS"
)
ROOT = Path(__file__).resolve().parents[1]


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def _launch(port: int, workers: int, **env: str) -> subprocess.Popen[bytes]:
    cmd = [
        sys.executable,
        "-m",
        "recsys.serving.launch",
        "--app",
        "tests._pid_app:app",
        "--host",
        "127.0.0.1",
        "--port",
        str(port),
        "--workers",
        str(workers),
    ]
    return subprocess.Popen(
        cmd,
        cwd=ROOT,
        env={**os.environ, **env},
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def _get_pid(port: int) -> int:
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)  # new TCP connection
    try:
        conn.request("GET", "/", headers={"Connection": "close"})
        return int(conn.getresponse().read())
    finally:
        conn.close()


def test_new_connections_spread_over_every_worker_and_sigterm_is_clean() -> None:
    port, workers = _free_port(), 4
    proc = _launch(port, workers)
    try:
        seen: set[int] = set()
        deadline = time.monotonic() + 30
        while len(seen) < workers and time.monotonic() < deadline:  # all workers bound
            try:
                seen.add(_get_pid(port))
            except OSError:
                time.sleep(0.1)
        counts = Counter(_get_pid(port) for _ in range(400))
        assert len(counts) == workers, counts
        # kernel hash of each connection's 4-tuple: ~100 each (binomial sd ~9)
        assert max(counts.values()) < 0.4 * 400, counts
        proc.send_signal(signal.SIGTERM)
        assert proc.wait(timeout=30) == 0
    finally:
        if proc.poll() is None:
            proc.kill()


def test_a_worker_that_cannot_start_is_not_restarted_forever() -> None:
    proc = _launch(_free_port(), 2, PID_APP_CRASH="1")
    try:
        assert proc.wait(timeout=60) == 1
    finally:
        if proc.poll() is None:
            proc.kill()
