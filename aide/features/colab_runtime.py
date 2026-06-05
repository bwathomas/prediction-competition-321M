"""Background-execution harness for Colab — ``run_bg`` / ``poll`` (Plan 4 §B, Task 6).

The Colab MCP ``run_code_cell`` is SYNCHRONOUS: a multi-minute derivation would freeze the
session. ``run_bg(name, fn)`` runs ``fn`` in a daemon thread that writes progress + result
to ``{root}/{name}.json`` and returns instantly; a tiny ``poll(name)`` cell reads that file
and returns at once. The status file is written atomically (``.tmp`` → ``replace``) so a
poll mid-run never reads a half-written file.

``fn`` receives one argument — a ``progress(message, frac=None, **extra)`` callback — and
returns a JSON-serializable result (e.g. a dict of shard counts). On exception the status
becomes ``"error"`` with the traceback, so a failed background job is never silent.
"""
from __future__ import annotations

import json
import threading
import traceback
from pathlib import Path


def _status_path(root, name: str) -> Path:
    return Path(root) / f"{name}.json"


def _atomic_write(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload), encoding="utf-8")
    tmp.replace(path)


class Job:
    """Handle for a launched background job (mostly for tests / explicit waits)."""

    def __init__(self, name: str, thread: threading.Thread, path: Path):
        self.name = name
        self.thread = thread
        self.path = path

    def join(self, timeout=None) -> "Job":
        self.thread.join(timeout)
        return self


def run_bg(name: str, fn, *, root="/content") -> Job:
    """Launch ``fn(progress)`` in a daemon thread; return immediately with a ``Job``."""
    path = _status_path(root, name)
    _atomic_write(path, {"name": name, "status": "running", "progress": 0.0,
                         "message": "started"})

    def progress(message: str = "", frac=None, **extra) -> None:
        payload: dict = {"name": name, "status": "running", "message": message}
        if frac is not None:
            payload["progress"] = float(frac)
        payload.update(extra)
        _atomic_write(path, payload)

    def runner() -> None:
        try:
            result = fn(progress)
            _atomic_write(path, {"name": name, "status": "done", "progress": 1.0,
                                 "result": result})
        except Exception as exc:  # noqa: BLE001 - any failure must surface in the file
            _atomic_write(path, {"name": name, "status": "error", "error": str(exc),
                                 "traceback": traceback.format_exc()})

    thread = threading.Thread(target=runner, name=f"run_bg:{name}", daemon=True)
    thread.start()
    return Job(name, thread, path)


def poll(name: str, *, root="/content") -> dict:
    """Read the status file for ``name``; ``{"status": "absent"}`` if never launched."""
    path = _status_path(root, name)
    if not path.exists():
        return {"name": name, "status": "absent"}
    return json.loads(path.read_text(encoding="utf-8"))
