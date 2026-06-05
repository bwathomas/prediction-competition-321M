"""Tests for the run_bg / poll background-execution harness (Plan 4 §B / Task 6).

Pure stdlib — runs locally. The harness exists because the Colab MCP ``run_code_cell`` is
synchronous; long derivation must run in a daemon thread that writes progress/result to a
JSON status file the kernel can poll without blocking.
"""
import json

from aide.features.colab_runtime import poll, run_bg


def test_run_bg_captures_result(tmp_path):
    job = run_bg("ok", lambda progress: {"answer": 42}, root=tmp_path)
    job.join(timeout=5)
    st = poll("ok", root=tmp_path)
    assert st["status"] == "done"
    assert st["result"] == {"answer": 42}
    assert st["progress"] == 1.0


def test_progress_updates_are_observable(tmp_path):
    seen = {}

    def work(progress):
        progress("half", 0.5, shards=3)
        seen["mid"] = json.loads((tmp_path / "p.json").read_text())
        return "fin"

    run_bg("p", work, root=tmp_path).join(timeout=5)
    assert seen["mid"]["status"] == "running"
    assert seen["mid"]["progress"] == 0.5
    assert seen["mid"]["shards"] == 3
    assert poll("p", root=tmp_path)["status"] == "done"


def test_error_is_recorded_with_traceback(tmp_path):
    def boom(progress):
        raise ValueError("kaboom")

    run_bg("bad", boom, root=tmp_path).join(timeout=5)
    st = poll("bad", root=tmp_path)
    assert st["status"] == "error"
    assert "kaboom" in st["error"]
    assert "ValueError" in st["traceback"]


def test_poll_absent_job(tmp_path):
    assert poll("nope", root=tmp_path)["status"] == "absent"


def test_status_file_is_valid_json_during_run(tmp_path):
    # the file must always be parseable (atomic writes) even mid-run
    run_bg("j", lambda progress: progress("x", 0.1) or "done", root=tmp_path).join(timeout=5)
    json.loads((tmp_path / "j.json").read_text())  # raises if corrupt
