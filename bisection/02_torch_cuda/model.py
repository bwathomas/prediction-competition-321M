"""Bisection bundle 02: imports torch, reports CUDA capabilities.

No model weights are loaded. Returns 0.5 for every prediction. Logs
``cuda_available``, ``cuda_device_count``, ``cuda_device_name``,
``cuda_total_vram_gb``, ``is_bf16_supported``, ``torch_version``,
``transformers_version``, etc. via the same progress-file mechanism
as bundle 01.

If this bundle's progress shows ``cuda_available=False``, the
platform routed us to a CPU container -- nothing batched will fit
in the runtime budget. If ``is_bf16_supported=False``, we need to
re-quantize the bundle. Either is a clear diagnosis.
"""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ARTIFACTS = HERE / "artifacts"

_T0 = time.time()
_STATE: dict = {
    "bundle": "bisection_02_torch_cuda",
    "started_at": _T0,
    "events": [],
}

_PROGRESS_PATHS = [
    ARTIFACTS / "runtime_progress.json",
    HERE / "runtime_progress.json",
    Path("/tmp/runtime_progress.json"),
    Path(os.environ.get("TMPDIR", "/tmp")) / "runtime_progress.json",
]


def _write_progress(stage: str, **info) -> None:
    try:
        ev = {
            "stage": stage,
            "t_since_start_s": round(time.time() - _T0, 3),
            **info,
        }
        _STATE["events"].append(ev)
        _STATE["latest"] = ev
        try:
            print(f"[runtime-02] {stage} {info}", flush=True)
        except Exception:
            pass
        body = json.dumps(_STATE, indent=2, default=str).encode("utf-8")
        for p in _PROGRESS_PATHS:
            try:
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(body)
            except Exception:
                continue
    except Exception:
        pass


_write_progress(
    "module_init_start",
    python_version=sys.version.split()[0],
)

# Import torch + transformers and capture every fact about the GPU
# environment we can. Wrap each step so that one missing dependency
# doesn't silently swallow the rest of the report.
_torch_info: dict = {}
try:
    import torch
    _torch_info["torch_version"] = str(torch.__version__)
    _torch_info["cuda_available"] = bool(torch.cuda.is_available())
    if torch.cuda.is_available():
        _torch_info["cuda_device_count"] = int(torch.cuda.device_count())
        _torch_info["cuda_device_name"] = str(torch.cuda.get_device_name(0))
        _torch_info["cuda_total_vram_gb"] = round(
            torch.cuda.get_device_properties(0).total_memory / (1024**3), 2
        )
        try:
            free_b, total_b = torch.cuda.mem_get_info()
            _torch_info["cuda_free_vram_gb"] = round(free_b / (1024**3), 2)
        except Exception as e:
            _torch_info["cuda_mem_get_info_error"] = repr(e)
        try:
            _torch_info["is_bf16_supported"] = bool(torch.cuda.is_bf16_supported())
        except Exception as e:
            _torch_info["is_bf16_supported_error"] = repr(e)
        try:
            cap = torch.cuda.get_device_capability(0)
            _torch_info["cuda_capability"] = f"{cap[0]}.{cap[1]}"
        except Exception as e:
            _torch_info["cuda_capability_error"] = repr(e)
except Exception as e:
    _torch_info["import_torch_error"] = repr(e)

try:
    import transformers
    _torch_info["transformers_version"] = str(transformers.__version__)
except Exception as e:
    _torch_info["import_transformers_error"] = repr(e)

try:
    # Detect FA2 + SDPA support without actually loading a model.
    try:
        import flash_attn  # type: ignore
        _torch_info["flash_attn_version"] = str(getattr(flash_attn, "__version__", "unknown"))
    except Exception as e:
        _torch_info["flash_attn_missing"] = repr(e)
    try:
        from torch.nn.functional import scaled_dot_product_attention  # noqa: F401
        _torch_info["sdpa_available"] = True
    except Exception as e:
        _torch_info["sdpa_available"] = False
        _torch_info["sdpa_error"] = repr(e)
except Exception as e:
    _torch_info["attn_probe_error"] = repr(e)

_write_progress("env_probed", **_torch_info)

# Pre-fetched HF cache state -- useful for diagnosing H11 (models.txt
# pre-fetch). Just count entries; don't actually load anything.
try:
    hf_cache_root = os.environ.get("HF_HOME") or os.environ.get(
        "TRANSFORMERS_CACHE"
    ) or str(Path.home() / ".cache" / "huggingface")
    hub_dir = Path(hf_cache_root) / "hub"
    if hub_dir.exists():
        repos = [d.name for d in hub_dir.iterdir() if d.is_dir()]
        _write_progress(
            "hf_cache_probed",
            hf_home=hf_cache_root,
            hub_dir=str(hub_dir),
            repo_count=len(repos),
            repos=repos[:40],
        )
    else:
        _write_progress(
            "hf_cache_missing", hf_home=hf_cache_root, hub_dir=str(hub_dir)
        )
except Exception as e:
    _write_progress("hf_cache_probe_failed", error=repr(e))

_write_progress("module_init_complete")

_PREDICT_COUNT = 0
_FIRST_PREDICT_LOGGED = False


def predict(input: dict, labeled=None) -> float:  # noqa: A002
    global _PREDICT_COUNT, _FIRST_PREDICT_LOGGED
    _PREDICT_COUNT += 1
    if not _FIRST_PREDICT_LOGGED:
        _FIRST_PREDICT_LOGGED = True
        _write_progress(
            "predict_first_call",
            n_labeled=(len(labeled) if labeled else 0),
        )
    if _PREDICT_COUNT % 500 == 0:
        _write_progress("predict_progress", predict_count=int(_PREDICT_COUNT))
    return 0.5
