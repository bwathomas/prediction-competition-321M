"""Run ONE exp_loo_category_mlp.fn() in THIS process, configured entirely from SHIP_* env.

Purpose: OOM ISOLATION. A driver spawns this as a fresh subprocess per
(family, model, mode, fold). Because each run is its own OS process:
  * all its memory (the ~37GB assembly + model) is fully reclaimed by the OS on exit
    — no accumulation across a long sequential sweep, no fragmentation;
  * if a single run OOMs, the OS OOM-killer takes THIS subprocess only — the parent
    kernel (and the Drive mount) survive, and the driver's subprocess.run returns a
    non-zero code which the driver logs and steps past.

Usage (from a run_bg driver):
    env = {**os.environ, "SHIP_FAMILY": fam, "SHIP_MODEL": m, "SHIP_MODE": "full"|"library",
           "SHIP_OOF_FOLD": str(f), "SHIP_ROW_SOURCE": "full", ...}
    r = subprocess.run([sys.executable, "/content/pc321/scripts/ship/run_one.py"],
                       env=env, capture_output=True, text=True, timeout=...)
    ok = (r.returncode == 0)

The exp module reads SHIP_* at import time, so this process must be started with the env
already set (subprocess env=), NOT mutated after import.
"""
import importlib.util
import os
import sys

EXP = os.environ.get("SHIP_EXP_PATH", "/content/pc321/scripts/ship/exp_loo_category_mlp.py")


def main() -> int:
    spec = importlib.util.spec_from_file_location("exp_one", EXP)
    exp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(exp)
    print(f"RUN family={exp.FAMILY} model={exp.MODEL} mode={os.environ.get('SHIP_MODE')} "
          f"fold={exp.OOF_FOLD} full_only={exp.FULL_ONLY} library={exp.LIBRARY} "
          f"save_root={exp.SAVE_ROOT}", flush=True)
    res = exp.fn()
    print(f"DONE ok={res.get('ok')} t_total_s={res.get('t_total_s')} "
          f"sl={res.get('soft_logloss')}", flush=True)
    return 0 if res.get("ok") else 1


if __name__ == "__main__":
    sys.exit(main())
