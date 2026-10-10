#!/usr/bin/env python3
"""Run any probe (or render_sim.py) with the env config's seed fixed, so a take scouted small can be
rendered again at full size and come out the same (native backend, CPU).

    python -I seeded.py SEED probe.py [probe args] -- [play args]
"""
import os
import runpy
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")
seed, script = int(sys.argv[1]), os.path.abspath(sys.argv[2])
sys.argv = [script, *sys.argv[3:]]
sys.path.insert(0, os.path.dirname(script))
sys.path.insert(0, REPO)
os.chdir(REPO)
import tasks as _tasks  # noqa: E402

_orig_load = _tasks.load_env_cfg


def _seeded_load(*a, **k):
    cfg = _orig_load(*a, **k)
    cfg.seed = seed
    return cfg


_tasks.load_env_cfg = _seeded_load
print(f"[seeded] seed {seed} -> {os.path.basename(script)}")
runpy.run_path(script, run_name="__main__")
