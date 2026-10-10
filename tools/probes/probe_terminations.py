#!/usr/bin/env python3
"""One-off probe, kept outside the repository (CLAUDE.md: task probes stay out of tests/).

Runs scripts/play.py exactly as given on the command line, but swaps its `_loop` for
one that records, every control step, the base height of environment 0 and which
termination terms fired. Prints a summary at the end: episodes, why each ended,
apex height and lowest base height, and writes the per-step log as CSV.

    python -I probe_terminations.py --out log.csv -- <play.py arguments>
"""
from __future__ import annotations

import csv
import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
STAND_Z = 0.09423387151092881  # ref clip meta: stand_base_z

# ── split our own args from play.py's ──
if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]
out_path = None
if "--out" in mine:
    out_path = mine[mine.index("--out") + 1]

os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import play  # noqa: E402  (the repository's scripts/play.py)

LOG: list[tuple[int, float, float, str]] = []
STEP_DT = [None]


def _root_z(robot) -> float:
    data = robot.data
    for name in ("root_link_pos_w", "root_pos_w", "root_com_pos_w"):
        t = getattr(data, name, None)
        if t is not None:
            return float(t[0, 2])
    raise AttributeError("no root position attribute on the entity data")


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=play._speed_text, monitor=None):
    import torch

    unwrapped = getattr(env, "unwrapped", env)
    tm = unwrapped.termination_manager
    names = list(getattr(tm, "active_terms", None) or getattr(tm, "_term_names", []))
    robot = unwrapped.scene["robot"]
    STEP_DT[0] = float(unwrapped.step_dt)
    print(f"[probe] termination terms: {names}; step_dt {STEP_DT[0]}")
    obs = env.get_observations()
    step = 0
    while True:
        if viewer is not None and not viewer.is_running:
            break
        if max_steps is not None and step >= max_steps:
            break
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        fired = [n for n in names if bool(tm.get_term(n)[0])]
        LOG.append((step, (step + 1) * STEP_DT[0], _root_z(robot), ";".join(fired)))
        if monitor is not None:
            monitor.update()
        step += 1
    print(f"\n[probe] stopped after {step} steps")


play._loop = _loop


def summary() -> None:
    if not LOG:
        print("[probe] nothing recorded")
        return
    zs = [r[2] for r in LOG]
    events = [(r[1], r[3]) for r in LOG if r[3]]
    print("\n[probe] ===== summary =====")
    print(f"[probe] steps {len(LOG)}  ({len(LOG) * STEP_DT[0]:.2f} s of simulation)")
    print(f"[probe] base z: max {max(zs):.3f} m (rise over stand {max(zs) - STAND_Z:+.3f} m), "
          f"min {min(zs):.3f} m (stand {STAND_Z:.3f})")
    print(f"[probe] episode ends: {len(events)}")
    for t, why in events:
        print(f"[probe]   t={t:6.3f}s  {why}")
    # apex per episode
    start = 0
    ep = 0
    for i, r in enumerate(LOG):
        if r[3] or i == len(LOG) - 1:
            seg = zs[start:i + 1]
            print(f"[probe] episode {ep}: {len(seg)} steps, apex z {max(seg):.3f} "
                  f"(rise {max(seg) - STAND_Z:+.3f}), min z {min(seg):.3f}")
            start = i + 1
            ep += 1
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "t", "base_z", "terminations"])
            w.writerows(LOG)
        print(f"[probe] log -> {out_path}")


sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
try:
    play.main()
except SystemExit as e:
    print(f"[probe] play.main exited with {e.code}")
finally:
    summary()
