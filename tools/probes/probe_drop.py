#!/usr/bin/env python3
"""One-off probe, outside the repository: the bed's natural period and the "arrow" at landing.

Wraps scripts/play.py like the other probes. Right after the first reset it can lift the
robot by --lift metres (a drop test: with `--agent zero` the PD servos hold HOME and the
robot is a passive spring-mass on the bed). Every control step of environment 0 it logs
the base height, the base velocity and angular velocity in the world frame, the total
foot force and whether a termination fired. The summary reports:

  * apexes: local maxima of the base height while airborne, their times, heights and
    the intervals between them (the bed's natural period for this body);
  * landings: the first contact after >= 50 ms airborne, with the "arrow" at that
    instant -- horizontal speed and heading, vertical speed, pitch/roll rate -- and
    the peak foot force in the 100 ms after it.

    python -I probe_drop.py --lift 0.10 --out log.csv -- --task jumper.posture --agent zero --scene trampoline ...
    python -I probe_drop.py --out log.csv -- --task jumper.jump --checkpoint ... --scene trampoline ...
"""
from __future__ import annotations

import csv
import math
import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
STAND_Z = 0.09423387151092881

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]
out_path = mine[mine.index("--out") + 1] if "--out" in mine else None
lift = float(mine[mine.index("--lift") + 1]) if "--lift" in mine else 0.0

os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import play  # noqa: E402

ROWS: list[list] = []
META: dict = {}


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=play._speed_text, monitor=None):
    import torch

    unwrapped = getattr(env, "unwrapped", env)
    tm = unwrapped.termination_manager
    terms = list(getattr(tm, "active_terms", None) or [])
    robot = unwrapped.scene["robot"]
    dt = float(unwrapped.step_dt)
    try:
        sensor = unwrapped.scene.sensors["feet_ground_contact"]
    except Exception:
        sensor = None
    META.update(dt=dt, sensor=sensor is not None, terms=terms)

    obs = env.get_observations()
    if lift > 0:
        # lift environment 0 straight up and let it fall: the drop test
        ids = torch.tensor([0], device=robot.data.root_link_pose_w.device)
        pose = robot.data.root_link_pose_w[ids].clone()
        pose[:, 2] += lift
        robot.write_root_link_pose_to_sim(pose, env_ids=ids)
        vel = torch.zeros((1, 6), device=pose.device, dtype=pose.dtype)
        robot.write_root_link_velocity_to_sim(vel, env_ids=ids)
        print(f"[probe] lifted env 0 by {lift:.3f} m (base z now {float(pose[0, 2]):.3f})")

    step = 0
    while True:
        if viewer is not None and not viewer.is_running:
            break
        if max_steps is not None and step >= max_steps:
            break
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        fired = [n for n in terms if bool(tm.get_term(n)[0])]
        z = float(robot.data.root_link_pos_w[0, 2])
        v = robot.data.root_link_lin_vel_w[0].tolist()
        w = robot.data.root_link_ang_vel_w[0].tolist()
        total = math.nan
        if sensor is not None:
            f = sensor.data.force[0].reshape(-1, 3)
            total = float(torch.linalg.norm(f, dim=1).sum())
        ROWS.append([step, round((step + 1) * dt, 4), z, *v, *w, total, ";".join(fired)])
        if monitor is not None:
            monitor.update()
        step += 1
    print(f"\n[probe] stopped after {step} steps")


play._loop = _loop


def summary() -> None:
    if not ROWS:
        print("[probe] nothing recorded")
        return
    dt = META["dt"]
    z = [r[2] for r in ROWS]
    F = [r[9] for r in ROWS]
    airborne = [(f < 1.0) if f == f else (zz > STAND_Z + 0.02) for f, zz in zip(F, z)]
    print("\n[probe] ===== summary =====")
    print(f"[probe] steps {len(ROWS)} ({len(ROWS) * dt:.2f} s); base z max {max(z):.3f} (rise {max(z) - STAND_Z:+.3f}), min {min(z):.3f}")
    ends = [(r[1], r[10]) for r in ROWS if r[10]]
    print(f"[probe] episode ends: {len(ends)} -> " + ", ".join(f"t={t:.2f}s {w}" for t, w in ends))
    # apexes: local maxima while airborne
    apex = []
    for k in range(1, len(z) - 1):
        if airborne[k] and z[k] >= z[k - 1] and z[k] > z[k + 1] and z[k] > STAND_Z + 0.005:
            if apex and ROWS[k][1] - apex[-1][0] < 0.08:   # merge jitter within 80 ms
                if z[k] > apex[-1][1]:
                    apex[-1] = (ROWS[k][1], z[k])
                continue
            apex.append((ROWS[k][1], z[k]))
    print(f"[probe] apexes (airborne local maxima of base z): {len(apex)}")
    for i, (t, h) in enumerate(apex):
        gap = f"  interval {t - apex[i - 1][0]:.3f} s" if i else ""
        ratio = f"  height ratio {(h - STAND_Z) / max(1e-6, apex[i - 1][1] - STAND_Z):.2f}" if i else ""
        print(f"[probe]   t={t:6.3f}s  z={h:.3f} (rise {h - STAND_Z:+.3f}){gap}{ratio}")
    if len(apex) >= 2:
        gaps = [apex[i][0] - apex[i - 1][0] for i in range(1, len(apex))]
        print(f"[probe]   mean interval {sum(gaps) / len(gaps):.3f} s over {len(gaps)} bounces")
    # landings with the arrow
    print("[probe] landings (first contact after >= 50 ms airborne) and the arrow at that instant:")
    low = 0
    for k, r in enumerate(ROWS):
        if airborne[k]:
            low += 1
            continue
        if low * dt >= 0.05:
            vx, vy, vz, wx, wy, wz = r[3:9]
            hspeed = math.hypot(vx, vy)
            heading = math.degrees(math.atan2(vy, vx))
            tilt_rate = math.hypot(wx, wy)
            window = [f for f in F[k:k + int(0.1 / dt)] if f == f]
            peak = max(window) if window else math.nan
            print(f"[probe]   t={r[1]:6.3f}s  horizontal {hspeed:.2f} m/s toward {heading:+.0f} deg, "
                  f"vertical {vz:+.2f} m/s, pitch/roll rate {tilt_rate:.2f} rad/s (yaw {wz:+.2f}), "
                  f"peak foot force {peak:.0f} N")
        low = 0
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "t", "base_z", "vx", "vy", "vz", "wx", "wy", "wz", "foot_force_total", "terminations"])
            w.writerows(ROWS)
        print(f"[probe] log -> {out_path}")


sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
try:
    play.main()
except SystemExit as e:
    print(f"[probe] play.main exited with {e.code}")
finally:
    summary()
