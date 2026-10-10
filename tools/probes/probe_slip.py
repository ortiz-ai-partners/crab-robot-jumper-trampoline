#!/usr/bin/env python3
"""One-off probe, outside the repository: foot slip and ground force during a replay.

Wraps scripts/play.py like probe_terminations.py and records, every control step of
environment 0: base height, termination terms that fired, each foot tip's horizontal
speed while in contact (tip z below a threshold), and the per-foot ground force from
the task's own `feet_ground_contact` sensor. Prints a summary at the end.

    python -I probe_slip.py --out log.csv -- <play.py arguments>
"""
from __future__ import annotations

import csv
import math
import os
import re
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
STAND_Z = 0.09423387151092881
CONTACT_Z = 0.03          # a tip below this is "on the bed" (tips sink below 0 on a soft bed)
# The front legs walk on the rear palm pad (constants.TIP_LINKS comment), the others on the toe tip.
FOOT_RE = re.compile(r"(LF_palm_pad_b_link|RF_palm_pad_b_link|LM_foot_tip_link|RM_foot_tip_link|LR_foot_tip_link|RR_foot_tip_link)$")

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]
out_path = mine[mine.index("--out") + 1] if "--out" in mine else None

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
    names = list(robot.body_names)
    foot_idx = [i for i, n in enumerate(names) if FOOT_RE.search(n)]
    if not foot_idx:
        foot_idx = [i for i, n in enumerate(names) if "toe_tip" in n or n.endswith("_F_Link")]
    foot_names = [names[i] for i in foot_idx]
    dt = float(unwrapped.step_dt)
    sensor = None
    try:
        sensor = unwrapped.scene.sensors["feet_ground_contact"]
    except Exception:
        sensor = None
    body_sensor = None
    try:
        body_sensor = unwrapped.scene.sensors["body_ground_contact"]
    except Exception:
        body_sensor = None
    # the contact sensor's slots follow constants.FEET; map each foot body to its slot by leg prefix
    sidx = list(range(len(foot_idx)))
    try:
        from tasks.jumper.common.constants import FEET
        order = [f.split("_")[0] for f in FEET]
        sidx = [order.index(n.split("_")[0]) for n in foot_names]
        print(f"[probe] sensor slot order {list(FEET)}")
    except Exception as exc:
        print(f"[probe] could not read constants.FEET ({exc}); assuming slot order = body order")
    META.update(terms=terms, feet=foot_names, dt=dt, sidx=sidx,
                sensor=sensor is not None, body_sensor=body_sensor is not None)
    print(f"[probe] feet {foot_names}; sensor {'yes' if sensor else 'no'}; dt {dt}")

    obs = env.get_observations()
    step = 0
    said = False
    while True:
        if viewer is not None and not viewer.is_running:
            break
        if max_steps is not None and step >= max_steps:
            break
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        fired = [n for n in terms if bool(tm.get_term(n)[0])]
        z = float(robot.data.root_link_pos_w[0, 2])
        pos = robot.data.body_link_pos_w[0, foot_idx]          # [6, 3]
        vel = robot.data.body_link_lin_vel_w[0, foot_idx]      # [6, 3]
        tip_z = pos[:, 2].tolist()
        vxy = torch.linalg.norm(vel[:, :2], dim=1).tolist()
        forces = [math.nan] * len(foot_idx)
        total = math.nan
        if sensor is not None:
            f = sensor.data.force[0]
            f = f.reshape(-1, 3) if f.dim() > 1 else f.reshape(-1, 3)
            if not said:
                print(f"[probe] sensor force shape per env {tuple(sensor.data.force.shape[1:])}")
                said = True
            mags = torch.linalg.norm(f, dim=1).tolist()
            forces = [mags[s] if s < len(mags) else math.nan for s in META["sidx"]]
            total = float(sum(m for m in mags if m == m))
        body_f = math.nan
        if body_sensor is not None:
            bf = body_sensor.data.force[0].reshape(-1, 3)
            body_f = float(torch.linalg.norm(bf, dim=1).sum())
        ROWS.append([step, round((step + 1) * dt, 4), z, ";".join(fired), *tip_z, *vxy, *forces, total, body_f])
        if monitor is not None:
            monitor.update()
        step += 1
    print(f"\n[probe] stopped after {step} steps")


play._loop = _loop


def summary() -> None:
    if not ROWS:
        print("[probe] nothing recorded")
        return
    feet = META["feet"]
    n = len(feet)
    dt = META["dt"]
    zs = [r[2] for r in ROWS]
    ends = [(r[1], r[3]) for r in ROWS if r[3]]
    print("\n[probe] ===== summary =====")
    print(f"[probe] steps {len(ROWS)} ({len(ROWS) * dt:.2f} s); base z max {max(zs):.3f} "
          f"(rise {max(zs) - STAND_Z:+.3f}), min {min(zs):.3f}")
    print(f"[probe] episode ends: {len(ends)} -> " + ", ".join(f"t={t:.2f}s {w}" for t, w in ends))
    # slip: horizontal travel of each tip while below CONTACT_Z
    slip = [0.0] * n
    vmax = [0.0] * n
    steps_in = [0] * n
    use_force = META["sensor"]
    for r in ROWS:
        tip_z = r[4:4 + n]
        vxy = r[4 + n:4 + 2 * n]
        fz = r[4 + 2 * n:4 + 3 * n]
        for i in range(n):
            on_bed = (fz[i] > 1.0) if use_force else (tip_z[i] < CONTACT_Z)
            if on_bed:
                slip[i] += vxy[i] * dt
                steps_in[i] += 1
                vmax[i] = max(vmax[i], vxy[i])
    how = "contact force > 1 N" if use_force else f"tip z < {CONTACT_Z} m"
    print(f"[probe] slip while on the bed ({how}): horizontal travel of the foot while loaded")
    for i, name in enumerate(feet):
        print(f"[probe]   {name:<18} travel {slip[i] * 100:6.1f} cm   max speed {vmax[i]:5.2f} m/s   "
              f"on bed {steps_in[i] * dt:5.2f} s")
    rear = [i for i, nm in enumerate(feet) if nm.startswith(("LR", "RR"))]
    front = [i for i, nm in enumerate(feet) if nm.startswith(("LF", "RF"))]
    print(f"[probe] rear feet travel {sum(slip[i] for i in rear) * 100:.1f} cm, "
          f"front {sum(slip[i] for i in front) * 100:.1f} cm, all {sum(slip) * 100:.1f} cm")
    if META["sensor"]:
        totals = [r[4 + 3 * n] for r in ROWS]
        per_foot_max = [max(r[4 + 2 * n + i] for r in ROWS) for i in range(n)]
        peak = max(totals)
        t_peak = ROWS[totals.index(peak)][1]
        # landings: total force rises through 1 N after being below it for >= 50 ms
        landings = []
        low = 0
        for k, r in enumerate(ROWS):
            if totals[k] < 1.0:
                low += 1
            else:
                if low * dt >= 0.05:
                    window = totals[k:k + int(0.1 / dt)]
                    landings.append((r[1], max(window)))
                low = 0
        print(f"[probe] ground force (feet_ground_contact): peak total {peak:.1f} N at t={t_peak:.2f}s; "
              f"standing weight is {2.543 * 9.81:.1f} N")
        print(f"[probe]   per-foot peak: " + ", ".join(f"{feet[i]} {per_foot_max[i]:.0f}" for i in range(n)))
        print(f"[probe]   landings (after >=50 ms airborne): " + ", ".join(f"t={t:.2f}s {f:.0f} N" for t, f in landings))
        bodies = [r[5 + 3 * n] for r in ROWS]
        if META["body_sensor"]:
            print(f"[probe] non-foot body contact force peak {max(bodies):.1f} N (0 means the shell never touched the bed)")
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["step", "t", "base_z", "terminations", *[f"tipz/{x}" for x in feet],
                        *[f"vxy/{x}" for x in feet], *[f"force/{x}" for x in feet], "force_total", "body_force"])
            w.writerows(ROWS)
        print(f"[probe] log -> {out_path}")


sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
try:
    play.main()
except SystemExit as e:
    print(f"[probe] play.main exited with {e.code}")
finally:
    summary()
