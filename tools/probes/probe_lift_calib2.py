#!/usr/bin/env python3
"""Which joint LIFTS each foot (up, toward the body) rather than swinging it sideways?

Standing with zero actions (PD holds HOME), each actuated joint is driven to +AMP and then
-AMP action units for HOLD seconds. For that leg's foot we record, relative to the base:
the rise of the foot tip (cm), its horizontal displacement (cm), and whether the foot
unloaded (its contact force fell below 1 N) -- a lifted foot leaves the ground, a swung one
drags along it. Prints a table and names, per leg, the joint+sign that lifts.

    python -I probe_lift_calib2.py --amp 2.0 --out lift_calib2.json -- --task jumper.posture --agent zero --scene studio --backend native --device cpu --headless --steps 6200 --speed 0
"""
from __future__ import annotations

import json
import math
import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
FOOT_OF = {"LF": "LF_palm_pad_b_link", "RF": "RF_palm_pad_b_link", "LM": "LM_foot_tip_link",
           "RM": "RM_foot_tip_link", "LR": "LR_foot_tip_link", "RR": "RR_foot_tip_link"}
JOINTS = ["LF_J0_joint", "LF_J1_joint", "LF_J2_joint", "LF_J3_joint", "RF_J0_joint", "RF_J1_joint",
          "RF_J2_joint", "RF_J3_joint", "LM_J0_joint", "LM_J1_joint", "LM_J2_joint", "RM_J0_joint",
          "RM_J1_joint", "RM_J2_joint", "LR_J0_joint", "LR_J1_joint", "LR_J2_joint", "RR_J0_joint",
          "RR_J1_joint", "RR_J2_joint"]

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]
out_path = mine[mine.index("--out") + 1] if "--out" in mine else None
AMP = float(mine[mine.index("--amp") + 1]) if "--amp" in mine else 2.0   # action units (x 0.25 rad)
HOLD, REST, SETTLE = 0.35, 0.40, 0.8

os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import play  # noqa: E402

RESULT: dict = {}


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=play._speed_text, monitor=None):
    import torch

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene["robot"]
    sensor = unwrapped.scene.sensors["feet_ground_contact"]
    from tasks.jumper.common.constants import FEET
    slot = {f.split("_")[0]: i for i, f in enumerate(FEET)}
    dt = float(unwrapped.step_dt)
    names = list(robot.body_names)
    foot_idx = {leg: names.index(body) for leg, body in FOOT_OF.items()}

    def foot_state(leg):
        pos = robot.data.body_link_pos_w[0, foot_idx[leg]]
        base = robot.data.root_link_pos_w[0]
        f = sensor.data.force[0].reshape(-1, 3)
        return (float(pos[0] - base[0]), float(pos[1] - base[1]), float(pos[2] - base[2]),
                float(torch.linalg.norm(f[slot[leg]])))

    obs = env.get_observations()
    zero = policy(obs)
    plan = []   # (start, end, joint index, sign)
    t0 = int(SETTLE / dt)
    for j in range(len(JOINTS)):
        for sgn in (+1.0, -1.0):
            plan.append((t0, t0 + int(HOLD / dt), j, sgn))
            t0 += int((HOLD + REST) / dt)
    rows = []
    step = 0
    ref = {}
    while max_steps is None or step < max_steps:
        act = torch.zeros_like(zero)
        cur = None
        for a, b, j, s in plan:
            if a <= step < b:
                act[0, j] = s * AMP
                cur = (a, b, j, s)
        if cur is not None and step == cur[0]:
            ref = {leg: foot_state(leg) for leg in FOOT_OF}
        with torch.inference_mode():
            obs, _, _, _ = env.step(act)
        if cur is not None and step == cur[1] - 1:
            j, s = cur[2], cur[3]
            leg = JOINTS[j].split("_")[0]
            x0, y0, z0, f0 = ref[leg]
            x1, y1, z1, f1 = foot_state(leg)
            rise = (z1 - z0) * 100
            slide = math.hypot(x1 - x0, y1 - y0) * 100
            rows.append((JOINTS[j], s, leg, rise, slide, f0, f1))
        step += 1

    print(f"\n[calib2] {AMP:+.1f} action units (= {AMP * 0.25:.2f} rad) held {HOLD} s, standing, zero actions")
    print(f"[calib2] {'joint':<14}{'sign':>5}{'foot rise cm':>13}{'slide cm':>10}{'force before->after N':>24}  verdict")
    best = {}
    for name, s, leg, rise, slide, f0, f1 in rows:
        verdict = "LIFT" if (rise > 1.0 and f1 < 1.0) else ("up" if rise > 1.0 else ("swing" if slide > 1.0 and rise <= 1.0 else "-"))
        print(f"[calib2] {name:<14}{s:>+5.0f}{rise:>13.1f}{slide:>10.1f}{f0:>11.1f} -> {f1:<8.1f}  {verdict}")
        score = rise if f1 < 1.0 else rise * 0.5
        if rise > 1.0 and (leg not in best or score > best[leg][2]):
            best[leg] = (name, s, score, rise, slide, f1)
    print("[calib2] per leg, the best LIFT (foot rises, unloads):")
    for leg in FOOT_OF:
        if leg in best:
            name, s, _, rise, slide, f1 = best[leg]
            print(f"[calib2]   {leg}: {name} {'+' if s > 0 else '-'}   rise {rise:.1f} cm, slide {slide:.1f} cm, force after {f1:.1f} N")
        else:
            print(f"[calib2]   {leg}: no joint lifted the foot by more than 1 cm")
    RESULT.update(amp=AMP, rows=rows, lift={leg: {"joint": v[0], "sign": v[1], "rise_cm": v[3], "slide_cm": v[4]} for leg, v in best.items()})
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(RESULT, fh, indent=1)
        print(f"[calib2] -> {out_path}")


play._loop = _loop
sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
play.main()
