#!/usr/bin/env python3
"""Which way does the body turn when limbs swing in the air? A direct measurement.

Lifts the robot (zero actions: PD holds HOME) and, while it falls, drives a chosen set of
legs' lift joints (J1, calibrated signs) with +amp action units from t0 to t1. Records the
body's projected gravity and angular rate; prints how the pitch/roll gravity components
changed during the swing, i.e. which way the body rotated in reaction.

    python -I probe_air_reaction.py --limbs front --amp 1.0 -- --task jumper.posture --agent zero --scene studio --backend native --device cpu --headless --steps 140 --speed 0
"""
from __future__ import annotations

import os
import sys

REPO = r"D:\IslandMaster\jumper"
LIFT = {"LF": (1, -1.0), "RF": (5, -1.0), "LM": (9, +1.0), "RM": (12, -1.0), "LR": (15, +1.0), "RR": (18, -1.0)}
GROUPS = {"front": ("LF", "RF"), "rear": ("LR", "RR"), "left": ("LF", "LM", "LR"), "right": ("RF", "RM", "RR"),
          "mid": ("LM", "RM"), "none": ()}

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]


def opt(name, default, cast=float):
    return cast(mine[mine.index(name) + 1]) if name in mine else default


LIMBS = opt("--limbs", "front", str)
AMP = opt("--amp", 1.0)
LIFT_M = opt("--lift", 0.5)
T0, T1 = opt("--t0", 0.10), opt("--t1", 0.35)
# The coach (10/9): swing from the ROOT of the limb with the knee / elbow EXTENDED, so the lever is
# long. J1 is the root pitch (thigh / upper arm); J2 is the knee / elbow, J3 the front wrist. "Extend"
# = drive J2 (and J3) toward zero, i.e. against the sign of their stance angle.
EXTEND = "--extend" in mine
EXT_AMP = opt("--ext-amp", 3.0)   # action units toward straight (3.0 = 0.75 rad)
EXT = {"LF": [(2, +1.0), (3, +1.0)], "RF": [(6, -1.0), (7, -1.0)], "LM": [(10, +1.0)], "RM": [(13, -1.0)],
       "LR": [(16, +1.0)], "RR": [(19, -1.0)]}
FOOT_OF = {"LF": "LF_palm_pad_b_link", "RF": "RF_palm_pad_b_link", "LM": "LM_foot_tip_link",
           "RM": "RM_foot_tip_link", "LR": "LR_foot_tip_link", "RR": "RR_foot_tip_link"}

os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import play  # noqa: E402


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=play._speed_text, monitor=None):
    import torch

    unwrapped = getattr(env, "unwrapped", env)
    robot = unwrapped.scene["robot"]
    dt = float(unwrapped.step_dt)
    obs = env.get_observations()
    ids = torch.tensor([0], device=robot.data.root_link_pose_w.device)
    pose = robot.data.root_link_pose_w[ids].clone()
    pose[:, 2] += LIFT_M
    robot.write_root_link_pose_to_sim(pose, env_ids=ids)
    robot.write_root_link_velocity_to_sim(torch.zeros((1, 6), device=pose.device, dtype=pose.dtype), env_ids=ids)
    legs = GROUPS[LIMBS]
    names = list(robot.body_names)
    foot_idx = {leg: names.index(FOOT_OF[leg]) for leg in legs}
    print(f"[air] lift {LIFT_M} m; swing {LIMBS} legs {legs} by {AMP:+.2f} action units from {T0}s to {T1}s"
          f"{' with knee/elbow extended (+%.1f)' % EXT_AMP if EXTEND else ' (knees as in stance)'}")
    g_at = {}
    step = 0
    while max_steps is None or step < max_steps:
        t = step * dt
        act = torch.zeros_like(policy(obs))
        if T0 <= t < T1:
            for leg in legs:
                idx, sgn = LIFT[leg]
                act[0, idx] = sgn * AMP
                if EXTEND:
                    for j, s in EXT[leg]:
                        act[0, j] = s * EXT_AMP
        with torch.inference_mode():
            obs, _, _, _ = env.step(act)
        g = robot.data.projected_gravity_b[0].tolist()
        w = robot.data.root_link_ang_vel_b[0].tolist()
        z = float(robot.data.root_link_pos_w[0, 2])
        feet = {leg: float(robot.data.body_link_pos_w[0, i, 2]) - z for leg, i in foot_idx.items()}
        for mark in (T0, T1, T1 + 0.1):
            if abs(t - mark) < dt / 2:
                g_at[round(mark, 2)] = (g, w, z, feet)
        step += 1
    k = sorted(g_at)
    print(f"[air] {'t':>5} {'gx (nose down +)':>18} {'gy (left down +)':>18} {'wx':>7} {'wy':>7} {'base z':>8}")
    for m in k:
        g, w, z, feet = g_at[m]
        print(f"[air] {m:>5.2f} {g[0]:>18.3f} {g[1]:>18.3f} {w[0]:>7.2f} {w[1]:>7.2f} {z:>8.3f}   feet rel. base: "
              + ", ".join(f"{leg} {v * 100:+.1f} cm" for leg, v in feet.items()))
    if len(k) >= 2:
        g0, g1 = g_at[k[0]][0], g_at[k[1]][0]
        f0, f1 = g_at[k[0]][3], g_at[k[1]][3]
        rise = ", ".join(f"{leg} {(f1[leg] - f0[leg]) * 100:+.1f} cm" for leg in f0)
        print(f"[air] during the swing: d(gx) = {g1[0] - g0[0]:+.3f} ({'nose went DOWN' if g1[0] > g0[0] else 'nose went UP'}), "
              f"d(gy) = {g1[1] - g0[1]:+.3f} ({'left side went DOWN' if g1[1] > g0[1] else 'left side went UP'}); "
              f"swung feet moved relative to the base: {rise}")


play._loop = _loop
sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
play.main()
