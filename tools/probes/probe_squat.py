#!/usr/bin/env python3
"""屈伸運動: a scripted, perfectly synchronised squat from standstill, no policy.

Zero actions hold HOME (the PD servos). After --settle seconds of standing still, every leg
flexes and extends together: a sine on the knee (J2) and, at half amplitude, on the thigh
(J1), all six legs in the same phase, left and right mirrored by their joint sign
conventions. The amplitude ramps up over --ramp seconds so it starts from a stop.
Optionally the sine is replaced by a bed-locked oscillator (--lock): extend when the base
starts going up, flex when it starts going down (the same rule as the reflex layer).

    python -I probe_squat.py --amp 2.0 --freq 2.8 --out squat.csv [--render clip.mp4] -- \
        --task jumper.posture --agent zero --scene trampoline --backend native --device cpu --headless --steps 3000 --speed 0
"""
from __future__ import annotations

import csv
import math
import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
STAND_Z = 0.09423387151092881
# action indices: LF 0-3, RF 4-7, LM 8-10, RM 11-13, LR 14-16, RR 17-19 (J0, J1, J2[, J3])
# "extend" (stand taller) signs, from the stance angles (knee J2 flexed: LM -1.83, RM +1.83 ...):
KNEE_EXT = {"LF": (2, +1.0), "RF": (6, -1.0), "LM": (10, +1.0), "RM": (13, -1.0), "LR": (16, +1.0), "RR": (19, -1.0)}
# thigh "push down" = opposite of the foot-lift sign measured on 10/9 (lift: LF -, RF -, LM +, RM -, LR +, RR -)
THIGH_DOWN = {"LF": (1, +1.0), "RF": (5, +1.0), "LM": (9, -1.0), "RM": (12, +1.0), "LR": (15, -1.0), "RR": (18, +1.0)}

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]


def opt(name, default, cast=float):
    return cast(mine[mine.index(name) + 1]) if name in mine else default


out_path = opt("--out", None, str)
AMP = opt("--amp", 2.0)          # action units on the knee (2.0 = 0.5 rad); the thigh gets half
FREQ = opt("--freq", 2.8)
SETTLE = opt("--settle", 2.0)
RAMP = opt("--ramp", 1.0)
THIGH = opt("--thigh", 0.5)      # thigh amplitude as a fraction of the knee's
LOCK = "--lock" in mine          # bed-locked instead of a clock
VZ_UP, VZ_DOWN = opt("--vz-up", -0.2), opt("--vz-down", 0.1)
RENDER = opt("--render", None, str)
CAM = (opt("--dist", 1.2), opt("--elev", -18.0), opt("--azim", 135.0))

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
    sensor = unwrapped.scene.sensors["feet_ground_contact"]
    META.update(dt=dt, terms=terms)
    print(f"[squat] amp {AMP} (knee) x{THIGH} (thigh), {'bed-locked' if LOCK else f'{FREQ} Hz'}, settle {SETTLE}s, ramp {RAMP}s")

    obs = env.get_observations()
    zero = policy(obs)
    step = 0
    phase = "DOWN"
    t_phase = 0.0
    armed = False
    u = 0.0
    while True:
        if viewer is not None and not viewer.is_running:
            break
        if max_steps is not None and step >= max_steps:
            break
        t = step * dt
        F = float(torch.linalg.norm(sensor.data.force[0].reshape(-1, 3), dim=1).sum())
        vz = float(robot.data.root_link_lin_vel_w[0, 2])
        z = float(robot.data.root_link_pos_w[0, 2])
        gain = 0.0 if t < SETTLE else min(1.0, (t - SETTLE) / max(RAMP, 1e-6))
        if LOCK:
            if t >= SETTLE:
                if phase == "DOWN":
                    if vz < -0.05:
                        armed = True
                    if t - t_phase > 0.3 and not armed:
                        armed = True       # standing still: start the first push anyway
                    if armed and F > 1.0 and vz > VZ_UP and t - t_phase >= 0.04:
                        phase, t_phase, armed = "UP", t, False
                elif phase == "UP":
                    if F < 1.0 or (vz < VZ_DOWN and t - t_phase >= 0.04):
                        phase, t_phase = "DOWN", t
            u = gain * (AMP if phase == "UP" else -AMP)
        else:
            u = gain * AMP * math.sin(2 * math.pi * FREQ * (t - SETTLE)) if t >= SETTLE else 0.0
        act = torch.zeros_like(zero)
        for leg, (j, s) in KNEE_EXT.items():
            act[0, j] = s * u
        for leg, (j, s) in THIGH_DOWN.items():
            act[0, j] = s * u * THIGH
        with torch.inference_mode():
            obs, _, _, _ = env.step(act)
        fired = [n for n in terms if bool(tm.get_term(n)[0])]
        w = robot.data.root_link_ang_vel_w[0]
        ROWS.append([step, round((step + 1) * dt, 4), z, vz, F, u, phase if LOCK else "", float(torch.linalg.norm(w[:2])), ";".join(fired)])
        if monitor is not None:
            monitor.update()
        step += 1
    print(f"\n[squat] stopped after {step} steps")


play._loop = _loop
if RENDER:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_sim
    render_sim.install(play, RENDER, cam=CAM, mask="--no-mask" not in mine, slow=opt("--slow", 3.0))


def summary() -> None:
    if not ROWS:
        return
    dt = META["dt"]
    z = [r[2] for r in ROWS]
    F = [r[4] for r in ROWS]
    airborne = [f < 1.0 for f in F]
    print("\n[squat] ===== summary =====")
    print(f"[squat] {len(ROWS) * dt:.1f} s; base z max {max(z):.3f} (rise {max(z) - STAND_Z:+.3f}), min {min(z):.3f}; airborne {sum(airborne) * dt:.2f} s")
    ends = [(r[1], r[8]) for r in ROWS if r[8]]
    print(f"[squat] episode ends: {len(ends)} -> " + ", ".join(f"t={t:.2f}s {w}" for t, w in ends))
    apex = []
    for k in range(1, len(z) - 1):
        if airborne[k] and z[k] >= z[k - 1] and z[k] > z[k + 1] and z[k] > STAND_Z + 0.005:
            if apex and ROWS[k][1] - apex[-1][0] < 0.08:
                if z[k] > apex[-1][1]:
                    apex[-1] = (ROWS[k][1], z[k])
                continue
            apex.append((ROWS[k][1], z[k]))
    print(f"[squat] airborne apexes: {len(apex)}" + ("; first " + ", ".join(f"t={t:.2f}s +{h - STAND_Z:.3f}" for t, h in apex[:6]) if apex else ""))
    post = [r for r in ROWS if r[1] >= SETTLE + RAMP + 1.0]
    if post:
        zz = [r[2] for r in post]
        print(f"[squat] after settling: base z swing {(max(zz) - min(zz)) * 100:.1f} cm, mean {sum(zz) / len(zz):.3f}")
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "t", "base_z", "vz", "foot_force_total", "u", "phase", "tilt_rate", "terminations"])
            w.writerows(ROWS)
        print(f"[squat] log -> {out_path}")


sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
try:
    play.main()
except SystemExit as e:
    print(f"[squat] play.main exited with {e.code}")
finally:
    summary()
