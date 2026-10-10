#!/usr/bin/env python3
"""One-off probe, outside the repository: the "bounce dance" (stage 0.5, no training).

Wraps scripts/play.py for `jumper.posture` and drives the posture command's HEIGHT
channel with a sine wave while the robot stands on the bed: knees bend and extend in
rhythm, legs never fully extended, the policy's own balancing does the rest.

    height(t) = NEUTRAL + amp * sin(2*pi*freq*(t - start) + phase)   for t >= start

Logs the base height, base velocity, foot force and the commanded height every control
step of environment 0, and summarises the apexes (local maxima while airborne) and
their growth, the same way probe_drop.py does.

    python -I probe_bounce_dance.py --amp 0.03 --freq 2.5 --start 1.0 --out log.csv -- \
        --task jumper.posture --checkpoint tasks/jumper/posture/out/example/model_74800.pt \
        --scene trampoline --backend native --device cpu --headless --steps 1500 --speed 0
"""
from __future__ import annotations

import csv
import math
import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
STAND_Z = 0.09423387151092881
NEUTRAL_HEIGHT = 0.10647          # tasks/jumper/posture: the standing height the command is centred on
HEIGHT_RANGE = (0.07, 0.15)       # the command's own range

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]


def opt(name, default, cast=float):
    return cast(mine[mine.index(name) + 1]) if name in mine else default


# options before "--", or environment variables (for the recording helper, which cannot pass "--")
out_path = opt("--out", os.environ.get("DANCE_OUT") or None, str)
AMP = opt("--amp", float(os.environ.get("DANCE_AMP", "0.03")))
FREQ = opt("--freq", float(os.environ.get("DANCE_FREQ", "2.5")))
START = opt("--start", float(os.environ.get("DANCE_START", "1.0")))
PHASE = opt("--phase", float(os.environ.get("DANCE_PHASE", "0.0")))       # radians
TERM = opt("--term", "posture", str)
# live-viewer camera for recordings, same variables as play_camera.py
CAM = None
if os.environ.get("REC_CAM_DISTANCE"):
    CAM = (float(os.environ["REC_CAM_DISTANCE"]), float(os.environ.get("REC_CAM_ELEVATION", "-18")),
           float(os.environ.get("REC_CAM_AZIMUTH", "135")))

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
    cmd = unwrapped.command_manager.get_term(TERM)
    try:
        sensor = unwrapped.scene.sensors["feet_ground_contact"]
    except Exception:
        sensor = None
    META.update(dt=dt, sensor=sensor is not None, terms=terms)
    print(f"[probe] command term {TERM}: {type(cmd).__name__}, layout {getattr(cmd, 'LAYOUT', '?')}; "
          f"amp {AMP} m, freq {FREQ} Hz, start {START} s, phase {PHASE} rad; dt {dt}")

    def set_height(h: float) -> None:
        h = min(max(h, HEIGHT_RANGE[0]), HEIGHT_RANGE[1])
        cmd.posture_command[:, 0:3] = 0.0
        cmd.posture_command[:, 3] = h

    obs = env.get_observations()
    set_height(NEUTRAL_HEIGHT)
    step = 0
    cam_left = 400 if (CAM and viewer is not None) else 0
    while True:
        if viewer is not None and not viewer.is_running:
            break
        if max_steps is not None and step >= max_steps:
            break
        if cam_left > 0:
            handle = getattr(viewer, "_handle", None)
            if handle is not None:
                try:
                    with handle.lock():
                        handle.cam.distance, handle.cam.elevation, handle.cam.azimuth = CAM
                    cam_left -= 1
                    if cam_left == 399:
                        print(f"[cam] distance {CAM[0]} elevation {CAM[1]} azimuth {CAM[2]}")
                except Exception as exc:
                    print(f"[cam] could not set the camera: {exc}")
                    cam_left = 0
        t = step * dt
        h = NEUTRAL_HEIGHT
        if t >= START:
            h = NEUTRAL_HEIGHT + AMP * math.sin(2 * math.pi * FREQ * (t - START) + PHASE)
        set_height(h)
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        set_height(h)   # again after the step, so the observation built inside it carries it next step too
        fired = [n for n in terms if bool(tm.get_term(n)[0])]
        z = float(robot.data.root_link_pos_w[0, 2])
        v = robot.data.root_link_lin_vel_w[0].tolist()
        total = math.nan
        if sensor is not None:
            f = sensor.data.force[0].reshape(-1, 3)
            total = float(torch.linalg.norm(f, dim=1).sum())
        ROWS.append([step, round((step + 1) * dt, 4), z, h, *v, total, ";".join(fired)])
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
    F = [r[7] for r in ROWS]
    airborne = [(f < 1.0) if f == f else (zz > STAND_Z + 0.02) for f, zz in zip(F, z)]
    print("\n[probe] ===== summary =====")
    print(f"[probe] steps {len(ROWS)} ({len(ROWS) * dt:.2f} s); base z max {max(z):.3f} (rise {max(z) - STAND_Z:+.3f}), min {min(z):.3f}")
    ends = [(r[1], r[8]) for r in ROWS if r[8]]
    print(f"[probe] episode ends: {len(ends)} -> " + ", ".join(f"t={t:.2f}s {w}" for t, w in ends))
    air_time = sum(1 for a in airborne if a) * dt
    print(f"[probe] airborne {air_time:.2f} s of {len(ROWS) * dt:.2f} s")
    apex = []
    for k in range(1, len(z) - 1):
        if airborne[k] and z[k] >= z[k - 1] and z[k] > z[k + 1] and z[k] > STAND_Z + 0.005:
            if apex and ROWS[k][1] - apex[-1][0] < 0.08:
                if z[k] > apex[-1][1]:
                    apex[-1] = (ROWS[k][1], z[k])
                continue
            apex.append((ROWS[k][1], z[k]))
    print(f"[probe] airborne apexes: {len(apex)}")
    for i, (t, h) in enumerate(apex):
        gap = f"  interval {t - apex[i - 1][0]:.3f} s" if i else ""
        print(f"[probe]   t={t:6.3f}s  z={h:.3f} (rise {h - STAND_Z:+.3f}){gap}")
    # height tracking: how much of the commanded swing the body actually did
    cmd_h = [r[3] for r in ROWS]
    post = [(c, zz) for c, zz, r in zip(cmd_h, z, ROWS) if r[1] >= START + 1.0]
    if post:
        c_span = max(c for c, _ in post) - min(c for c, _ in post)
        z_span = max(zz for _, zz in post) - min(zz for _, zz in post)
        print(f"[probe] after settling: commanded height swing {c_span * 100:.1f} cm, base z swing {z_span * 100:.1f} cm, "
              f"base z mean {sum(zz for _, zz in post) / len(post):.3f}")
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "t", "base_z", "height_cmd", "vx", "vy", "vz", "foot_force_total", "terminations"])
            w.writerows(ROWS)
        print(f"[probe] log -> {out_path}")


RENDER = opt("--render", os.environ.get("DANCE_RENDER") or None, str)
if RENDER:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_sim
    render_sim.install(play, RENDER, fps=int(opt("--fps", 30, int)),
                       cam=CAM or (1.2, -18.0, 135.0), mask="--no-mask" not in mine, slow=opt("--slow", 1.0))
sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
try:
    play.main()
except SystemExit as e:
    print(f"[probe] play.main exited with {e.code}")
finally:
    summary()
