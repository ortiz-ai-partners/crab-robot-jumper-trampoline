#!/usr/bin/env python3
"""Reflex layer, first version (stage 0.5 -> the handover's R2/R3/R4), no training.

Rides on `jumper.posture`: the policy balances and tracks a commanded body height; this
layer decides the height command from the bed's own feedback instead of a clock.
Privileged simulator signals (total foot force, base vertical velocity) are used, as
the handover allows for video; on the robot they become IMU estimates.

State machine (one environment):

    AIR       flying. command h_air (knees bent, ready). -> CONTACT when foot force > f_touch
    CONTACT   bed compressing. hold the command (legs stay at the same length: no absorbing,
              "one stick"). -> PUSH at the bottom: base vertical velocity crosses upward
              (or, with --trigger force, when the foot force has passed its peak)
    PUSH      command h_push (extend). -> AIR when foot force < f_air (took off);
              if still on the bed after --push-max seconds, nothing came of it: back to
              AIR's command to re-arm (a standing re-crouch), then try again.

The first hop is kicked at --start: crouch to h_air for --crouch seconds, then push.

    python -I probe_reflex_bounce.py --h-air 0.09 --h-push 0.15 --out log.csv [--render clip.mp4] -- \
        --task jumper.posture --checkpoint tasks/jumper/posture/out/example/model_74800.pt \
        --scene trampoline --backend native --device cpu --headless --steps 2400 --speed 0
"""
from __future__ import annotations

import csv
import math
import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
STAND_Z = 0.09423387151092881
NEUTRAL_HEIGHT = 0.10647
HEIGHT_RANGE = (0.07, 0.15)

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]


def opt(name, default, cast=float):
    return cast(mine[mine.index(name) + 1]) if name in mine else default


out_path = opt("--out", os.environ.get("REFLEX_OUT") or None, str)
H_AIR = opt("--h-air", float(os.environ.get("REFLEX_H_AIR", "0.09")))
H_PUSH = opt("--h-push", float(os.environ.get("REFLEX_H_PUSH", "0.15")))
F_TOUCH = opt("--f-touch", 5.0)       # N, total foot force that counts as touchdown
F_AIR = opt("--f-air", 1.0)           # N, below this the robot is airborne
START = opt("--start", 1.0)           # s, when the first hop is kicked
CROUCH = opt("--crouch", 0.3)         # s, crouch before the first push
PUSH_MAX = opt("--push-max", 0.35)    # s, give up a push that did not take off
TRIGGER = opt("--trigger", "vz", str)  # "vz": bottom = vertical velocity turns upward; "force": force peak passed
MIN_CONTACT = opt("--min-contact", 0.02)  # s, ignore the bottom test for this long after touchdown
MODE = opt("--mode", "fsm", str)      # "fsm": the state machine above; "vzlock": the self-excited oscillator below
VZ_UP = opt("--vz-up", 0.0)           # vzlock: push when the base's vertical velocity rises above this (negative = before the bottom)
VZ_DOWN = opt("--vz-down", 0.0)       # vzlock: crouch when it falls below this (positive = before the top)
MIN_PHASE = opt("--min-phase", 0.04)  # vzlock: s, no switching sooner than this
# vzlock: two phases locked to the body's own vertical motion, standing or flying.
#   DOWN  command h_air (legs bent, held stiff) while the body is going down; at the bottom
#         (vz > VZ_UP) -> UP.  In flight stays DOWN, so the landing is met with bent, stiff legs.
#   UP    command h_push (extend) while the body is going up; at the top (vz < VZ_DOWN) or as
#         soon as the robot is airborne -> DOWN.
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
    cmd = unwrapped.command_manager.get_term("posture")
    sensor = unwrapped.scene.sensors["feet_ground_contact"]
    META.update(dt=dt, terms=terms)
    print(f"[reflex] h_air {H_AIR} h_push {H_PUSH} touch {F_TOUCH} N air {F_AIR} N trigger {TRIGGER} "
          f"start {START}s crouch {CROUCH}s push-max {PUSH_MAX}s; dt {dt}")

    clamp = "--no-clamp" not in mine   # --no-clamp lets h_push exceed the command's trained range (0.07-0.15)

    def set_height(h: float) -> None:
        if clamp:
            h = min(max(h, HEIGHT_RANGE[0]), HEIGHT_RANGE[1])
        cmd.posture_command[:, 0:3] = 0.0
        cmd.posture_command[:, 3] = h

    def foot_force() -> float:
        f = sensor.data.force[0].reshape(-1, 3)
        return float(torch.linalg.norm(f, dim=1).sum())

    obs = env.get_observations()
    phase = "STAND"          # before the first kick
    h = NEUTRAL_HEIGHT
    set_height(h)
    t_phase = 0.0
    f_peak = 0.0
    kicks = 0
    pushes = 0
    armed = False
    step = 0
    while True:
        if viewer is not None and not viewer.is_running:
            break
        if max_steps is not None and step >= max_steps:
            break
        t = step * dt
        F = foot_force()
        vz = float(robot.data.root_link_lin_vel_w[0, 2])
        # ---- the reflex ----
        if MODE == "vzlock":
            if phase == "STAND":
                if t >= START:
                    phase, t_phase, h = "DOWN", t, H_AIR
                    kicks += 1
            elif phase == "DOWN":
                if vz < -0.05:
                    armed = True          # the body really went down; the next upturn is a bottom
                if armed and F > F_AIR and vz > VZ_UP and t - t_phase >= MIN_PHASE:
                    phase, t_phase, h = "UP", t, H_PUSH
                    pushes += 1
                    armed = False
            elif phase == "UP":
                if F < F_AIR or (vz < VZ_DOWN and t - t_phase >= MIN_PHASE):
                    phase, t_phase, h = "DOWN", t, H_AIR
        elif phase == "STAND":
            if t >= START:
                phase, t_phase, h = "CROUCH", t, H_AIR
                kicks += 1
        elif phase == "CROUCH":
            if t - t_phase >= CROUCH:
                phase, t_phase, h = "PUSH", t, H_PUSH
                pushes += 1
        elif phase == "AIR":
            if F > F_TOUCH:
                phase, t_phase, f_peak = "CONTACT", t, F
        elif phase == "CONTACT":
            f_peak = max(f_peak, F)
            settled = t - t_phase >= MIN_CONTACT
            if TRIGGER == "force":
                bottom = settled and F < 0.8 * f_peak
            else:
                bottom = settled and vz > 0.0
            if bottom:
                phase, t_phase, h = "PUSH", t, H_PUSH
                pushes += 1
        elif phase == "PUSH":
            if F < F_AIR:
                phase, t_phase, h = "AIR", t, H_AIR
            elif t - t_phase >= PUSH_MAX:
                phase, t_phase, h = "CROUCH", t, H_AIR   # it did not take off: re-arm from a crouch
        set_height(h)
        with torch.inference_mode():
            obs, _, _, _ = env.step(policy(obs))
        set_height(h)
        fired = [n for n in terms if bool(tm.get_term(n)[0])]
        z = float(robot.data.root_link_pos_w[0, 2])
        w = robot.data.root_link_ang_vel_w[0]
        tilt_rate = float(torch.linalg.norm(w[:2]))
        ROWS.append([step, round((step + 1) * dt, 4), z, vz, F, phase, h, tilt_rate, ";".join(fired)])
        if monitor is not None:
            monitor.update()
        step += 1
    META.update(kicks=kicks, pushes=pushes)
    print(f"\n[reflex] stopped after {step} steps; {pushes} pushes")


play._loop = _loop
if RENDER:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import render_sim
    render_sim.install(play, RENDER, cam=CAM, mask="--no-mask" not in mine, slow=opt("--slow", 1.0))


def summary() -> None:
    if not ROWS:
        print("[reflex] nothing recorded")
        return
    dt = META["dt"]
    z = [r[2] for r in ROWS]
    F = [r[4] for r in ROWS]
    airborne = [f < F_AIR for f in F]
    print("\n[reflex] ===== summary =====")
    print(f"[reflex] steps {len(ROWS)} ({len(ROWS) * dt:.2f} s); pushes {META.get('pushes')}; "
          f"base z max {max(z):.3f} (rise {max(z) - STAND_Z:+.3f}), min {min(z):.3f}")
    ends = [(r[1], r[8]) for r in ROWS if r[8]]
    print(f"[reflex] episode ends: {len(ends)} -> " + ", ".join(f"t={t:.2f}s {w}" for t, w in ends))
    print(f"[reflex] airborne {sum(airborne) * dt:.2f} s of {len(ROWS) * dt:.2f} s")
    apex = []
    for k in range(1, len(z) - 1):
        if airborne[k] and z[k] >= z[k - 1] and z[k] > z[k + 1] and z[k] > STAND_Z + 0.005:
            if apex and ROWS[k][1] - apex[-1][0] < 0.08:
                if z[k] > apex[-1][1]:
                    apex[-1] = (ROWS[k][1], z[k])
                continue
            apex.append((ROWS[k][1], z[k]))
    print(f"[reflex] airborne apexes: {len(apex)}")
    for i, (t, hh) in enumerate(apex):
        gap = f"  interval {t - apex[i - 1][0]:.3f} s" if i else ""
        print(f"[reflex]   t={t:6.3f}s  rise {hh - STAND_Z:+.3f}{gap}")
    # contact phases: duration and the time from touchdown to push
    print("[reflex] contacts (touchdown -> push -> takeoff):")
    k = 0
    n = len(ROWS)
    shown = 0
    while k < n and shown < 12:
        if ROWS[k][5] == "CONTACT" and (k == 0 or ROWS[k - 1][5] != "CONTACT"):
            t_td = ROWS[k][1]
            j = k
            while j < n and ROWS[j][5] == "CONTACT":
                j += 1
            t_push = ROWS[j][1] if j < n else float("nan")
            m = j
            while m < n and ROWS[m][5] == "PUSH":
                m += 1
            t_off = ROWS[m][1] if m < n else float("nan")
            peak = max(F[k:m + 1]) if m > k else float("nan")
            tilt = max(r[7] for r in ROWS[k:m + 1]) if m > k else float("nan")
            print(f"[reflex]   touchdown {t_td:6.3f}s  bottom/push +{t_push - t_td:.3f}s  takeoff +{t_off - t_td:.3f}s  "
                  f"peak force {peak:.0f} N  max tilt rate {tilt:.2f} rad/s")
            shown += 1
            k = m
        else:
            k += 1
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "t", "base_z", "vz", "foot_force_total", "phase", "height_cmd", "tilt_rate", "terminations"])
            w.writerows(ROWS)
        print(f"[reflex] log -> {out_path}")


sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
try:
    play.main()
except SystemExit as e:
    print(f"[reflex] play.main exited with {e.code}")
finally:
    summary()
