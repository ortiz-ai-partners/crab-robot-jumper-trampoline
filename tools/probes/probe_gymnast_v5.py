#!/usr/bin/env python3
"""The gymnast: a fully scripted trampoline bounce, no policy, from standstill.

Everything the coach asked for in one scripted controller on top of the PD servos (zero
actions = HOME):

  stance    standing still for --settle s
  DOWN      going down (airborne or compressing the bed): all six legs held in the same
            flexed, stiff pose (knees bent, "one stick"), ready for the bed. Legs always
            flex and extend TOGETHER (同じタイミングで屈伸).
  UP        from the bed's bottom (base vertical velocity turns upward): extend every leg
            from the root (thigh J1 down, knee J2 straight) with amplitude --amp (ramped up
            over --ramp s from the first push), and --arms-delay s later thrust the front
            arms forward (shoulder J1 +, elbow straight) for --arms-dur s.
  flight    R1 air posture: swing the root joints against the tilt (rear legs up with the
            knee extended for a low nose, front arms forward-down), bang-bang; yaw reflex:
            all hip-yaw joints against the yaw rate; both off just before landing.
            Optional --wave: after the apex, raise the right claw (the greeting).

Logs base z, vz, foot force, phase, tilt and yaw rate per step and prints apex statistics.

    python -I probe_gymnast.py --amp 4.0 --arms 2.0 --out g.csv [--render g.mp4 --slow 3] -- \
        --task jumper.posture --agent zero --scene trampoline --backend native --device cpu --headless --steps 3000 --speed 0
"""
from __future__ import annotations

import csv
import math
import os
import sys

REPO = r"D:\IslandMaster\jumper"
STAND_Z = 0.09423387151092881
# action indices: LF 0-3, RF 4-7, LM 8-10, RM 11-13, LR 14-16, RR 17-19
KNEE_EXT = {"LF": (2, +1.0), "RF": (6, -1.0), "LM": (10, +1.0), "RM": (13, -1.0), "LR": (16, +1.0), "RR": (19, -1.0)}
THIGH_DOWN = {"LF": (1, +1.0), "RF": (5, +1.0), "LM": (9, -1.0), "RM": (12, +1.0), "LR": (15, -1.0), "RR": (18, +1.0)}
LIFT = {"LF": (1, -1.0), "RF": (5, -1.0), "LM": (9, +1.0), "RM": (12, -1.0), "LR": (15, +1.0), "RR": (18, -1.0)}
EXT = {"LM": [(10, +1.0)], "RM": [(13, -1.0)], "LR": [(16, +1.0)], "RR": [(19, -1.0)]}
FRONT, REAR, LEFT = ("LF", "RF"), ("LR", "RR"), ("LF", "LM", "LR")
J0_ALL = (0, 4, 8, 11, 14, 17)

if "--" in sys.argv:
    cut = sys.argv.index("--")
    mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
else:
    mine, theirs = [], sys.argv[1:]


def opt(name, default, cast=float):
    return cast(mine[mine.index(name) + 1]) if name in mine else default


out_path = opt("--out", None, str)
AMP = opt("--amp", 4.0)            # push amplitude on the knees (action units, x0.25 rad); thighs get --thigh of it
THIGH = opt("--thigh", 1.0)
FLEX = opt("--flex", 1.5)          # DOWN pose: knees flexed by this much (action units) and held stiff
SETTLE = opt("--settle", 2.0)
RESETTLE = opt("--resettle", 0.5)   # s of standing after a fall's reset before bouncing again
LEGACY = "--legacy-no-reset" in mine  # the second version's behaviour: keep pushing through a fall's reset
                                      # (produces the chain of falls, and the lucky 37.6 cm run, the coach liked)
RAMP = opt("--ramp", 2.0)          # seconds over which the push amplitude grows to AMP ("少しずつ高く")
VZ_UP = opt("--vz-up", 0.0)        # push when vz rises above this (0 = the true bottom)
VZ_DOWN = opt("--vz-down", 0.0)    # flex when vz falls below this while on the bed (the top of a standing bob)
ARMS = opt("--arms", 2.0)          # arm thrust (front J1 +, elbow +), 0 = off
ARMS_DELAY = opt("--arms-delay", 0.02)
ARMS_DUR = opt("--arms-dur", 0.12)
R1 = "--no-r1" not in mine
R1_KP, R1_KD, R1_UMAX, R1_DEAD = opt("--r1-kp", 2.0), opt("--r1-kd", 0.3), opt("--r1-umax", 3.0), opt("--r1-dead", 0.1)
YAW = "--no-yaw" not in mine
YAW_UMAX, YAW_DEAD = opt("--yaw-umax", 3.0), opt("--yaw-dead", 0.3)
OFF_BELOW = opt("--off-below", 0.03)   # m above stand: descending below this, reflexes off and the DOWN pose is held
WAVE = "--wave" in mine            # greeting: raise the right claw after the apex
# "頭が下がった時は両手を wave にしたら着地を耐えられるのでは" (the coach, 10/9 22:40): when the nose is low,
# raise BOTH claws (the wave motion: front J1 toward the lift sign, elbow straight).
#   flight  : replaces R1's front-arm component while airborne
#   landing : only in the last stretch before touchdown (so the claws do not dig in; the four legs take the landing)
NOSE_WAVE = opt("--nose-wave", "off", str)
NOSE_WAVE_G = opt("--nose-wave-g", 0.1)    # projected-gravity x above which the nose counts as low (0.1 = about 6 deg)
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
    print(f"[gym] amp {AMP} thigh x{THIGH} flex {FLEX} ramp {RAMP}s; arms {ARMS} at +{ARMS_DELAY}s for {ARMS_DUR}s; "
          f"R1 {'on' if R1 else 'off'} yaw {'on' if YAW else 'off'} wave {'on' if WAVE else 'off'}")

    obs = env.get_observations()
    zero = policy(obs)
    phase = "STAND"
    t_phase = 0.0
    t_first_push = None
    SETTLE_AT = [SETTLE]
    armed = False
    in_air = False
    apex_passed = False
    step = 0
    while True:
        if viewer is not None and not viewer.is_running:
            break
        if max_steps is not None and step >= max_steps:
            break
        t = step * dt
        F = float(torch.linalg.norm(sensor.data.force[0].reshape(-1, 3), dim=1).sum())
        vz = float(robot.data.root_link_lin_vel_w[0, 2])
        z = float(robot.data.root_link_pos_w[0, 2])
        on_bed = F > 1.0
        landing_soon = (not on_bed) and vz < 0 and z < STAND_Z + OFF_BELOW
        # ---- phases ----
        if phase == "STAND":
            if t >= SETTLE_AT[0]:
                phase, t_phase = "DOWN", t
        elif phase == "DOWN":
            if vz < -0.05:
                armed = True
            if on_bed and armed and vz > VZ_UP and t - t_phase >= 0.04:
                phase, t_phase = "UP", t
                armed = False
                if t_first_push is None:
                    t_first_push = t
            elif on_bed and not armed and t - t_phase > 0.4:
                armed = True                      # standing still: let the first push happen
        elif phase == "UP":
            if (not on_bed) or (vz < VZ_DOWN and t - t_phase >= 0.04):
                phase, t_phase = "DOWN", t
        if not on_bed and not in_air:
            in_air, apex_passed = True, False
        if on_bed:
            in_air = False
        if in_air and vz < 0:
            apex_passed = True
        gain = 1.0 if t_first_push is None else min(1.0, (t - t_first_push) / max(RAMP, 1e-6))
        push = AMP * gain
        # ---- the legs ----
        act = torch.zeros_like(zero)
        if phase == "UP":
            for leg, (j, s) in KNEE_EXT.items():
                act[0, j] += s * push
            for leg, (j, s) in THIGH_DOWN.items():
                act[0, j] += s * push * THIGH
            since = t - t_phase
            if ARMS > 0 and ARMS_DELAY <= since < ARMS_DELAY + ARMS_DUR:
                act[0, 1] += ARMS; act[0, 5] += ARMS          # front J1 +: claws forward/down, the human arm swing
                act[0, 2] += ARMS; act[0, 6] -= ARMS          # elbows straight
        elif phase == "DOWN":
            for leg, (j, s) in KNEE_EXT.items():
                act[0, j] -= s * FLEX                         # knees bent, held (stiff by the PD)
        # ---- flight reflexes ----
        u_pitch = u_roll = u_yaw = 0.0
        if in_air and not landing_soon:
            g = robot.data.projected_gravity_b[0]
            wb = robot.data.root_link_ang_vel_b[0]
            if R1:
                e_p = R1_KP * float(g[0]) + R1_KD * float(wb[1])
                e_r = R1_KP * float(g[1]) - R1_KD * float(wb[0])
                u_pitch = R1_UMAX * (1.0 if e_p > R1_DEAD else -1.0 if e_p < -R1_DEAD else 0.0)
                u_roll = R1_UMAX * (1.0 if e_r > R1_DEAD else -1.0 if e_r < -R1_DEAD else 0.0)
                for leg, (idx, sgn) in LIFT.items():
                    lift = (-u_pitch if leg in FRONT else u_pitch if leg in REAR else 0.0) + (-u_roll if leg in LEFT else u_roll)
                    act[0, idx] += sgn * lift
                    if lift > 0 and leg in EXT:
                        for j, s in EXT[leg]:
                            act[0, j] += s * R1_UMAX * min(1.0, lift / R1_UMAX)
            if YAW:
                wz = float(robot.data.root_link_ang_vel_w[0, 2])
                u_yaw = YAW_UMAX * (1.0 if wz > YAW_DEAD else -1.0 if wz < -YAW_DEAD else 0.0)
                for j in J0_ALL:
                    act[0, j] += u_yaw
            if WAVE and apex_passed:
                act[0, 5] += -3.0                                 # RF J1 toward the lift sign (-): the right claw up
                act[0, 6] -= 2.0                                  # RF elbow straight
            if NOSE_WAVE == "flight" and float(g[0]) > NOSE_WAVE_G:
                # both claws up, overriding R1's front component (which would have pushed them forward/down)
                for idx in (1, 5):
                    act[0, idx] = -3.0
                act[0, 2] = +2.0; act[0, 6] = -2.0
        if NOSE_WAVE == "landing" and in_air and landing_soon and float(robot.data.projected_gravity_b[0, 0]) > NOSE_WAVE_G:
            for idx in (1, 5):
                act[0, idx] = -3.0
            act[0, 2] = +2.0; act[0, 6] = -2.0
        with torch.inference_mode():
            obs, _, _, _ = env.step(act)
        fired = [n for n in terms if bool(tm.get_term(n)[0])]
        if fired and not LEGACY:
            # the episode was reset (a fall): start again from a standing stance, with a short settle
            # and a fresh amplitude ramp, instead of pushing on in whatever phase the script was in
            phase, t_phase, armed, in_air, apex_passed = "STAND", t, False, False, False
            t_first_push = None
            SETTLE_AT[0] = t + RESETTLE
        w = robot.data.root_link_ang_vel_w[0]
        ROWS.append([step, round((step + 1) * dt, 4), z, vz, F, phase, push if phase == "UP" else 0.0,
                     float(torch.linalg.norm(w[:2])), float(w[2]), u_pitch, u_roll, u_yaw, ";".join(fired),
                     float(robot.data.root_link_pos_w[0, 0]), float(robot.data.root_link_pos_w[0, 1])])
        if monitor is not None:
            monitor.update()
        step += 1
    print(f"\n[gym] stopped after {step} steps")


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
    print("\n[gym] ===== summary =====")
    print(f"[gym] {len(ROWS) * dt:.1f} s; base z max {max(z):.3f} (rise {max(z) - STAND_Z:+.3f}), min {min(z):.3f}; airborne {sum(airborne) * dt:.2f} s")
    ends = [(r[1], r[12]) for r in ROWS if r[12]]
    print(f"[gym] episode ends: {len(ends)} -> " + ", ".join(f"t={t:.2f}s {w}" for t, w in ends))
    apex = []
    for k in range(1, len(z) - 1):
        if airborne[k] and z[k] >= z[k - 1] and z[k] > z[k + 1] and z[k] > STAND_Z + 0.005:
            if apex and ROWS[k][1] - apex[-1][0] < 0.08:
                if z[k] > apex[-1][1]:
                    apex[-1] = (ROWS[k][1], z[k])
                continue
            apex.append((ROWS[k][1], z[k]))
    rises = [h - STAND_Z for _, h in apex]
    if rises:
        print(f"[gym] apexes {len(rises)}: mean {sum(rises) / len(rises) * 100:.1f} cm, max {max(rises) * 100:.1f}, "
              f">=20 cm: {sum(r >= 0.20 for r in rises)}, >=25: {sum(r >= 0.25 for r in rises)}, >=30: {sum(r >= 0.30 for r in rises)}")
        # longest run of consecutive apexes >= 0.30 / 0.25 / 0.20
        for thr in (0.30, 0.25, 0.20):
            best = cur = 0
            for r in rises:
                cur = cur + 1 if r >= thr else 0
                best = max(best, cur)
            print(f"[gym]   longest streak of apexes >= {thr * 100:.0f} cm: {best}")
        print("[gym]   apex list (cm): " + " ".join(f"{r * 100:.0f}" for r in rises))
    if out_path:
        with open(out_path, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["step", "t", "base_z", "vz", "foot_force_total", "phase", "push", "tilt_rate", "yaw_rate", "u_pitch", "u_roll", "u_yaw", "terminations", "base_x", "base_y"])
            w.writerows(ROWS)
        print(f"[gym] log -> {out_path}")


SEED = opt("--seed", None, int)
if SEED is not None:
    # play.py has no seed flag; set the env config's seed so a take found headless can be rendered again
    import tasks as _tasks
    _orig_load = _tasks.load_env_cfg

    def _seeded_load(*a, **k):
        cfg = _orig_load(*a, **k)
        cfg.seed = SEED
        return cfg

    _tasks.load_env_cfg = _seeded_load
    print(f"[gym] seed {SEED}")
sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
try:
    play.main()
except SystemExit as e:
    print(f"[gym] play.main exited with {e.code}")
finally:
    summary()
