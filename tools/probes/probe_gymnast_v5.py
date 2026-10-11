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

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
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
# the ending (the coach, 10/11: "最後の 1〜2 回の跳びで空中ずっと片手を挙げる、にっこり笑顔で"):
#   --wave-after T : from T s on, the right claw is up for the whole of every flight (takeoff to touchdown)
#   --finish-after T2 : the first touchdown after T2 ends the bouncing: knees held bent for --finish-hold s to
#                       take the landing, then the standing pose; with --wave-stay the claw stays up meanwhile
WAVE_AFTER = opt("--wave-after", float("inf"))
FINISH_AFTER = opt("--finish-after", float("inf"))
FINISH_HOLD = opt("--finish-hold", 0.6)
WAVE_STAY = "--wave-stay" in mine
# the trampolinist's "kill": while the bed pushes the body back up, fold the legs (knees and thighs) by this much
# more, so the bed's return goes into folding legs instead of into another bounce (action units, 0 = off)
FINISH_KILL = opt("--finish-kill", 0.0)
FINISH_KILL_THIGH = opt("--finish-kill-thigh", 1.0)   # the thighs' share of the fold (0 = knees only: folding the
                                                      # thighs too let the body bottom out on the bed, 10/11)
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

    if FALL_LIMIT_DEG is not None and "fell_over" in terms:
        tm.get_term_cfg("fell_over").params["limit_angle"] = math.radians(FALL_LIMIT_DEG)
        print(f"[gym] fell_over limit {FALL_LIMIT_DEG:g} deg from the start")
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
        on_floor = bool(META.get("dropped")) and on_bed
        if phase != "FLOOR" and on_floor:
            phase, t_phase = "FLOOR", t                 # landed outside the circle, on the gym floor
            print(f"[gym] landed on the stool at t={t:.2f}s")
        if phase != "FINISH" and phase != "FLOOR" and t >= FINISH_AFTER and on_bed and phase == "DOWN":
            phase, t_phase = "FINISH", t
        if phase in ("FINISH", "FLOOR"):
            pass
        elif phase == "STAND":
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
        elif phase == "DOWN" or (phase == "FINISH" and t - t_phase < FINISH_HOLD) or                 (phase == "FLOOR" and t - t_phase < FLOOR_HOLD):
            for leg, (j, s) in KNEE_EXT.items():
                act[0, j] -= s * FLEX                         # knees bent, held (stiff by the PD)
        if phase == "FINISH" and LAUNCH_SWAP and t >= LAUNCH_AT - 0.25 and not META.get("launched"):
            q = robot.data.root_link_quat_w[0]
            qw, qx, qy, qz = (float(v) for v in q)
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            ox_ = float(robot.data.root_link_pos_w[0, 0]) - _rb_cx; oy_ = float(robot.data.root_link_pos_w[0, 1]) - _rb_cy
            nrm = math.hypot(ox_, oy_) or 1.0
            obx = (math.cos(yaw) * ox_ + math.sin(yaw) * oy_) / nrm
            oby = (-math.sin(yaw) * ox_ + math.cos(yaw) * oy_) / nrm
            pushing = t >= LAUNCH_AT and on_bed and (META.get("swap_push") or vz > VZ_UP)
            if pushing:
                META["swap_push"] = True
            if META.get("swap_push") and not on_bed:
                META["launched"] = t
                print(f"[gym] swap launch at t={t:.2f}s")
                if NO_FALL_TERM and "fell_over" in terms:
                    tm.get_term_cfg("fell_over").params["limit_angle"] = math.pi
                    print("[gym]   fell_over limit raised to 180 deg from here (video take)")
            elif on_bed:
                lp = AMP * LAUNCH_GAIN
                for leg, (j, sg) in KNEE_EXT.items():
                    w = _HIP[leg][0] * obx + _HIP[leg][1] * oby          # + outward side, - inward side
                    jt, st_ = THIGH_DOWN[leg]
                    if META.get("swap_push"):
                        # outward legs fold, inward legs drive
                        u = lp * max(0.0, -w) - LAUNCH_SWAP * max(0.0, w)
                        act[0, j] = sg * u
                        act[0, jt] = st_ * u * THIGH
                        if LAUNCH_ROW:
                            hx, hy = _HIP[leg]
                            act[0, _J0[leg]] = LAUNCH_ROW * (-hy * obx + hx * oby)
                    elif vz <= VZ_UP:
                        # on the way down: outward legs stretched, inward legs bent
                        act[0, j] = sg * (-FLEX + LAUNCH_SWAP * w)
        if phase == "FINISH" and LAUNCH_LEAN and t >= LAUNCH_AT - 0.25 and not META.get("launched")                 and not META.get("launching") and on_bed and vz <= VZ_UP:
            q = robot.data.root_link_quat_w[0]
            qw, qx, qy, qz = (float(v) for v in q)
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            ox_ = float(robot.data.root_link_pos_w[0, 0]) - _rb_cx; oy_ = float(robot.data.root_link_pos_w[0, 1]) - _rb_cy
            nrm = math.hypot(ox_, oy_) or 1.0
            obx = (math.cos(yaw) * ox_ + math.sin(yaw) * oy_) / nrm
            oby = (-math.sin(yaw) * ox_ + math.cos(yaw) * oy_) / nrm
            for leg, (j, sg) in KNEE_EXT.items():
                w = _HIP[leg][0] * obx + _HIP[leg][1] * oby          # + on the outward side
                act[0, j] = -sg * (FLEX + LAUNCH_LEAN * w)
        if phase == "FINISH" and t >= LAUNCH_AT and not META.get("launched") and not LAUNCH_SWAP:
            if on_bed and vz > VZ_UP:
                META["launching"] = True
            if META.get("launching") and not on_bed:
                print(f"[gym] launched at t={t:.2f}s")
                if LAUNCH_REPEAT:
                    META["launching"] = False
                else:
                    META["launched"] = t
            if META.get("launching") and on_bed:
                # the outward direction (from the trampoline's centre to the base) in the body frame
                q = robot.data.root_link_quat_w[0]
                qw, qx, qy, qz = (float(v) for v in q)
                yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                bx_, by_ = float(robot.data.root_link_pos_w[0, 0]), float(robot.data.root_link_pos_w[0, 1])
                ox_, oy_ = bx_ - _rb_cx, by_ - _rb_cy
                nrm = math.hypot(ox_, oy_) or 1.0
                ox_, oy_ = ox_ / nrm, oy_ / nrm
                obx = math.cos(yaw) * ox_ + math.sin(yaw) * oy_
                oby = -math.sin(yaw) * ox_ + math.cos(yaw) * oy_
                lp = AMP * LAUNCH_GAIN
                for leg, (j, sg) in KNEE_EXT.items():
                    w = -(_HIP[leg][0] * obx + _HIP[leg][1] * oby)
                    k_ = max(0.0, 1.0 + LAUNCH_TILT * w)
                    act[0, j] = sg * lp * k_
                    jt, st_ = THIGH_DOWN[leg]
                    act[0, jt] = st_ * lp * k_ * THIGH
                    if LAUNCH_ROW:
                        hx, hy = _HIP[leg]
                        act[0, _J0[leg]] = LAUNCH_ROW * (-hy * obx + hx * oby)
        if phase == "FINISH" and FINISH_KILL > 0 and on_bed and vz > 0 and not META.get("launching")                 and not (LAUNCH_SWAP and t >= LAUNCH_AT - 0.25 and not META.get("launched")):
            for leg, (j, s) in KNEE_EXT.items():
                act[0, j] -= s * FINISH_KILL
            for leg, (j, s) in THIGH_DOWN.items():
                act[0, j] -= s * FINISH_KILL * FINISH_KILL_THIGH
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
        if META.get("launched") and in_air and LAUNCH_SWAP and (AIR_SPREAD or AIR_FOLD):
            g = robot.data.projected_gravity_b[0]
            gx_, gy_ = float(g[0]), float(g[1])
            tilt = math.degrees(math.asin(min(1.0, math.hypot(gx_, gy_))))
            if tilt > AIR_TILT_DEG:
                lx_, ly_ = gx_ / (math.hypot(gx_, gy_) or 1.0), gy_ / (math.hypot(gx_, gy_) or 1.0)   # toward the low side
                q = robot.data.root_link_quat_w[0]
                qw, qx, qy, qz = (float(v) for v in q)
                yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
                ox_ = float(robot.data.root_link_pos_w[0, 0]) - _rb_cx; oy_ = float(robot.data.root_link_pos_w[0, 1]) - _rb_cy
                nrm = math.hypot(ox_, oy_) or 1.0
                obx = (math.cos(yaw) * ox_ + math.sin(yaw) * oy_) / nrm
                oby = (-math.sin(yaw) * ox_ + math.cos(yaw) * oy_) / nrm
                for leg, (j, sg) in KNEE_EXT.items():
                    hx, hy = _HIP[leg]
                    w_low = hx * lx_ + hy * ly_
                    if AIR_SPREAD and w_low > 0:
                        li, ls = LIFT[leg]
                        act[0, li] = ls * AIR_SPREAD * w_low        # up from the root: the low side opens like a hand
                        act[0, j] = sg * AIR_SPREAD * w_low         # and straight
                    if AIR_FOLD and tilt > AIR_FOLD_DEG and (hx * obx + hy * oby) < 0:
                        jt, st_ = THIGH_DOWN[leg]
                        act[0, jt] = -st_ * AIR_FOLD                # the pushing (inward) legs fold at the root
                        act[0, j] = -sg * AIR_FOLD
        if HAND_DELAY is not None:
            hs = META.setdefault("hand", {"high": False, "t_land": None, "n": 0})
            if in_air:
                if hs["t_land"] is not None:                      # a new flight: judge it afresh
                    hs["high"], hs["t_land"] = False, None
                if z - STAND_Z > HAND_TRIGGER and not hs["high"]:
                    hs["high"] = True
                    if HAND_FREE_FALL and "fell_over" in terms:
                        tm.get_term_cfg("fell_over").params["limit_angle"] = math.pi
                        hs["free_until"] = None
                if hs["high"]:
                    for idx in (1, 5):
                        act[0, idx] = -HAND_UP                    # claws up: the feet land first
                    act[0, 2] = +2.0; act[0, 6] = -2.0
            elif hs["high"]:
                if hs["t_land"] is None:
                    hs["t_land"] = t
                    hs["n"] += 1
                if t - hs["t_land"] < HAND_DELAY:
                    for idx in (1, 5):
                        act[0, idx] = -HAND_UP
                    act[0, 2] = +2.0; act[0, 6] = -2.0
                else:
                    if HAND_FREE_FALL and hs.get("free_until") is None:
                        hs["free_until"] = t + HAND_FREE_AFTER
                    for idx in (1, 5):
                        act[0, idx] = +HAND_DOWN                  # then the hands come down onto the bed
                    act[0, 2] = +2.0; act[0, 6] = -2.0
                    if not hs.get("logged_" + str(hs["n"])):
                        hs["logged_" + str(hs["n"])] = True
                        if hs["n"] <= 12:
                            print(f"[gym] hands down at t={t:.2f}s ({HAND_DELAY:g} s after the feet)")
        if HAND_FREE_FALL and META.get("hand", {}).get("free_until") is not None and t >= META["hand"]["free_until"]:
            tm.get_term_cfg("fell_over").params["limit_angle"] = math.radians(50.0)
            META["hand"]["free_until"] = None
        if META.get("launched") and in_air and LAUNCH_SWAP and AIR_FLIP:
            q = robot.data.root_link_quat_w[0]
            qw, qx, qy, qz = (float(v) for v in q)
            yaw = math.atan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz))
            ox_ = float(robot.data.root_link_pos_w[0, 0]) - _rb_cx; oy_ = float(robot.data.root_link_pos_w[0, 1]) - _rb_cy
            nrm = math.hypot(ox_, oy_) or 1.0
            obx = (math.cos(yaw) * ox_ + math.sin(yaw) * oy_) / nrm
            oby = (-math.sin(yaw) * ox_ + math.cos(yaw) * oy_) / nrm
            since = t - META["launched"]
            for leg, (j, sg) in KNEE_EXT.items():
                hx, hy = _HIP[leg]
                if (hx * obx + hy * oby) < 0:                     # the legs that pushed (inward side)
                    jt, st_ = THIGH_DOWN[leg]
                    u = -0.5 * AIR_FLIP if since < AIR_FLIP_BACK else AIR_FLIP
                    act[0, jt] = st_ * u
                    act[0, j] = sg * u
        if t >= WAVE_AFTER and (in_air or (phase in ("FINISH", "FLOOR") and WAVE_STAY)):
            act[0, 5] = -3.0                                  # the right claw up for the whole flight
            act[0, 6] = -2.0                                  # elbow straight
        if phase == "FLOOR" and not in_air and t - t_phase >= FLOOR_HOLD:
            act[0, 5] = -HELLO_LIFT; act[0, 6] = -2.0         # hello: the right claw up, elbow straight,
            act[0, 4] = HELLO_AMP * math.sin(2 * math.pi * HELLO_HZ * (t - t_phase - FLOOR_HOLD))   # waved side to side
        if NOSE_WAVE == "landing" and in_air and landing_soon and float(robot.data.projected_gravity_b[0, 0]) > NOSE_WAVE_G:
            for idx in (1, 5):
                act[0, idx] = -3.0
            act[0, 2] = +2.0; act[0, 6] = -2.0
        if RIGHT is not None:
            gz_ = float(robot.data.projected_gravity_b[0, 2])
            rs = META.setdefault("right", {"prev_vz": 0.0, "t_bottom": None, "n": 0})
            if gz_ < -0.7:
                rs["k"] = 0                                         # upright again: the next fall starts afresh
            if gz_ > RIGHT_GZ:
                if rs["prev_vz"] < 0.0 <= vz:                       # the bottom of a bounce on the back
                    rs["t_bottom"] = t
                    rs["n"] += 1
                    rs["k"] = rs.get("k", 0) + 1                    # bottoms in this fall
                    if rs["n"] <= 20:
                        print(f"[gym] upside down, bottom at t={t:.2f}s: {RIGHT}")
                since_b = None if rs["t_bottom"] is None else t - rs["t_bottom"]
                for leg, (j, sg) in KNEE_EXT.items():
                    jt, st_ = THIGH_DOWN[leg]
                    li, ls = LIFT[leg]
                    mode = RIGHT if RIGHT != "combo" else ("extend" if rs.get("k", 0) <= 2 else "curl")
                    if mode == "extend":
                        if since_b is not None and since_b < RIGHT_DUR:
                            act[0, j] = sg * RIGHT_U                 # knee straight
                            act[0, jt] = st_ * RIGHT_U               # the leg out along the underside's normal
                        else:
                            act[0, j] = -sg * RIGHT_U * 0.6          # folded again in the air
                            act[0, li] = ls * RIGHT_U * 0.6
                    elif mode == "curl":
                        if since_b is not None:
                            act[0, li] = ls * RIGHT_U                # thighs up toward the back: knees reach the mat
                            act[0, j] = -sg * RIGHT_U
            rs["prev_vz"] = vz
        if ROUND_BED and not META.get("dropped") and not on_bed and                 math.hypot(float(robot.data.root_link_pos_w[0, 0]) - _rb_cx, float(robot.data.root_link_pos_w[0, 1]) - _rb_cy) > _rb_r:
            _drop_the_floor(unwrapped)
            META["dropped"] = t
            print(f"[gym] airborne outside the circle at t={t:.2f}s: the floor is now the landing mat (the stool)")
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
                     float(robot.data.root_link_pos_w[0, 0]), float(robot.data.root_link_pos_w[0, 1]),
                     float(robot.data.projected_gravity_b[0, 2]), float(robot.data.projected_gravity_b[0, 0]),
                     float(robot.data.projected_gravity_b[0, 1])])
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
            w.writerow(["step", "t", "base_z", "vz", "foot_force_total", "phase", "push", "tilt_rate", "yaw_rate", "u_pitch", "u_roll", "u_yaw", "terminations", "base_x", "base_y", "gz", "gx", "gy"])
            w.writerows(ROWS)
        print(f"[gym] log -> {out_path}")


# --round-bed R,CX,CY (the coach, 10/11: "飛び出した先は少し低い床に着地できるの？" "硬さは吸着" "着地する場所に
# ふかふかの切り株スツールを 0 地点で"): the round trampoline's edge, made physical the simple way. The bed stays
# the endless soft plane it always was, and the moment the crab is airborne with its base outside the circle of
# radius R about (CX, CY), the plane's contact becomes a landing mat that gives nothing back: the feet's own
# stiffness, over-damped (--floor-damp > 1) and a firm grip (--floor-friction). Its height stays 0: render_sim
# draws the island's stump stool there (STOOL=x,y,r), top at the bed's level. Everything before that moment is
# the take exactly. Tried and dropped (10/11): a cylinder of bed (its convex collision with the feet does not
# sink like the plane; the crab barely bounced, 2-4 cm), and lowering the plane 0.25 m at the moment (the ground
# is a static geom and its world position is not recomputed from geom_pos; it stayed at 0).
ROUND_BED = opt("--round-bed", None, str)
FLOOR_DAMP = opt("--floor-damp", 3.0)
FLOOR_FRICTION = opt("--floor-friction", 2.0)
# after landing on the gym floor: knees take it for --finish-hold, then stand and wave the raised right claw
# side to side (front-right hip yaw J0) at --hello-hz with amplitude --hello-amp (action units)
# --launch-at T (the coach, 10/11: "もっと遠くに吹っ飛べない？"): during the finish, the first bottom of the bed after
# T gets one more push, --launch-gain x AMP, leaning outward: each leg pushes by (1 + --launch-tilt x w), w being how
# far its hip points INTO the trampoline (away from where the crab should fly), so the bed throws it off the edge
LAUNCH_AT = opt("--launch-at", float("inf"))
LAUNCH_GAIN = opt("--launch-gain", 1.0)
LAUNCH_TILT = opt("--launch-tilt", 1.0)
# --launch-row U: also row with the hips while the feet are down: each leg's J0 by U x (its hip's tangent . the
# outward direction), so every foot sweeps toward the trampoline's centre and the body slides out (same-sign J0 on
# all legs turns the body; this sign pattern is meant to cancel the turn and keep the slide). Sign unknown: try +/-
LAUNCH_ROW = opt("--launch-row", 0.0)
LAUNCH_REPEAT = "--launch-repeat" in mine
# --launch-lean L: the long jumper's way. While the bed is still taking the crab down before the launch (vz < 0),
# legs on the outward side bend by L and those on the inward side stretch by L, so the body leans out; the push
# then goes along the leaning body, through the centre of mass, instead of twisting it
LAUNCH_LEAN = opt("--launch-lean", 0.0)
# --launch-swap S (the coach, 10/11: "着地の時に移動したい方向の足を伸ばして反対の足を曲げる。跳ねるときに、伸ばした足を
# 縮めて、曲げた足を伸ばす"): on the bed's way down before the launch, the legs on the outward side stretch by S and
# the inward ones bend by S (the body tips in, loading the inward legs); at the push the roles swap: the outward
# legs fold by S and the inward ones extend with the launch push (gain x AMP), throwing the body outward
LAUNCH_SWAP = opt("--launch-swap", 0.0)
# after the swap launch, in the air (the coach: "空中で斜めになったら、下がった足を広げる。進行方向の足をパッと広げて、
# 空中でパーになる。それでも戻らなければ、ジャンプの時に伸ばした足を根元から縮める"):
#   --air-spread P : legs on the low side lift from the root and straighten, by P x how far they point down-tilt
#   --air-fold F   : once the tilt passes --air-fold-deg, the legs that drove the push (inward side) fold at the root
AIR_SPREAD = opt("--air-spread", 0.0)
AIR_FOLD = opt("--air-fold", 0.0)
AIR_FOLD_DEG = opt("--air-fold-deg", 20.0)
AIR_TILT_DEG = opt("--air-tilt-deg", 5.0)
# --air-flip F (the coach: "それでも治らなければさらに回転して宙返り。飛び出したときに曲がってる足はそのまま、伸ばしてる足を
# 半分ぐらい上げ戻して、全速力でもう一回開く"): after the swap launch, the legs that drove the push (inward side)
# fold back half way at the root for --air-flip-back s, then open again as fast as they can (F action units)
AIR_FLIP = opt("--air-flip", 0.0)
AIR_FLIP_BACK = opt("--air-flip-back", 0.08)
# --no-fall-term: for a video take, from the launch on, the env's fell_over termination (50 deg of tilt, a training
# rule that resets the robot to its start) no longer fires: its limit is raised to 180 deg at the launch. The physics
# is unchanged and everything before the launch is the take exactly
NO_FALL_TERM = "--no-fall-term" in mine
# --fall-limit-deg D: fell_over's limit from the first step (180 = never; the coach, 10/11: "最初から180度にして、
# どれだけ転倒してもいいから60秒回し続ける")
FALL_LIMIT_DEG = opt("--fall-limit-deg", None)
# --hand-delay DT (the coach: "足と手が0.2秒くらいの差で着く。足が先について、その0.2秒後に手がつく。高さが20センチを
# 超えたあたりでそれが作動するように。そしたら後ろ宙返りができる気がする"): in a flight whose base rises more than
# --hand-trigger-cm above standing, both claws stay up (front J1 lifted, elbows straight) until the feet touch, and
# come down to the bed DT after the touchdown (front J1 down/forward by --hand-down), until the next takeoff
HAND_DELAY = opt("--hand-delay", None)
HAND_TRIGGER = opt("--hand-trigger-cm", 20.0) / 100.0
HAND_UP = opt("--hand-up", 3.0)
HAND_DOWN = opt("--hand-down", 3.0)
# --hand-free-fall: fell_over's limit is 180 deg only while the hand move runs (from the 20 cm trigger until
# --hand-free-after s after the hands come down), so a back somersault is not cut short; outside it, the usual 50
HAND_FREE_FALL = "--hand-free-fall" in mine
HAND_FREE_AFTER = opt("--hand-free-after", 0.8)
# --right MODE (the coach, 10/11: "ひっくり返った時は最初の2バウンドが勝負。背中が沈みきった段階で全速力で背伸び（足を
# 体から垂直に突き上げる）。空中でまた曲げて、1番沈みきったとこで背伸び。効果がなかったら、沈みきったとこで全部を曲げて
# 足の関節がマットについたら多分成功"): while upside down (projected gravity z above --right-gz), at the bottom of
# each bounce on the back (vertical velocity turning from down to up):
#   extend : every leg straight and pointing away from the body's underside, at full effort (--right-u), for
#            --right-dur s; folded again in the air
#   combo  : the coach's order: extend at the first two bottoms of a fall, curl from the third on
#   curl   : every leg folded hard (thighs lifted, knees bent) so the knees reach for the mat, held to the next bottom
RIGHT = opt("--right", None, str)
RIGHT_U = opt("--right-u", 6.0)
RIGHT_DUR = opt("--right-dur", 0.15)
RIGHT_GZ = opt("--right-gz", 0.3)   # every bottom of the bed after T, not only the first (a run-up)
_J0 = {"LF": 0, "RF": 4, "LM": 8, "RM": 11, "LR": 14, "RR": 17}
_HIP = {"LF": (0.7, 0.7), "RF": (0.7, -0.7), "LM": (0.0, 1.0), "RM": (0.0, -1.0), "LR": (-0.7, 0.7), "RR": (-0.7, -0.7)}
FLOOR_HOLD = opt("--floor-hold", 0.8)    # s of bent knees after landing on the gym floor, before the hello
HELLO_HZ = opt("--hello-hz", 1.5)
HELLO_LIFT = opt("--hello-lift", 6.0)      # standing, the claw needs more than the -3 that raises it in flight
HELLO_AMP = opt("--hello-amp", 1.5)
_rb_r = _rb_cx = _rb_cy = _rb_h = 0.0
if ROUND_BED:
    _rb_r, _rb_cx, _rb_cy = (float(v) for v in ROUND_BED.split(",")[:3])


def _drop_the_floor(unwrapped):
    """Give the ground plane the landing mat's contact, in every model the sim steps."""
    import mujoco as _mj
    import re as _re
    import scenes.registry as _reg
    from tasks.jumper.common.constants import _CONTACT_SOLIMP, _CONTACT_SOLREF
    sim = unwrapped.sim
    models = [sim.mj_model] + list(getattr(sim, "_models", None) or [])
    pats = [_re.compile(p_) for p_ in _reg.GROUND]
    for m in models:
        for gid in range(m.ngeom):
            name = _mj.mj_id2name(m, _mj.mjtObj.mjOBJ_GEOM, gid) or ""
            if any(p_.search(name) for p_ in pats):
                m.geom_solref[gid] = [float(_CONTACT_SOLREF[0]), FLOOR_DAMP]
                m.geom_solimp[gid] = list(_CONTACT_SOLIMP)
                m.geom_friction[gid, 0] = FLOOR_FRICTION
                print(f"[gym]   floor geom {name!r}: solref {m.geom_solref[gid].tolist()}, "
                      f"friction {m.geom_friction[gid, 0]:g}")
    # the native backend keeps the domain-randomised fields in per-environment buffers and writes them back into
    # the models whenever they may have changed, so a field edited only in the models reverts: edit the buffers too
    bufs = getattr(sim, "_expanded_buf", {}) or {}
    m0 = sim.mj_model
    for f in ("geom_solref", "geom_solimp", "geom_friction"):
        if f in bufs:
            bufs[f][:] = getattr(m0, f)
    if hasattr(sim, "_model_dirty"):
        sim._model_dirty = True
    print(f"[gym]   sim {type(sim).__name__}, {len(models)} model(s); per-env buffers rewritten: "
          f"{[f for f in ('geom_solref', 'geom_solimp', 'geom_friction') if f in bufs]}")


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
