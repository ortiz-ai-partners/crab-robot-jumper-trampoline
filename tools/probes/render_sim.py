#!/usr/bin/env python3
"""Draw a replay in SIMULATION time, off screen: one frame every 1/fps seconds of
simulated time, whatever the wall clock does. Writes an mp4 (RGB) and, optionally, a
second mp4 that is the robot's mask (white robot on black), for cutting the background.

Works two ways:

1. On its own, around scripts/play.py:
       python render_sim.py --out clip.mp4 [--fps 30 --size 1080x1920 --dist 1.2 --elev -18 --azim 135 --no-mask] -- <play.py arguments, with --headless>

2. As a module, on top of another probe that has already replaced `play._loop`:
       import render_sim; render_sim.install(play, "clip.mp4", ...)
   It wraps `env.step` inside the current `play._loop`, so the probe's own loop keeps
   running and every step it takes is drawn.

How a frame is made (the live viewer's recipe, rl/mjrl/viewer/live.py): the native
backend keeps a `render_model` with the appearance meshes; environment 0's qpos/qvel
are copied into an MjData of that model, `mj_forward`, then `mujoco.Renderer` draws
it with a tracking camera locked on the robot's base. With FACE set, the display module shows
a face that follows the simulation (see face_display.py). The mask comes from the same
renderer in segmentation mode: every geom whose body belongs to the robot's kinematic
tree is the robot.
"""
from __future__ import annotations

import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in


#: The visual bed ("床のテクスチャも沈み込むように", the coach, 10/9 23:40): a height field drawn in place
#: of the flat plane, dished under every foot by its sink depth. Rendering only; the physics stays the
#: flat compliant plane.
DISH_DEPTH = 0.15      # m, the deepest the drawn bed can dish
# m, width of the dish around a foot. 0.08 looked like a mat far too soft (the coach, 10/10: "脚が着いたときに
#影響される幅を 10 倍に、マットが柔らかすぎる"); a real bed under tension sags over a wide area, so 0.8
DISH_SIGMA = float(os.environ.get("DISH_SIGMA", "0.8"))
DISH_GAIN = float(os.environ.get("DISH_GAIN", "1.5"))   # visual exaggeration of the sink (1.0 = as measured)
# m, the drawn bed's surface raised above the physics plane (the coach, 10/10: "床面を 3 cm 上に").
# With the sink exaggerated the dish bottom is drawn below the feet and the robot looks to float; lifting
# the surface puts the feet back on (in) the drawn net. Visual only.
DISH_LIFT = float(os.environ.get("DISH_LIFT", "0.0"))
# "body": ONE dish for the whole crab, deepest under its centre (the coach, 10/10: "カニの中心が一番へこむ、
# カニを一つの存在として凹みが発生するイメージ"). Its depth is set so that, averaged over the feet, the drawn
# bed under each foot is as deep as that foot has sunk. "feet": one dish per foot (the earlier look).
DISH_MODE = os.environ.get("DISH_MODE", "hex")
DISH_HALF = 12.0       # m, half-size of the drawn bed patch (it follows the robot; large so the edge sits at the horizon)
DISH_N = 961           # grid cells per side (2.5 cm cells; an 8 cm dish needs this, 6 cm cells made it invisible)
# "色は白等すぐ例のチェックで！トランポリンの網みたいにしよう" (the coach, 10/9 23:50): the bed is drawn as a
# fine white/grey checker that reads as a woven net at this scale.
NET_RGB1 = (0.66, 0.66, 0.70)   # mid tones: the scene's lights push a white bed into saturation and the dish's shading vanishes
NET_RGB2 = (0.36, 0.38, 0.46)
NET_CELL = 0.025       # m, one checker cell (the weave; fine enough that a dish bends the pattern visibly)
FOOT_BODIES = ("LF_palm_pad_b_link", "RF_palm_pad_b_link", "LM_foot_tip_link", "RM_foot_tip_link",
               "LR_foot_tip_link", "RR_foot_tip_link")
FOOT_RADIUS = 0.012
FOOT_STANCE_Z = -0.012  # m, where a foot tip body sits when standing on the bed (its sink below this is drawn)
# A round trampoline ("ゆかを丸にして丸いトランポリンみたいに", the coach, 10/10). Visual only: the physics bed
# is still the infinite plane, so the circle must contain the robot's path (seed 3 wanders over 0.7 x 1.4 m,
# measured 10/10; falls reset it to the middle). ROUND_R > 0 switches it on: the net becomes a fixed disc of
# that radius centred at (ROUND_CX, ROUND_CY), with a blue pad round the rim, legs, and a gym floor
# ROUND_HEIGHT below the bed.
ROUND_R = float(os.environ.get("ROUND_R", "0"))
ROUND_CX = float(os.environ.get("ROUND_CX", "0"))
ROUND_CY = float(os.environ.get("ROUND_CY", "0"))
ROUND_HEIGHT = float(os.environ.get("ROUND_HEIGHT", "0.25"))
ROUND_SAG = float(os.environ.get("ROUND_SAG", "0.5"))   # share of the depth carried by the whole-bed sag to the frame
HF_Z = DISH_DEPTH                     # the height field's vertical range (elevation 1.0 = the bed surface)
if ROUND_R > 0:
    DISH_HALF = ROUND_R + 0.10        # the disc's bounding square (fixed in the world, not following the robot)
    DISH_N = int(round(2 * DISH_HALF / 0.02)) + 1     # 2 cm cells
    HF_Z = ROUND_HEIGHT + 0.01        # outside the disc the field drops 1 cm below the gym floor and is hidden


#: "hex" mode (the coach, 10/10: "カニを一個の六角形の物体として、足と足の間は同じ凹み具合、物理的に正しい描写で。
#: 床の素材をもうちょっと固くした感じ"): the crab presses the bed like one rigid hexagonal plate.
#:   inside the hexagon of the six feet: the bed follows a plane fitted through the loaded feet's sink
#:     (a level crab presses it evenly; a tilted one, tilted), so between the feet it is as deep as at them;
#:   outside: a tensioned membrane's return to the surface. For a membrane under uniform tension loaded over
#:     a region of radius a and held at radius R, the deflection is w(r) = w_a ln(R/r) / ln(R/a) (Laplace's
#:     equation, radial). Here r = a + d, d the distance from the hexagon, R = a + DISH_FALLOFF. A shorter
#:     falloff is a stiffer bed.
#: The sink is drawn as measured (no exaggeration), so the net touches the feet.
DISH_FALLOFF = float(os.environ.get("DISH_FALLOFF", "0.25"))   # 0.25: the coach's choice, 10/10 ("いまのかたさがいい")


def _convex_hull(points):
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]          # counter-clockwise


def _hex_dish(h, data, foot_ids, bx, by, cell):
    """Write the hexagonal-plate dish into `h` (grid coordinates relative to (bx, by)); return the window."""
    import numpy as np

    feet = []
    for fid in foot_ids:
        fx, fy, fz = (float(v) for v in data.xpos[fid])
        feet.append((fx - bx, fy - by, max(0.0, (FOOT_STANCE_Z - fz) * DISH_GAIN)))
    loaded = [f for f in feet if f[2] > 0.002]
    if not loaded:
        return None
    if len(loaded) >= 3:
        A = np.array([[1.0, x, y] for x, y, _ in loaded])
        b = np.array([s for _, _, s in loaded])
        coef = np.linalg.lstsq(A, b, rcond=None)[0]
    else:
        coef = np.array([float(np.mean([s for _, _, s in loaded])), 0.0, 0.0])
    hull = np.array(_convex_hull([(round(x, 5), round(y, 5)) for x, y, _ in feet]), dtype=float)
    if len(hull) < 3:
        return None
    hx, hy = hull[:, 0], hull[:, 1]
    area = 0.5 * abs(float(np.dot(hx, np.roll(hy, -1)) - np.dot(hy, np.roll(hx, -1))))
    a = max(0.03, (area / np.pi) ** 0.5)
    L = DISH_FALLOFF
    R = a + L
    x0, x1 = hx.min() - L, hx.max() + L
    y0, y1 = hy.min() - L, hy.max() + L
    c0 = max(0, int(np.floor((x0 + DISH_HALF) / cell))); c1 = min(DISH_N, int(np.ceil((x1 + DISH_HALF) / cell)) + 1)
    r0 = max(0, int(np.floor((y0 + DISH_HALF) / cell))); r1 = min(DISH_N, int(np.ceil((y1 + DISH_HALF) / cell)) + 1)
    if ROUND_R > 0 and ROUND_SAG > 0:
        r0, r1, c0, c1 = 0, DISH_N, 0, DISH_N          # the sag reaches the frame: the whole disc
    if r1 <= r0 or c1 <= c0:
        return None
    grid = np.linspace(-DISH_HALF, DISH_HALF, DISH_N)
    X, Y = np.meshgrid(grid[c0:c1], grid[r0:r1])
    inside = np.ones_like(X, dtype=bool)
    best_d = np.full_like(X, np.inf)
    best_qx = np.zeros_like(X)
    best_qy = np.zeros_like(X)
    n = len(hull)
    for i in range(n):
        ax, ay = hull[i]
        ex, ey = hull[(i + 1) % n] - hull[i]
        ln = (ex * ex + ey * ey) ** 0.5
        if ln < 1e-9:
            continue
        inside &= ((X - ax) * ey - (Y - ay) * ex) / ln <= 0.0      # outward normal (ey, -ex) for a CCW polygon
        t = np.clip(((X - ax) * ex + (Y - ay) * ey) / (ln * ln), 0.0, 1.0)
        qx, qy = ax + t * ex, ay + t * ey
        d = np.hypot(X - qx, Y - qy)
        m = d < best_d
        best_d = np.where(m, d, best_d)
        best_qx = np.where(m, qx, best_qx)
        best_qy = np.where(m, qy, best_qy)
    plane_in = coef[0] + coef[1] * X + coef[2] * Y
    plane_edge = coef[0] + coef[1] * best_qx + coef[2] * best_qy
    membrane = np.where(best_d < L, np.log(R / (a + best_d)) / np.log(R / a), 0.0)
    if ROUND_R > 0 and ROUND_SAG > 0:
        # the whole bed sags too ("トランポリン全体がしずむ、今の範囲の延長線上に緩やかに", the coach, 10/10):
        # a membrane is held only at the frame, so the same log solution runs from the hexagon out to the
        # rim, along each ray from the hexagon's centre C; t_rim is where that ray meets the frame
        # (the disc is centred on the field's origin). The near 25 cm return keeps (1 - ROUND_SAG) of the
        # depth, the long sag to the frame carries ROUND_SAG of it, and it is exactly zero at the frame.
        cxh, cyh = float(hx.mean()), float(hy.mean())
        vx, vy = X - cxh, Y - cyh
        s = np.hypot(vx, vy) + 1e-9
        cu = (cxh * vx + cyh * vy) / s
        t_rim = -cu + np.sqrt(np.maximum(cu * cu - (cxh * cxh + cyh * cyh) + ROUND_R * ROUND_R, 0.0))
        ok = t_rim > a * 1.05
        far = np.where(ok, np.log(np.maximum(t_rim, 1e-6) / np.maximum(s, a)) /
                       np.log(np.maximum(t_rim, a * 1.05) / a), 0.0)
        far = np.clip(far, 0.0, 1.0)
        membrane = (1.0 - ROUND_SAG) * membrane + ROUND_SAG * far
    val = np.where(inside, plane_in, plane_edge * membrane)
    h[r0:r1, c0:c1] = np.clip(val, 0.0, DISH_DEPTH)
    return (r0, r1, c0, c1)


def _add_round_trampoline(spec):
    """The frame of a round trampoline round the net disc: blue pad, steel ring, legs, gym floor. Visual only."""
    import math
    import mujoco

    cx, cy, R, H = ROUND_CX, ROUND_CY, ROUND_R, ROUND_HEIGHT
    n = 64
    pad_r = 0.05
    # STOOL=x,y,r (the coach, 10/11): the island's stump stool, pushed against the frame where the crab lands after
    # jumping off; the pad, ring and legs are left out where it stands
    stool = [float(v) for v in os.environ["STOOL"].split(",")] if os.environ.get("STOOL") else None

    def clear_of_stool(x, y, margin):
        return stool is None or math.hypot(x - stool[0], y - stool[1]) > stool[2] + margin

    for i in range(n):
        a0, a1 = 2 * math.pi * i / n, 2 * math.pi * (i + 1) / n
        am = 0.5 * (a0 + a1)
        if not clear_of_stool(cx + (R + 0.04) * math.cos(am), cy + (R + 0.04) * math.sin(am), 0.07):
            continue
        spec.worldbody.add_geom(
            name=f"tramp_pad_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[cx + (R + pad_r * 0.6) * math.cos(a0), cy + (R + pad_r * 0.6) * math.sin(a0), DISH_LIFT + 0.01,
                    cx + (R + pad_r * 0.6) * math.cos(a1), cy + (R + pad_r * 0.6) * math.sin(a1), DISH_LIFT + 0.01],
            size=[pad_r, 0, 0], rgba=[0.10, 0.34, 0.86, 1.0], contype=0, conaffinity=0)
        spec.worldbody.add_geom(
            name=f"tramp_ring_{i}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[cx + (R + 0.06) * math.cos(a0), cy + (R + 0.06) * math.sin(a0), -0.045,
                    cx + (R + 0.06) * math.cos(a1), cy + (R + 0.06) * math.sin(a1), -0.045],
            size=[0.012, 0, 0], rgba=[0.55, 0.57, 0.60, 1.0], contype=0, conaffinity=0)
    for k in range(6):
        a = 2 * math.pi * k / 6 + math.pi / 6
        x, y = cx + (R + 0.06) * math.cos(a), cy + (R + 0.06) * math.sin(a)
        if not clear_of_stool(x, y, 0.05):
            continue
        spec.worldbody.add_geom(
            name=f"tramp_leg_{k}", type=mujoco.mjtGeom.mjGEOM_CAPSULE,
            fromto=[x, y, -0.045, x * 1.0 + 0.04 * math.cos(a), y + 0.04 * math.sin(a), -H],
            size=[0.012, 0, 0], rgba=[0.55, 0.57, 0.60, 1.0], contype=0, conaffinity=0)
    spec.worldbody.add_geom(
        name="gym_floor", type=mujoco.mjtGeom.mjGEOM_PLANE, pos=[cx, cy, -H], size=[10, 10, 0.1],
        rgba=[0.66, 0.50, 0.34, 1.0], contype=0, conaffinity=0)
    if stool is not None:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import stump_stool
        stump_stool.add_stool(spec, stool[0], stool[1], stool[2], floor_z=-H, top_z=0.004)
        print(f"[render] stump stool at ({stool[0]:+.2f}, {stool[1]:+.2f}), radius {stool[2]} m, top at the bed's level")


def _dished_render_model(unwrapped, model):
    """Add a height-field bed to the scene's MjSpec and compile a render-only model; hide the plane.
    Returns (model, hfield_id, hfield_geom_id, plane_geom_id) or None when the spec is unavailable."""
    import mujoco
    import numpy as np

    spec = getattr(unwrapped.scene, "spec", None)
    if spec is None:
        return None
    # the ground plane's material, so the bed keeps the scene's look
    plane_material = ""
    for g in spec.geoms:
        if g.type == mujoco.mjtGeom.mjGEOM_PLANE:
            plane_material = g.material or ""
            break
    hf = spec.add_hfield(name="bed_dish", size=[DISH_HALF, DISH_HALF, HF_Z, 0.01], nrow=DISH_N, ncol=DISH_N)
    hf.userdata = np.ones(DISH_N * DISH_N, dtype=np.float32)
    # the net: a fine two-tone checker, repeated so one cell is NET_CELL metres across the patch
    tex = spec.add_texture(name="bed_net_tex", type=mujoco.mjtTexture.mjTEXTURE_2D,
                           builtin=mujoco.mjtBuiltin.mjBUILTIN_CHECKER, width=512, height=512)
    tex.rgb1[:] = NET_RGB1
    tex.rgb2[:] = NET_RGB2
    mat = spec.add_material(name="bed_net")
    mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = "bed_net_tex"
    repeat = (2.0 * DISH_HALF) / NET_CELL
    mat.texrepeat[:] = (repeat, repeat)
    mat.texuniform = False
    mat.reflectance = 0.05
    g = spec.worldbody.add_geom(name="bed_dish_geom", type=mujoco.mjtGeom.mjGEOM_HFIELD, hfieldname="bed_dish",
                                pos=[ROUND_CX if ROUND_R > 0 else 0.0, ROUND_CY if ROUND_R > 0 else 0.0,
                                     -HF_Z + DISH_LIFT], contype=0, conaffinity=0)
    if os.environ.get("DISH_DEBUG"):
        g.rgba[:] = (0.8, 0.8, 0.85, 1.0)      # flat grey, so the geometry's shading shows without the texture
    else:
        g.material = "bed_net"
    if ROUND_R > 0:
        _add_round_trampoline(spec)
    _add_face(spec)
    m2 = spec.compile()
    hid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_HFIELD, "bed_dish")
    gid = mujoco.mj_name2id(m2, mujoco.mjtObj.mjOBJ_GEOM, "bed_dish_geom")
    plane = next((i for i in range(m2.ngeom) if m2.geom_type[i] == mujoco.mjtGeom.mjGEOM_PLANE
                  and mujoco.mj_id2name(m2, mujoco.mjtObj.mjOBJ_GEOM, i) != "gym_floor"), -1)
    if plane >= 0:
        m2.geom_rgba[plane, 3] = 0.0          # the dish is drawn instead of the plane
    return m2, hid, gid, plane


#: faces on the display module (the coach, 10/10: "練習中は真剣な目、回っているときはぐるぐる目、転んでいる
#: ときは ><"). FACE=auto|neutral|serious|spiral|cross|happy; unset or 0 = the plain black panel.
FACE = os.environ.get("FACE", "0")
#: CAM_FOLLOW=180 keeps the camera on the crab's face (its front looks along +x at yaw 0, so azimuth = heading + 180);
#: unset = the fixed azimuth as before. CAM_FOLLOW_TAU: seconds of simulation the camera takes to swing round
CAM_FOLLOW = float(os.environ["CAM_FOLLOW"]) if os.environ.get("CAM_FOLLOW") else None
CAM_FOLLOW_TAU = float(os.environ.get("CAM_FOLLOW_TAU", "0.8"))
FACE_HAPPY_AFTER = float(os.environ.get("FACE_HAPPY_AFTER", "inf"))   # s of simulation: ^ ^ from then on
_face = {}


def _add_face(spec):
    if FACE in ("", "0"):
        return
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    import face_display
    _face["geom"], _face["mats"] = face_display.add_screen(spec)


def _faced_render_model(unwrapped):
    """The scene's render model with only the face added (no dished bed)."""
    spec = getattr(unwrapped.scene, "spec", None)
    if spec is None or FACE in ("", "0"):
        return None
    _add_face(spec)
    return spec.compile()


def install(play, out_path, fps=30, width=1080, height=1920, cam=(1.2, -18.0, 135.0), mask=True, slow=1.0, dish=True):
    """`slow` > 1 renders slow motion: frames are taken every 1/(fps*slow) s of simulation and
    played back at `fps`, so slow=3 shows the motion at one third speed (the coach's request,
    10/9: easier to read the movement and the direction of the kinetic energy).
    `dish` draws the bed as a height field that sinks under the feet (see DISH_*)."""
    inner_loop = play._loop
    # RENDER_SIZE=270x480 for a quick look (the face's log and the take's timing are the same at any size)
    if os.environ.get("RENDER_SIZE"):
        width, height = (int(v) for v in os.environ["RENDER_SIZE"].lower().split("x"))
    # RENDER_DISH=0 draws the plain scene floor, as before the net bed existed (for reproducing older clips)
    dish = dish and os.environ.get("RENDER_DISH", "1") != "0"

    def _loop(env, policy, viewer, max_steps, speed=1.0, readout=play._speed_text, monitor=None):
        import imageio_ffmpeg
        import mujoco
        import numpy as np

        unwrapped = getattr(env, "unwrapped", env)
        sim = unwrapped.sim
        model = getattr(sim, "render_model", None) or sim.mj_model
        dished = _dished_render_model(unwrapped, model) if dish else None
        if dished is not None:
            model, hfield_id, hfield_geom, plane_geom = dished
            print(f"[render] bed drawn as a height field that dishes under the feet (depth up to {DISH_DEPTH} m)")
        elif dish:
            print("[render] no MjSpec on the scene; the bed stays a flat plane")
        else:
            faced = _faced_render_model(unwrapped)
            if faced is not None:
                model = faced
        face = None
        if _face:
            import face_display
            face = face_display.FaceDriver(model, _face["geom"], _face["mats"],
                                           mode=FACE if FACE != "1" else "auto")
            print(f"[render] face on the display: {face.mode}"
                  f"{f', base {face.base}' if face.mode == 'auto' else ''}"
                  f"{f', ^ ^ from t={FACE_HAPPY_AFTER:g} s' if FACE_HAPPY_AFTER < float('inf') else ''}")
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        dt = float(unwrapped.step_dt)
        # the robot's bodies carry an attachment prefix in the scene model (e.g. "robot/LR_foot_tip_link"),
        # so match by suffix
        all_bodies = [(i, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i) or "") for i in range(model.nbody)]
        foot_ids = [i for n in FOOT_BODIES for i, name in all_bodies if name == n or name.endswith("/" + n)]
        if dish and len(foot_ids) != len(FOOT_BODIES):
            print(f"[render] dish: found {len(foot_ids)} of {len(FOOT_BODIES)} foot bodies; names look like "
                  f"{[name for _, name in all_bodies if 'foot' in name or 'palm' in name][:3]}")
        grid = np.linspace(-DISH_HALF, DISH_HALF, DISH_N)
        gx, gy = np.meshgrid(grid, grid)            # row = y, col = x (MuJoCo hfield rows run along y)
        round_mask = (gx ** 2 + gy ** 2) <= ROUND_R ** 2 if ROUND_R > 0 else None

        # the robot's base: the body carrying the free joint; everything in its tree is "robot"
        free = int(mujoco.mjtJoint.mjJNT_FREE)
        base = next(b for b in range(model.nbody)
                    if model.body_jntnum[b] > 0 and model.jnt_type[model.body_jntadr[b]] == free)
        robot_geoms = np.array([model.body_rootid[model.geom_bodyid[g]] == base for g in range(model.ngeom)])

        camera = mujoco.MjvCamera()
        camera.type = mujoco.mjtCamera.mjCAMERA_TRACKING
        camera.trackbodyid = base
        camera.distance, camera.elevation, camera.azimuth = cam

        rgb = mujoco.Renderer(model, height=height, width=width)
        seg = None
        if mask:
            seg = mujoco.Renderer(model, height=height, width=width)
            seg.enable_segmentation_rendering()

        writer = imageio_ffmpeg.write_frames(out_path, (width, height), fps=fps, quality=8,
                                             macro_block_size=1, ffmpeg_log_level="error")
        writer.send(None)
        mask_writer = None
        if mask:
            root, ext = os.path.splitext(out_path)
            mask_writer = imageio_ffmpeg.write_frames(root + ".mask" + ext, (width, height), fps=fps,
                                                      quality=8, macro_block_size=1, ffmpeg_log_level="error")
            mask_writer.send(None)
        state = {"step": 0, "frames": 0, "next": 0.0}
        interval = 1.0 / (fps * max(slow, 1e-6))
        print(f"[render] {width}x{height} @ {fps} fps, one frame per {interval * 1000:.1f} ms of simulation"
              f"{f' = {slow:g}x slow motion' if slow != 1 else ''} -> {out_path}"
              f"{' (+ mask)' if mask else ''}; camera distance {cam[0]} elevation {cam[1]} azimuth {cam[2]}")

        def dish_the_bed():
            # move the drawn bed under the robot and dish it where the feet have sunk below the plane
            bx, by = float(data.xpos[base, 0]), float(data.xpos[base, 1])
            if ROUND_R > 0:
                ox, oy = ROUND_CX, ROUND_CY        # the round trampoline stays put; the robot moves over it
            else:
                ox, oy = bx, by                    # the endless bed patch follows the robot
                model.geom_pos[hfield_geom, 0] = bx
                model.geom_pos[hfield_geom, 1] = by
                # keep the drawn bed's world position in step with this frame (mj_forward ran before the move)
                data.geom_xpos[hfield_geom, 0] = bx
                data.geom_xpos[hfield_geom, 1] = by
            # the dish field, updated only in windows around the feet (the full 961x961 grid per frame was 1 s)
            h = state.setdefault("dish_h", np.zeros((DISH_N, DISH_N), dtype=np.float64))
            h[:] = 0.0 if not state.get("dish_touched") else h
            for (r0, r1, c0, c1) in state.get("dish_touched", []):
                h[r0:r1, c0:c1] = 0.0
            touched = []
            cell = (2.0 * DISH_HALF) / (DISH_N - 1)
            win = int(3.0 * DISH_SIGMA / cell) + 1
            # how deep each foot has sunk below where a foot tip sits when standing
            # (measured 10/9 on the physics logs: the tip bodies sit at z = -0.006 .. -0.018 when standing,
            # and reach -0.055 .. -0.087 at the deep-bed landings)
            presses = []
            if DISH_MODE == "hex":
                win_box = _hex_dish(h, data, foot_ids, ox, oy, cell)
                if win_box is not None:
                    touched.append(win_box)
            elif DISH_MODE == "body":
                num = den = 0.0
                for fid in foot_ids:
                    fx, fy, fz = (float(v) for v in data.xpos[fid])
                    sink = max(0.0, (FOOT_STANCE_Z - fz) * DISH_GAIN)
                    w = float(np.exp(-((fx - bx) ** 2 + (fy - by) ** 2) / (2.0 * DISH_SIGMA ** 2)))
                    num += sink
                    den += w
                depth = num / den if den > 1e-6 else 0.0
                if depth > 0.003:
                    presses.append((bx, by, depth))
            else:
                for fid in foot_ids:
                    fx, fy, fz = (float(v) for v in data.xpos[fid])
                    depth = (FOOT_STANCE_Z - fz) * DISH_GAIN
                    if depth > 0.003:
                        presses.append((fx, fy, depth))
            for fx, fy, depth in presses:
                lx, ly = fx - ox, fy - oy
                ci = int(round((lx + DISH_HALF) / cell)); ri = int(round((ly + DISH_HALF) / cell))
                r0, r1 = max(0, ri - win), min(DISH_N, ri + win + 1)
                c0, c1 = max(0, ci - win), min(DISH_N, ci + win + 1)
                if r1 <= r0 or c1 <= c0:
                    continue
                r2 = (gx[r0:r1, c0:c1] - lx) ** 2 + (gy[r0:r1, c0:c1] - ly) ** 2
                h[r0:r1, c0:c1] = np.maximum(h[r0:r1, c0:c1],
                                             min(depth, DISH_DEPTH) * np.exp(-r2 / (2.0 * DISH_SIGMA ** 2)))
                touched.append((r0, r1, c0, c1))
            state["dish_touched"] = touched
            if float(h.max()) > state.get("dish_max", 0.0):
                state["dish_max"] = float(h.max())
                state["dish_frame"] = state["frames"]
                k = int(np.argmax(h))
                state["dish_at"] = (float(gx.ravel()[k]) + ox, float(gy.ravel()[k]) + oy, bx, by)
                if os.environ.get("DISH_DEBUG"):
                    feet_xy = [(round(float(data.xpos[f, 0]), 3), round(float(data.xpos[f, 1]), 3), round(float(data.xpos[f, 2]), 3)) for f in foot_ids]
                    print(f"[render] dish debug frame {state['frames']}: deepest cell world xy "
                          f"({state['dish_at'][0]:.3f}, {state['dish_at'][1]:.3f}), grid index {divmod(k, DISH_N)}, "
                          f"base ({bx:.3f}, {by:.3f}), feet xyz {feet_xy}, hfield geom xpos {data.geom_xpos[hfield_geom].tolist()}")
            state["dish_frames"] = state.get("dish_frames", 0) + (1 if h.max() > 0.005 else 0)
            adr = int(model.hfield_adr[hfield_id])
            hd = model.hfield_data[adr:adr + DISH_N * DISH_N].reshape(DISH_N, DISH_N)
            np.subtract(1.0, h / HF_Z, out=hd, casting="unsafe")
            if ROUND_R > 0:
                hd[~round_mask] = 0.0          # outside the disc: down below the gym floor, out of sight
            # each Renderer owns its own GL context, and an upload goes to whichever context is CURRENT:
            # the mask renderer was created last, so without this the dish went to it and never to the RGB one
            rgb._gl_context.make_current()
            mujoco.mjr_uploadHField(model, rgb._mjr_context, hfield_id)
            if seg is not None:
                seg._gl_context.make_current()
                mujoco.mjr_uploadHField(model, seg._mjr_context, hfield_id)

        def draw():
            q = sim.data.qpos[0]
            v = sim.data.qvel[0]
            data.qpos[:] = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
            data.qvel[:] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            mujoco.mj_forward(model, data)
            # where the base wanders, so a round trampoline can be fitted to the take (island, 10/11)
            bxy = (float(data.xpos[base, 0]), float(data.xpos[base, 1]))
            lo, hi = state.setdefault("base_lo", list(bxy)), state.setdefault("base_hi", list(bxy))
            for i in (0, 1):
                lo[i], hi[i] = min(lo[i], bxy[i]), max(hi[i], bxy[i])
            if CAM_FOLLOW is not None:
                # keep the camera in front of the crab: azimuth = heading + CAM_FOLLOW, low-passed over CAM_FOLLOW_TAU
                qw, qx, qy, qz = (float(v) for v in data.xquat[base])
                yaw = np.degrees(np.arctan2(2 * (qw * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz)))
                target = yaw + CAM_FOLLOW
                cur = state.setdefault("cam_az", target)
                diff = (target - cur + 180.0) % 360.0 - 180.0
                k = min(1.0, interval / max(CAM_FOLLOW_TAU, 1e-6))
                state["cam_az"] = cur + k * diff
                camera.azimuth = state["cam_az"]
            if face is not None:
                t_sim = state["step"] * dt
                if t_sim >= FACE_HAPPY_AFTER:
                    face.mode = "happy"
                face(data, t_sim, base)
            if dished is not None:
                dish_the_bed()
            rgb.update_scene(data, camera=camera)
            frame = rgb.render()
            writer.send(np.ascontiguousarray(frame).tobytes())
            if seg is not None:
                seg.update_scene(data, camera=camera)
                ids = seg.render()[..., 0]
                m = (ids >= 0) & robot_geoms[np.clip(ids, 0, model.ngeom - 1)]
                img = np.repeat((m.astype(np.uint8) * 255)[..., None], 3, axis=2)
                mask_writer.send(np.ascontiguousarray(img).tobytes())
            state["frames"] += 1

        original_step = env.step

        def step(actions):
            out = original_step(actions)
            state["step"] += 1
            t = state["step"] * dt
            while t >= state["next"] - 1e-9:
                draw()
                state["next"] += interval
            return out

        draw()                      # frame 0, the starting pose
        state["next"] = interval
        env.step = step
        try:
            return inner_loop(env, policy, viewer, max_steps, speed, readout, monitor)
        finally:
            env.step = original_step
            writer.close()
            if mask_writer is not None:
                mask_writer.close()
            print(f"\n[render] {state['frames']} frames = {state['frames'] / fps:.2f} s of video for "
                  f"{state['step'] * dt:.2f} s of simulation ({state['step']} control steps) -> {out_path}")
            if "base_lo" in state:
                lo, hi = state["base_lo"], state["base_hi"]
                print(f"[render] base x {lo[0]:+.2f}..{hi[0]:+.2f} m, y {lo[1]:+.2f}..{hi[1]:+.2f} m")
            if face is not None:
                print("[render] face: " + ", ".join(f"{t:.2f}s {e}" for t, e in face.log)
                      + f"  (video time = simulation time x {max(slow, 1e-6):g}); "
                      f"fastest steady turn {face.state.get('peak', 0.0):.2f} rad/s")
            if dished is not None:
                print(f"[render] dish: deepest {state.get('dish_max', 0.0) * 100:.1f} cm at frame {state.get('dish_frame')} "
                      f"(video t={state.get('dish_frame', 0) / fps:.2f} s), world xy {state.get('dish_at')}; drawn deeper than 5 mm in "
                      f"{state.get('dish_frames', 0)} of {state['frames']} frames")

    play._loop = _loop


def _main():
    if "--" in sys.argv:
        cut = sys.argv.index("--")
        mine, theirs = sys.argv[1:cut], sys.argv[cut + 1:]
    else:
        mine, theirs = [], sys.argv[1:]

    def opt(name, default, cast=float):
        return cast(mine[mine.index(name) + 1]) if name in mine else default

    out = opt("--out", "render.mp4", str)
    fps = opt("--fps", 30, int)
    size = opt("--size", "1080x1920", str)
    w, h = (int(x) for x in size.lower().split("x"))
    cam = (opt("--dist", 1.2), opt("--elev", -18.0), opt("--azim", 135.0))
    mask = "--no-mask" not in mine
    slow = opt("--slow", 1.0)
    dish = "--no-dish" not in mine

    os.chdir(REPO)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import play
    install(play, out, fps=fps, width=w, height=h, cam=cam, mask=mask, slow=slow, dish=dish)
    sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
    play.main()


if __name__ == "__main__":
    _main()
