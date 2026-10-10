#!/usr/bin/env python3
"""Faces on the crab's display module (rendering only).

The robot's front panel is a display module (`display_module_link`, a 10.7 x 3.9 cm plate).
This module puts a thin textured screen on it in the render-only model and picks an
expression every frame from the simulation state, so the face turns, tilts and
foreshortens with the body exactly as the panel does (the coach, 10/10: "練習中は真剣な目、
回っているときはぐるぐる目、転んでいるときは ><、最後は ^ ^").

    FACE=auto      serious while practising, spiral while spinning, >< while falling
    FACE=neutral   round eyes (before any lesson)
    FACE=serious / spiral / cross / happy   one expression for the whole take
    FACE_BASE=neutral|serious   what `auto` shows when nothing is happening (default serious)
    FACE=0 or unset             no screen (the plain black panel, as before)

The physics model is untouched: the screen is a contype=0, density=0 visual geom that exists
only in the render model.
"""
from __future__ import annotations

import os
import tempfile

import numpy as np

EXPRESSIONS = ("neutral", "serious", "spiral", "cross", "happy")
GLOW = (120, 235, 255)          # the display's colour
TEX_W, TEX_H = 640, 232         # texture size, about the panel's aspect (10.0 x 3.6 cm drawn area)

# auto rules, on the base body (simulation time, so they hold at any slow-motion factor)
SPIN_ON, SPIN_OFF = 0.8, 0.5    # rad/s of yaw, smoothed: spiral eyes on above / off below (a slow, steady
                                # turn reads as "spinning": 180 degrees in 4 s is 0.8 rad/s)
TILT_ON, TILT_OFF = 50.0, 35.0  # deg from upright: >< on above / off below
HOLD = 0.35                     # s an expression stays at least, so the face does not flicker
CROSS_HOLD = 0.6                # s >< stays after a fall (at 1/3 speed, 1.8 s of video: long enough to read)
SMOOTH = 0.5                    # s, time constant of the yaw-rate filter. The rate is filtered with its sign,
                                # so the twitch of each bounce cancels out and only a steady turn remains
RESET_JUMP = 0.08               # m the base moves sideways between two frames only when a fall resets it


# ---------------------------------------------------------------- the pictures
def _draw(expr):
    from PIL import Image, ImageDraw, ImageFilter

    W, H = TEX_W, TEX_H
    img = Image.new("L", (W, H), 0)
    d = ImageDraw.Draw(img)
    lw = 22
    for side in (-1, 1):
        cx, cy = W / 2 + side * W * 0.22, H * 0.52
        if expr == "neutral":                       # round eyes
            r = H * 0.20
            d.ellipse((cx - r * 0.8, cy - r, cx + r * 0.8, cy + r), fill=255)
        elif expr == "serious":                     # focused: eyes cut by a lid slanting down to the centre
            w, h = H * 0.30, H * 0.20
            d.rounded_rectangle((cx - w, cy - h, cx + w, cy + h), radius=int(h * 0.6), fill=255)
            inner = cx - side * w                   # the lid is lower on the side nearer the centre
            outer = cx + side * w
            d.polygon([(inner, cy - h - 4), (inner, cy + h * 0.15), (outer, cy - h * 0.55),
                       (outer, cy - h - 4)], fill=0)
            d.line([(inner - side * 6, cy - h * 0.05), (outer + side * 4, cy - h * 0.95)], fill=255, width=lw // 2)
        elif expr == "spiral":                      # ぐるぐる
            pts = []
            turns, rmax = 2.6, H * 0.30
            for k in range(240):
                t = k / 239
                a = side * t * turns * 2 * np.pi
                r = rmax * t
                pts.append((cx + r * np.cos(a), cy + r * np.sin(a)))
            d.line(pts, fill=255, width=lw // 2 + 4, joint="curve")
        elif expr == "cross":                       # > <, pointing at each other
            a, b = H * 0.20, H * 0.26                # left eye ">" points right, right eye "<" points left
            d.line([(cx + side * a, cy - b), (cx - side * a, cy), (cx + side * a, cy + b)],
                   fill=255, width=lw, joint="curve")
        elif expr == "happy":                       # ^ ^
            a, b = H * 0.26, H * 0.22
            d.line([(cx - a, cy + b * 0.7), (cx, cy - b), (cx + a, cy + b * 0.7)], fill=255, width=lw, joint="curve")
        else:
            raise ValueError(expr)
    glow = img.filter(ImageFilter.GaussianBlur(9))
    a = np.clip(np.asarray(img, np.float32) / 255 + np.asarray(glow, np.float32) / 255 * 0.55, 0, 1)
    base = np.array((6, 8, 12), np.float32)        # the screen's own black, a shade lighter than the panel
    rgb = base + (np.array(GLOW, np.float32) - base) * a[..., None]
    return Image.fromarray(rgb.astype(np.uint8), "RGB")


def write_textures(folder=None):
    folder = folder or tempfile.mkdtemp(prefix="crab_face_")
    os.makedirs(folder, exist_ok=True)
    paths = {}
    for e in EXPRESSIONS:
        p = os.path.join(folder, f"face_{e}.png")
        _draw(e).save(p)
        paths[e] = p
    return paths


# ---------------------------------------------------------------- the screen
def _display_frame(geom_pos, geom_quat):
    """The plate's mesh frame -> body frame."""
    import mujoco

    R = np.zeros(9)
    mujoco.mju_quat2Mat(R, np.asarray(geom_quat, float))
    return np.asarray(geom_pos, float), R.reshape(3, 3)


def add_screen(spec, folder=None):
    """Add the screen geom and one material per expression to `spec` (a scene or robot MjSpec).
    Returns the screen's geom name and {expression: material name}."""
    import mujoco

    body = next((b for b in spec.bodies if b.name == "display_module_link" or b.name.endswith("/display_module_link")),
                None)
    if body is None:
        raise RuntimeError("no display_module_link in this model")
    prefix = body.name[: -len("display_module_link")]
    # the plate's geometry, measured on the shipped mesh (MuJoCo centres a mesh on its inertia frame):
    # 3.7 mm thick along mesh x, 3.9 cm along mesh y, 10.7 cm along mesh z; geom pos/quat place that
    # frame in the body (assets/jumper/jumper.xml, display_module_link_visual)
    pos, R = _display_frame((-0.00170, 0.0, -0.00070), (0.6747, 0.6748, -0.2115, 0.2115))
    nx = R @ np.array([1.0, 0.0, 0.0])
    front = 1.0 if nx[0] > 0 else -1.0             # the face the robot looks out of (+x of the body is forward)
    off = front * (0.0019 + 0.0004)                # 0.4 mm proud of the plate's surface
    hy, hz = 0.0170, 0.0500                        # half extents of the drawn area
    corners = [(off, -hy, -hz), (off, -hy, hz), (off, hy, hz), (off, hy, -hz)]
    verts = np.array([pos + R @ np.array(c) for c in corners])
    # which mesh-y direction is "up" on the face (towards the body's +z)
    up_y = 1.0 if (R @ np.array([0.0, 1.0, 0.0]))[2] > 0 else -1.0
    # left/right as the robot is seen from the front: looking at it, its +y side is on the viewer's right
    right_z = 1.0 if (R @ np.array([0.0, 0.0, 1.0]))[1] > 0 else -1.0
    uv = []
    for (_, y, z) in corners:
        u = 0.5 + 0.5 * (z / hz) * right_z
        v = 0.5 - 0.5 * (y / hy) * up_y            # texture rows run downwards
        uv.append((u, v))
    # wind the two triangles so their normal points out of the face (the renderer culls back faces)
    n = np.cross(verts[1] - verts[0], verts[2] - verts[0])
    faces = [0, 1, 2, 0, 2, 3] if float(n @ (R @ np.array([front, 0.0, 0.0]))) > 0 else [0, 2, 1, 0, 3, 2]
    mesh = spec.add_mesh(name=prefix + "face_screen_mesh")
    mesh.uservert = verts.ravel().tolist()
    mesh.usertexcoord = np.array(uv).ravel().tolist()
    mesh.userface = faces
    mesh.inertia = mujoco.mjtMeshInertia.mjMESH_INERTIA_SHELL   # a flat sheet has no volume
    paths = write_textures(folder)
    mats = {}
    for e, p in paths.items():
        tex = spec.add_texture(name=prefix + f"face_tex_{e}", type=mujoco.mjtTexture.mjTEXTURE_2D, file=p)
        mat = spec.add_material(name=prefix + f"face_mat_{e}")
        mat.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = tex.name
        mat.emission = 1.0
        mat.specular = 0.15
        mat.shininess = 0.6
        mat.texuniform = False
        mats[e] = mat.name
    g = body.add_geom(name=prefix + "face_screen", type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh.name,
                      contype=0, conaffinity=0, density=0, group=2, material=mats["serious"])
    return g.name, mats


class FaceDriver:
    """Picks the expression each frame and sets the screen's material."""

    def __init__(self, model, screen_geom, mats, mode=None, base=None):
        import mujoco

        self.mujoco = mujoco
        self.model = model
        self.gid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, screen_geom)
        self.mat = {e: mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_MATERIAL, n) for e, n in mats.items()}
        self.mode = mode or os.environ.get("FACE", "auto")
        self.base = base or os.environ.get("FACE_BASE", "serious")
        self.state = {"expr": self.base, "since": -1e9, "yaw": None, "t": None, "pos": None, "rate": 0.0,
                      "cross_until": -1e9}
        self.log = []
        self.debug = open(os.environ["FACE_DEBUG"], "w") if os.environ.get("FACE_DEBUG") else None
        if self.debug:
            self.debug.write("t,tilt_deg,yaw_rad,rate_rad_s,expr\n")

    def __call__(self, data, t, body):
        """`t` simulation time (s), `body` the base body id. Returns the expression shown."""
        s = self.state
        if self.mode in EXPRESSIONS:
            expr = self.mode
        else:
            xm = data.xmat[body].reshape(3, 3)
            pos = np.array(data.xpos[body], dtype=float)
            tilt = float(np.degrees(np.arccos(np.clip(xm[2, 2], -1.0, 1.0))))
            yaw = float(np.arctan2(xm[1, 0], xm[0, 0]))
            reset = s["pos"] is not None and float(np.hypot(*(pos[:2] - s["pos"][:2]))) > RESET_JUMP
            if s["yaw"] is not None and t > s["t"]:
                dyaw = (yaw - s["yaw"] + np.pi) % (2 * np.pi) - np.pi
                # a spin is the body turning while upright; a tumble or a reset is not a spin
                rate = dyaw / (t - s["t"]) if (tilt < 30.0 and not reset and abs(dyaw) < 0.6) else 0.0
                k = min(1.0, (t - s["t"]) / SMOOTH)
                s["rate"] += k * (rate - s["rate"])
                s["peak"] = max(s.get("peak", 0.0), abs(s["rate"]))
            s["yaw"], s["t"], s["pos"] = yaw, t, pos
            cur = s["expr"]
            if tilt > TILT_ON or reset or (cur == "cross" and tilt > TILT_OFF):
                s["cross_until"] = t + CROSS_HOLD
            if t < s["cross_until"]:
                want = "cross"                      # a fall shows at once, whatever was on before
            elif abs(s["rate"]) > SPIN_ON or (cur == "spiral" and abs(s["rate"]) > SPIN_OFF):
                want = "spiral"
            else:
                want = self.base
            expr = cur
            if want != cur and (want == "cross" or t - s["since"] >= HOLD):
                expr, s["since"] = want, t
            s["expr"] = expr
            if self.debug:
                self.debug.write(f"{t:.4f},{tilt:.1f},{yaw:.3f},{s['rate']:.3f},{expr}\n")
        self.model.geom_matid[self.gid] = self.mat[expr]
        if not self.log or self.log[-1][1] != expr:
            self.log.append((round(t, 3), expr))
        return expr


if __name__ == "__main__":
    # write the five pictures next to this file, for a look
    for e, p in write_textures(os.path.join(os.path.dirname(os.path.abspath(__file__)), "face_textures")).items():
        print(e, p)
