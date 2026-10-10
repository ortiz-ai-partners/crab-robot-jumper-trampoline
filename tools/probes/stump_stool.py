"""A stump stool (the island's own: a soft pouf printed as a tree stump, rings on top, grey bark on the side),
drawn where the crab lands after jumping off the round trampoline. Visual only (contype 0); the physics is the
landing mat the probe switches the floor to, at the same height as the stool's top (z = 0).

    STOOL=x,y,r  in render_sim: centre and radius of the stool, top at z = 0, standing on the gym floor (ROUND_HEIGHT)
"""
from __future__ import annotations

import math

import numpy as np


def _rings_texture(n=512, seed=7):
    """Year rings: warm tan, darker rings, a little wobble and a crack, lighter toward the rim (the cut edge)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[-1:1:complex(n), -1:1:complex(n)]
    r = np.hypot(x, y)
    a = np.arctan2(y, x)
    wob = 0.012 * np.sin(5 * a + 1.3) + 0.008 * np.sin(11 * a + 0.4)
    rr = r + wob * r
    rings = 0.5 + 0.5 * np.sin(rr * 2 * math.pi * 14 + 0.6 * np.sin(rr * 40))
    base = np.stack([0.80, 0.52, 0.30])[:, None, None] * np.ones((3, n, n))
    dark = np.stack([0.58, 0.33, 0.17])[:, None, None]
    img = base * (1 - 0.35 * rings) + dark * 0.35 * rings
    img *= (0.92 + 0.08 * rng.random((1, n, n)))
    rim = np.clip((r - 0.86) / 0.10, 0, 1)                 # the pale cut edge just inside the bark
    img = img * (1 - rim) + np.stack([0.92, 0.80, 0.62])[:, None, None] * rim
    crack = (np.abs(a - 0.35) < 0.012) & (r > 0.15) & (r < 0.8)
    img[:, crack] *= 0.55
    return (np.clip(img.transpose(1, 2, 0), 0, 1) * 255).astype(np.uint8)


def _bark_texture(w=512, h=256, seed=11):
    """Grey bark: long irregular vertical plates split by dark furrows (random streaks stretched along the trunk)."""
    rng = np.random.default_rng(seed)
    coarse = rng.random((h // 24 + 2, w))
    # stretch along y (linear interpolation between coarse rows), blur a little along x
    ys = np.linspace(0, coarse.shape[0] - 1.001, h)
    y0 = ys.astype(int); fy = (ys - y0)[:, None]
    f = coarse[y0] * (1 - fy) + coarse[y0 + 1] * fy
    k = np.array([1, 2, 3, 2, 1], float); k /= k.sum()
    f = np.apply_along_axis(lambda r: np.convolve(np.r_[r[-2:], r, r[:2]], k, mode="valid"), 1, f)
    furrow = np.clip((0.47 - f) / 0.10, 0, 1)
    v = 0.50 + 0.25 * f + 0.08 * rng.random((h, w)) - 0.33 * furrow
    img = np.stack([v * 0.93, v * 0.92, v * 0.95], axis=-1)
    return (np.clip(img, 0, 1) * 255).astype(np.uint8)


def _texture(spec, name, img):
    import mujoco
    t = spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=img.shape[1], height=img.shape[0],
                         nchannel=3)
    t.data = img.tobytes()
    m = spec.add_material(name=name)
    m.textures[mujoco.mjtTextureRole.mjTEXROLE_RGB] = name
    m.reflectance = 0.0
    return m


def add_stool(spec, cx, cy, r, floor_z, top_z=0.0, n=72):
    """Add the stool to the render-only spec: a capped top (rings) and a bulging side (bark), soft-edged."""
    import mujoco
    _texture(spec, "stool_rings", _rings_texture())
    _texture(spec, "stool_bark", _bark_texture())
    h = top_z - floor_z
    # top: a disc fan, slightly domed (a pouf), texture mapped across the disc
    verts, tex, faces = [0.0, 0.0, 0.006], [0.5, 0.5], []
    ring_r = r * 0.97
    for i in range(n):
        a = 2 * math.pi * i / n
        verts += [ring_r * math.cos(a), ring_r * math.sin(a), 0.0]
        tex += [0.5 + 0.5 * math.cos(a), 0.5 + 0.5 * math.sin(a)]
    for i in range(n):
        faces += [0, 1 + i, 1 + (i + 1) % n]
    spec.add_mesh(name="stool_top", uservert=verts, usertexcoord=tex, userface=faces)
    # side: a tube that bulges a little at mid height and rounds into the top
    rows = [(0.0, r * 0.97), (0.012, r), (0.25, r * 1.03), (0.6, r * 1.03), (0.95, r * 0.99), (1.0, r * 0.95)]
    sv, st, sf = [], [], []
    for k, (f, rad) in enumerate(rows):
        z = -f * h
        for i in range(n + 1):
            a = 2 * math.pi * i / n
            sv += [rad * math.cos(a), rad * math.sin(a), z]
            st += [i / n * 2.0, f]
    w = n + 1
    for k in range(len(rows) - 1):
        for i in range(n):
            p, q = k * w + i, (k + 1) * w + i
            sf += [p, q, q + 1, p, q + 1, p + 1]
    spec.add_mesh(name="stool_side", uservert=sv, usertexcoord=st, userface=sf)
    for name, mesh, mat in (("stool_top_geom", "stool_top", "stool_rings"), ("stool_side_geom", "stool_side", "stool_bark")):
        g = spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_MESH, meshname=mesh, material=mat,
                                    pos=[cx, cy, top_z], contype=0, conaffinity=0)
        g.group = 0
    return h


if __name__ == "__main__":
    # a look at the stool alone
    import sys
    import mujoco
    spec = mujoco.MjSpec()
    spec.worldbody.add_light(pos=[0, 0, 3], dir=[0, 0, -1], diffuse=[0.8, 0.8, 0.8])
    spec.worldbody.add_geom(type=mujoco.mjtGeom.mjGEOM_PLANE, size=[3, 3, 0.1], pos=[0, 0, -0.25],
                            rgba=[0.66, 0.50, 0.34, 1])
    add_stool(spec, 0.0, 0.0, 0.17, -0.25)
    m = spec.compile()
    d = mujoco.MjData(m)
    mujoco.mj_forward(m, d)
    rend = mujoco.Renderer(m, 480, 640)
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, -0.1]
    cam.distance, cam.elevation, cam.azimuth = 1.0, -25, 60
    rend.update_scene(d, camera=cam)
    img = rend.render()
    out = sys.argv[1] if len(sys.argv) > 1 else "stool.png"
    import zlib, struct  # tiny PNG writer, no extra dependency

    def png(path, a):
        raw = b"".join(b"\x00" + a[i].tobytes() for i in range(a.shape[0]))
        def chunk(t, b):
            return struct.pack(">I", len(b)) + t + b + struct.pack(">I", zlib.crc32(t + b) & 0xffffffff)
        with open(path, "wb") as fh:
            fh.write(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", a.shape[1], a.shape[0], 8, 2, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))
    png(out, img)
    print("wrote", out)
