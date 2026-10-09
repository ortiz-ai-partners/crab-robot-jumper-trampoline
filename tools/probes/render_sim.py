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
it with a tracking camera locked on the robot's base. The mask comes from the same
renderer in segmentation mode: every geom whose body belongs to the robot's kinematic
tree is the robot.
"""
from __future__ import annotations

import os
import sys

REPO = r"D:\IslandMaster\jumper"


def install(play, out_path, fps=30, width=1080, height=1920, cam=(1.2, -18.0, 135.0), mask=True):
    inner_loop = play._loop

    def _loop(env, policy, viewer, max_steps, speed=1.0, readout=play._speed_text, monitor=None):
        import imageio_ffmpeg
        import mujoco
        import numpy as np

        unwrapped = getattr(env, "unwrapped", env)
        sim = unwrapped.sim
        model = getattr(sim, "render_model", None) or sim.mj_model
        model.vis.global_.offwidth = max(int(model.vis.global_.offwidth), width)
        model.vis.global_.offheight = max(int(model.vis.global_.offheight), height)
        data = mujoco.MjData(model)
        dt = float(unwrapped.step_dt)

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
        print(f"[render] {width}x{height} @ {fps} fps of simulation time -> {out_path}"
              f"{' (+ mask)' if mask else ''}; camera distance {cam[0]} elevation {cam[1]} azimuth {cam[2]}")

        def draw():
            q = sim.data.qpos[0]
            v = sim.data.qvel[0]
            data.qpos[:] = q.detach().cpu().numpy() if hasattr(q, "detach") else np.asarray(q)
            data.qvel[:] = v.detach().cpu().numpy() if hasattr(v, "detach") else np.asarray(v)
            mujoco.mj_forward(model, data)
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
                state["next"] += 1.0 / fps
            return out

        draw()                      # frame 0, the starting pose
        state["next"] = 1.0 / fps
        env.step = step
        try:
            return inner_loop(env, policy, viewer, max_steps, speed, readout, monitor)
        finally:
            env.step = original_step
            writer.close()
            if mask_writer is not None:
                mask_writer.close()
            print(f"\n[render] {state['frames']} frames = {state['frames'] / fps:.2f} s of simulation "
                  f"({state['step']} control steps) -> {out_path}")

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

    os.chdir(REPO)
    sys.path.insert(0, os.path.join(REPO, "scripts"))
    import play
    install(play, out, fps=fps, width=w, height=h, cam=cam, mask=mask)
    sys.argv = [os.path.join(REPO, "scripts", "play.py"), *theirs]
    play.main()


if __name__ == "__main__":
    _main()
