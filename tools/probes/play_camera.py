#!/usr/bin/env python3
"""One-off wrapper around scripts/play.py that moves the live viewer's tracking camera
closer, for recording. Nothing in the repository changes. Camera from env vars:

    REC_CAM_DISTANCE   metres from the followed body (play's default is >= 6)
    REC_CAM_ELEVATION  degrees, negative looks down (play's default is -25)
    REC_CAM_AZIMUTH    degrees around the body

    python play_camera.py <play.py arguments>
"""
from __future__ import annotations

import os
import sys

REPO = os.environ.get("JUMPER_REPO", r"D:\IslandMaster\jumper")  # the jumper checkout this overlay is installed in
DIST = float(os.environ.get("REC_CAM_DISTANCE", "1.2"))
ELEV = float(os.environ.get("REC_CAM_ELEVATION", "-18"))
AZIM = float(os.environ.get("REC_CAM_AZIMUTH", "135"))

os.chdir(REPO)
sys.path.insert(0, os.path.join(REPO, "scripts"))
import play  # noqa: E402

_orig_loop = play._loop


def _loop(env, policy, viewer, max_steps, speed=1.0, readout=play._speed_text, monitor=None):
    state = {"left": 400, "said": False}  # re-apply over the first 400 steps (2 s at 200 Hz)

    def readout_and_aim(entity, index, env_):
        if viewer is not None and state["left"] > 0:
            handle = getattr(viewer, "_handle", None)
            if handle is not None:
                try:
                    with handle.lock():
                        handle.cam.distance = DIST
                        handle.cam.elevation = ELEV
                        handle.cam.azimuth = AZIM
                    state["left"] -= 1
                    if not state["said"]:
                        print(f"[cam] distance {DIST} elevation {ELEV} azimuth {AZIM}")
                        state["said"] = True
                except Exception as exc:  # the viewer may be mid-teardown
                    print(f"[cam] could not set the camera: {exc}")
                    state["left"] = 0
        return readout(entity, index, env_)

    return _orig_loop(env, policy, viewer, max_steps, speed, readout_and_aim, monitor)


play._loop = _loop
sys.argv = [os.path.join(REPO, "scripts", "play.py"), *sys.argv[1:]]
play.main()
