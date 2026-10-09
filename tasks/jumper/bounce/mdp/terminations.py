"""How an episode of jumper.bounce ends early, besides falling over.

`drifted_off`: the base has travelled `limit` metres horizontally from where it
started. The bed in the simulation is an infinite plane; a real one is not, and
a robot that bounces its way off the edge has stopped bouncing on a
trampoline. The shipped jump replayed on the deep bed slid its rear feet 1.1 to
2.5 m in 3.5 s (measured 2026-10-09, Island Master), which is the failure this
names: a landing recovered by skating is not a landing.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity


def drifted_off(env, limit: float = 1.0) -> torch.Tensor:
    robot: Entity = env.scene["robot"]
    d = robot.data.root_link_pos_w[:, :2] - env.scene.env_origins[:, :2]
    return d.norm(dim=-1) > limit


__all__ = ["drifted_off"]
