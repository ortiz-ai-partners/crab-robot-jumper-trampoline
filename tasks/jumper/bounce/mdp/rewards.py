"""Rewards for bouncing on a bed: the coach's words, one term each.

The prior pair (`ref_free_jump`'s `prior_joint_pos` / `prior_joint_vel`) is
reused as it is for the first jump and fades out on the same schedule. What is
here is the objective, and none of it reads the recording.

| coach                                          | term               |
|------------------------------------------------|--------------------|
| bounce, and higher each time                   | `bounce_apex`, `apex_growth`, `flight_height` |
| one rod: don't absorb, let the bed be the spring | `absorb`         |
| hands and feet down at the same time           | `touchdown_sync`   |
| come down upright and still; the arrow         | `air_upright`      |
| don't stretch the legs in the air              | `air_posture`      |
| stay on the bed                                | `horizontal_drift` (ref_free_jump's) |

Every one of them is gated on the command term's bookkeeping
(`mdp/commands.py`), so an episode that has not bounced yet is paid nothing
for bouncing and charged nothing for the landing it has not made.
"""

from __future__ import annotations

import torch
from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg


def bounce_apex(
    env, command_name: str = "jump", scale: float = 0.10, expire_s: float = 1.0
) -> torch.Tensor:
    """The last counted bounce's apex, in units of `scale`, paid every step for
    `expire_s` after its touchdown and nothing after. Linear and uncapped.

    A stream rather than a one-off because the reward manager scales by dt:
    a single payment of 1.2 would be 0.006. The expiry is what makes it a
    bouncing reward rather than a jumping one: standing still after one good
    jump stops paying within a bounce period, and the only way to be paid
    again is the next bounce."""
    cmd = env.command_manager.get_term(command_name)
    fresh = cmd.steps_since_touchdown * env.step_dt < expire_s
    return cmd.last_apex.clamp(min=0.0) / scale * fresh.float()


def apex_growth(
    env, command_name: str = "jump", scale: float = 0.10, clip: float = 1.0,
    expire_s: float = 1.0,
) -> torch.Tensor:
    """How much higher the last bounce went than the one before, in units of
    `scale`, clipped to +-`clip`, paid as `bounce_apex` is: from the second
    bounce on, for `expire_s` after the touchdown. Negative when the bounce
    decayed."""
    cmd = env.command_manager.get_term(command_name)
    fresh = (cmd.steps_since_touchdown * env.step_dt < expire_s) & (cmd.bounces >= 2.0)
    growth = ((cmd.last_apex - cmd.prev_apex) / scale).clamp(-clip, clip)
    return growth * fresh.float()


def flight_height(env, command_name: str = "jump", scale: float = 0.10) -> torch.Tensor:
    """How far the robot's lowest point is above HOME while in a flight of its
    own: the dense half, so a higher arc pays before its landing does."""
    cmd = env.command_manager.get_term(command_name)
    rise = cmd.lowest_rise().clamp(min=0.0) / scale
    return rise * cmd.in_flight.float()


def absorb(
    env, asset_cfg: SceneEntityCfg, command_name: str = "jump",
    sigma_sq: float = 0.01, cap: float = 4.0, window_s: float = 0.12,
) -> torch.Tensor:
    """Mean squared distance of the driven joints from the pose they touched
    down in, over `sigma_sq`, while on the ground, still going down and within
    `window_s` of the touchdown -- the bed's compression stroke -- capped at
    `cap`. Zero once the base is rising, so the push from the bottom is free;
    zero before the first bounce; and zero after the window, so standing on the
    bed afterwards (where the base's vertical speed dithers about zero) is not
    charged against a pose it rightly left. Measured without the window on the
    shallow bed: a robot standing 3 s after its landing paid 3.5 a step.

    The coach's "one rod": on a hard floor a soft landing bends the knees; on a
    bed the bed does the giving, and knees that bend take the stroke away from
    it. jumper.jump's `soft_landing` is this term with its sign turned."""
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    e = ((q - cmd.q_touchdown) ** 2).mean(dim=1) / sigma_sq
    recent = cmd.steps_since_touchdown * env.step_dt < window_s
    gate = cmd.grounded & cmd.descending & recent & (cmd.bounces >= 1.0)
    return e.clamp(max=cap) * gate.float()


def touchdown_sync(
    env, command_name: str = "jump", window_steps: int = 4
) -> torch.Tensor:
    """The share of the six feet not yet on the ground, every step of the
    `window_steps` after a counted touchdown. "Hands and feet down at the same
    time": the recording's landing puts the claws down late, and the bed then
    throws the robot over the front. Four steps is 20 ms at 200 Hz."""
    cmd = env.command_manager.get_term(command_name)
    window = (cmd.steps_since_touchdown < float(window_steps)) & (cmd.bounces >= 1.0)
    missing = 1.0 - cmd.feet_down.mean(dim=1)
    return missing * window.float()


def air_upright(
    env, command_name: str = "jump", sigma_sq: float = 0.10, omega_sigma_sq: float = 2.0
) -> torch.Tensor:
    """exp(-|g_xy|^2 / sigma_sq) * exp(-|w_xy|^2 / omega_sigma_sq) while in a
    flight of its own and descending: come down level, and not turning. It is
    paid for the outcome; how the legs buy it (the coach's "drop the leg on the
    side that came up") is the policy's to find, with the IMU it already
    observes. `omega_sigma_sq` 2.0 (rad/s)^2 is four times the IMU noise
    squared, so noise alone cannot empty it."""
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene["robot"]
    g = robot.data.projected_gravity_b[:, :2]
    w = robot.data.root_link_ang_vel_b[:, :2]
    level = torch.exp(-(g**2).sum(dim=1) / sigma_sq)
    still = torch.exp(-(w**2).sum(dim=1) / omega_sigma_sq)
    return level * still * (cmd.in_flight & cmd.descending).float()


def air_posture(
    env, asset_cfg: SceneEntityCfg, command_name: str = "jump", sigma_sq: float = 0.05
) -> torch.Tensor:
    """exp(-e / sigma_sq) while in a flight of its own, e the mean squared gap
    of the driven joints from HOME. "Stretching the legs in the air is what
    breaks the landing": the tuck is HOME, held, and the legs that right the
    body (`air_upright`) move within this kernel's width."""
    cmd = env.command_manager.get_term(command_name)
    robot: Entity = env.scene[asset_cfg.name]
    q = robot.data.joint_pos[:, asset_cfg.joint_ids]
    home = robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    e = ((q - home) ** 2).mean(dim=1)
    return torch.exp(-e / sigma_sq) * cmd.in_flight.float()


__all__ = [
    "absorb",
    "air_posture",
    "air_upright",
    "apex_growth",
    "bounce_apex",
    "flight_height",
    "touchdown_sync",
]
