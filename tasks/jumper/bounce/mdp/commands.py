"""The go command, with the bookkeeping repeated bouncing needs.

`jumper.ref_free_jump`'s `RefFreeJumpCommand` keeps one jump per episode and
takes the apex back if the robot flies again. Here flying again is the task,
so the bookkeeping is per **bounce**: a flight that the episode itself pushed
off the ground, at whose peak the robot's lowest point rose `min_rise` above
HOME, ended by a touchdown. Each touchdown latches what the rewards read:

- `last_apex` / `prev_apex`: the apex of the bounce just ended and of the one
  before it, so the height stream and the growth term are both one number;
- `q_touchdown`: the 20 driven joints' angles at the moment of touchdown, which
  `rewards.absorb` holds the robot to while the bed compresses ("one rod, push
  at the bottom");
- `steps_since_touchdown`: the touchdown-synchrony window and the height
  stream's expiry are counted from it;
- `touchdown_speed_xy`, `touchdown_ang_speed`: the "arrow" at touchdown, for
  the log and for the critic.

**A flight has to start from the ground.** `takeoff` requires the robot to have
been on the ground on the previous control step: an episode spawned in the air
-- by the drop event, or by RSI mid-flight -- is not credited with the fall it
is already in. That is the same farming `ref_free_jump` closed with its
`can_jump` gate, which is kept for the first flight: a spawn past the bottom of
the recording's crouch rides the recording's push and does not own that jump.
Once it has touched down, every later flight is its own.

Inherits `RefFreeJumpCommand` for `lowest_rise` and `_heights_at_home`; the
parent's one-jump logic is replaced wholesale in `_update_command`, and its
`home_steps` / `second_jump` are left at zero.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import torch

from ...ref_free_jump.mdp.commands import RefFreeJumpCommand, RefFreeJumpCommandCfg

if TYPE_CHECKING:
    from mjlab.envs import ManagerBasedRlEnv


@dataclass(kw_only=True)
class BounceCommandCfg(RefFreeJumpCommandCfg):
    #: `ref_free_jump` requires both; nothing here reads them.
    home_tolerance: float = 0.05
    home_speed_sq: float = 0.5

    go_after_landing_s: float = 0.10
    """For a spawn that is not on the recording: the go comes once the robot
    has been on the ground for this long without a break, not at the episode's
    first step. A HOME spawn is dropped onto the bed (`env_cfg.py`), and a
    recording fed forward from step 0 would crouch and push in the air:
    measured with the recording alone on the shallow bed, dropped 5 cm, the
    jump rose 0.07 m against the 0.11 to 0.12 the Island Master replayed from
    a stand (2026-10-09). Continuous contact rather than a delay from the first
    touchdown, because a bed with restitution 0.9 returns a dropped robot into
    the air several times before it settles, and a go fixed 0.10 s after the
    first touchdown landed in one of those hops (the same 0.06 m)."""

    def __post_init__(self):
        self.class_type = BounceCommand

    def build(self, env: ManagerBasedRlEnv) -> BounceCommand:
        return BounceCommand(self, env)


class BounceCommand(RefFreeJumpCommand):
    cfg: BounceCommandCfg

    def __init__(self, cfg: BounceCommandCfg, env: ManagerBasedRlEnv):
        super().__init__(cfg, env)
        n = self.num_envs
        dev = self.device
        #: On the ground on the previous control step (a takeoff needs one).
        self.grounded_prev = torch.zeros(n, dtype=torch.bool, device=dev)
        #: Waiting for the robot to settle on the ground to set the go (non-RSI
        #: spawns), and the consecutive grounded steps counted towards it.
        self.go_pending = torch.zeros(n, dtype=torch.bool, device=dev)
        self.settled_steps = torch.zeros(n, device=dev)
        #: A go step no episode reaches, while the go is pending.
        self._never = float(self._env.max_episode_length) * 4.0
        #: Any touchdown seen this episode, counted or not: from then on every
        #: flight is the episode's own.
        self.landed_once = torch.zeros(n, dtype=torch.bool, device=dev)
        #: Bounces counted this episode.
        self.bounces = torch.zeros(n, device=dev)
        #: Apex (lowest point above HOME) of the last counted bounce and of the
        #: one before it.
        self.last_apex = torch.zeros(n, device=dev)
        self.prev_apex = torch.zeros(n, device=dev)
        #: Best apex this episode, for the log.
        self.best_apex = torch.zeros(n, device=dev)
        #: Control steps since the last counted touchdown (large before any).
        self.steps_since_touchdown = torch.full((n,), 1.0e6, device=dev)
        #: The 20 driven joints at the last counted touchdown.
        self.q_touchdown = self.robot.data.default_joint_pos[:, self._gait_ids].clone()
        #: Horizontal speed and angular speed of the base at the last touchdown.
        self.touchdown_speed_xy = torch.zeros(n, device=dev)
        self.touchdown_ang_speed = torch.zeros(n, device=dev)
        #: Read by rewards on the current step: on the ground now, descending now.
        self.grounded = torch.zeros(n, dtype=torch.bool, device=dev)
        self.descending = torch.zeros(n, dtype=torch.bool, device=dev)
        self.feet_down = torch.zeros(n, 6, device=dev)
        for k in ("bounces", "best_apex", "last_apex", "td_speed_xy", "td_ang_speed"):
            self.metrics[k] = torch.zeros(n, device=dev)

    def _resample_command(self, env_ids: torch.Tensor) -> None:
        super()._resample_command(env_ids)
        # The parent set the go for a non-RSI spawn at `command_delay_range`
        # from step 0 (go_step >= 0). Hold it until the first touchdown instead.
        pending = self.go_step[env_ids] >= 0.0
        self.go_pending[env_ids] = pending
        self.go_step[env_ids] = torch.where(
            pending, torch.full_like(self.go_step[env_ids], self._never), self.go_step[env_ids]
        )
        self.settled_steps[env_ids] = 0.0
        self.grounded_prev[env_ids] = False
        self.landed_once[env_ids] = False
        self.bounces[env_ids] = 0.0
        self.last_apex[env_ids] = 0.0
        self.prev_apex[env_ids] = 0.0
        self.best_apex[env_ids] = 0.0
        self.steps_since_touchdown[env_ids] = 1.0e6
        self.q_touchdown[env_ids] = self.robot.data.default_joint_pos[env_ids][:, self._gait_ids]
        self.touchdown_speed_xy[env_ids] = 0.0
        self.touchdown_ang_speed[env_ids] = 0.0
        self.grounded[env_ids] = False
        self.descending[env_ids] = False
        self.feet_down[env_ids] = 0.0

    def _update_metrics(self) -> None:
        super()._update_metrics()
        self.metrics["bounces"] = self.bounces
        self.metrics["best_apex"] = self.best_apex
        self.metrics["last_apex"] = self.last_apex
        self.metrics["td_speed_xy"] = self.touchdown_speed_xy
        self.metrics["td_ang_speed"] = self.touchdown_ang_speed

    def _update_command(self, env_ids: torch.Tensor | None = None) -> None:
        # Skip `RefFreeJumpCommand`'s one-jump bookkeeping; keep `JumpMotionCommand`'s
        # metrics (apex_rise, flight_s, takeoff_vz).
        super(RefFreeJumpCommand, self)._update_command(env_ids)
        if env_ids is not None:
            return

        forces = self._env.scene.sensors[self.cfg.contact_sensor_name].data.force
        assert forces is not None
        down = forces.norm(dim=-1) > self.cfg.force_threshold
        self.feet_down.copy_(down.float())
        airborne = ~down.any(dim=1)
        grounded = ~airborne
        rise = self.lowest_rise()
        data = self.robot.data
        self.descending.copy_(data.root_link_lin_vel_w[:, 2] < 0.0)
        self.grounded.copy_(grounded)

        # A dropped spawn settles: `go_after_landing_s` of unbroken contact, then
        # the go is now -- the next control step is the recording's first.
        self.settled_steps.copy_(
            torch.where(grounded, self.settled_steps + 1.0, torch.zeros_like(self.settled_steps))
        )
        need = self.cfg.go_after_landing_s / self._env.step_dt
        go_now = self.go_pending & (self.settled_steps >= need)
        if go_now.any():
            self.go_step[go_now] = self._env.episode_length_buf[go_now].float()
            self.go_pending &= ~go_now

        own = self.can_jump | self.landed_once
        takeoff = (
            self.grounded_prev & airborne & ~self.in_flight
            & (self.time_since_go >= 0.0) & own
        )
        self.in_flight |= takeoff
        # In place, as `ref_free_jump` explains: rebinding under inference mode
        # makes the next in-place write raise.
        self.apex_rise.copy_(
            torch.where(self.in_flight, torch.maximum(self.apex_rise, rise), self.apex_rise)
        )

        touchdown = self.in_flight & grounded
        counted = touchdown & (self.apex_rise > self.cfg.min_rise)
        self.in_flight &= ~touchdown
        self.landed_once |= grounded
        self.landed |= counted

        if counted.any():
            ids = counted.nonzero().flatten()
            self.prev_apex[ids] = self.last_apex[ids]
            self.last_apex[ids] = self.apex_rise[ids]
            self.best_apex[ids] = torch.maximum(self.best_apex[ids], self.apex_rise[ids])
            self.bounces[ids] += 1.0
            self.steps_since_touchdown[ids] = 0.0
            self.q_touchdown[ids] = data.joint_pos[ids][:, self._gait_ids]
            self.touchdown_speed_xy[ids] = data.root_link_lin_vel_w[ids, :2].norm(dim=-1)
            self.touchdown_ang_speed[ids] = data.root_link_ang_vel_b[ids, :2].norm(dim=-1)
        self.steps_since_touchdown.add_(1.0)
        self.steps_since_touchdown[counted] = 0.0
        self.apex_rise.masked_fill_(touchdown, 0.0)
        self.grounded_prev.copy_(grounded)


__all__ = ["BounceCommand", "BounceCommandCfg"]
