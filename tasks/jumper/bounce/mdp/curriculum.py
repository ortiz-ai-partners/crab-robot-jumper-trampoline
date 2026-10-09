"""Schedules: the reward cross-fade is `ref_free_jump`'s; the drop height is new.

`anneal_drop_height` writes the event manager's own term config, for the
reason `ref_free_jump/mdp/curriculum.py` gives for the reward weights: the
manager calls each term with the params on the config it holds, so an
assignment there is the range the next reset samples from, and an assignment
to `env.cfg.events` would change nothing.

A schedule by training step, not a performance gate: the coach's "start low,
learn the air, then go higher" is a plan, and a plan that waits for the policy
to succeed at a height before raising it never raises it for a policy that
plateaus.
"""

from __future__ import annotations

import torch

from ...ref_free_jump.mdp.curriculum import anneal_reward_weight, report_feedforward


def anneal_drop_height(
    env,
    env_ids: torch.Tensor | None,
    term_name: str,
    start_range: tuple[float, float],
    end_range: tuple[float, float],
    start_step: int,
    end_step: int,
) -> float:
    """`start_range` until `start_step`, `end_range` from `end_step`, both ends
    moving linearly between. Returns the upper end, for the log."""
    del env_ids
    if not 0 <= start_step < end_step:
        raise ValueError(f"need 0 <= start_step < end_step, got {start_step} and {end_step}")
    s = env.common_step_counter
    frac = min(max((s - start_step) / (end_step - start_step), 0.0), 1.0)
    lo = start_range[0] + (end_range[0] - start_range[0]) * frac
    hi = start_range[1] + (end_range[1] - start_range[1]) * frac
    cfg = env.event_manager.get_term_cfg(term_name)
    cfg.params["pose_range"] = {**cfg.params["pose_range"], "z": (lo, hi)}
    return hi


__all__ = ["anneal_drop_height", "anneal_reward_weight", "report_feedforward"]
