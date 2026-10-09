"""Critic-only observation: the bounce bookkeeping the rewards are gated on.

The actor sees its body and nothing else, as in `ref_free_jump`; the critic is
thrown away at export and may see what the rewards read. `mirror_kind` is read
off the params by `common/mdp/symmetry.py`: every column is left-right
symmetric, so "scalar".
"""

from __future__ import annotations

import torch


def bounce_state(
    env,
    command_name: str = "jump",
    scale: float = 0.10,
    mirror_kind: str | None = None,
) -> torch.Tensor:
    """(in_flight, grounded, descending, bounces / 10, last apex / scale,
    seconds since touchdown capped at 2, share of feet down)."""
    del mirror_kind
    cmd = env.command_manager.get_term(command_name)
    since = (cmd.steps_since_touchdown * env.step_dt).clamp(max=2.0) / 2.0
    return torch.stack(
        [
            cmd.in_flight.float(),
            cmd.grounded.float(),
            cmd.descending.float(),
            cmd.bounces / 10.0,
            cmd.last_apex.clamp(min=0.0) / scale,
            since,
            cmd.feet_down.mean(dim=1),
        ],
        dim=1,
    )


__all__ = ["bounce_state"]
