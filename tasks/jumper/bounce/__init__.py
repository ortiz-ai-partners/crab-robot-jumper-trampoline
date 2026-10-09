"""jumper.bounce -- keep bouncing on a trampoline bed, as high as it can.

jumper.ref_free_jump is one jump from HOME and the way back. This task is the
same policy asked to go on: land, hold the landing pose through the bed's
compression, push from the bottom of the stroke, fly, and land again -- until
it falls or the episode runs out. The floor is `--scene trampoline`; the task
itself owns no bed, so the same policy can be trained shallow or deep by the
scene's dials. See `env_cfg.py`.
"""

from __future__ import annotations

from ...registry import register
from ..common.assets import JUMPER_ASSETS

register(
    id="jumper.bounce",
    assets=JUMPER_ASSETS,
    description=(
        "jumper hexapod bouncing on a trampoline bed: repeated jumps, paid per bounce "
        "for the apex and its growth (train with --scene trampoline)"
    ),
    tags=("locomotion", "jumper", "jump", "action-prior", "trampoline"),
)
