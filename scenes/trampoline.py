"""A trampoline bed, approximated by under-damped soft contact on the ground.

**This scene changes the physics, not the look**, through the same mechanism as
`soft.py`, `ice.py` and `rubber.py`: the ground geoms get a contact parameter set
and the priority that makes it count. Read `registry.set_ground_contact` for the
three traps; read `soft.py` for what `solref` and `solimp` mean.

## What a bed is, in MuJoCo's vocabulary

MuJoCo's contact is a spring-damper on penetration. `solref = (timeconst,
dampratio)`: `timeconst` is how slowly the contact pushes back (longer is softer,
deeper), and `dampratio` is how much of the stored energy is lost on the way out.
`soft.py` pins `dampratio = 1.0` precisely so that mud does not behave like a
trampoline. This scene is the other choice: `dampratio` well below 1 leaves energy
in the contact, and the body carries it away as rebound.

## Measured, not guessed

A 2.5 kg stand-in (a box on four foot-like pads carrying the robot's own foot
contact parameters, `tools/calibrate_bed.py`) dropped from 0.30 m:

    dampratio      0.03   0.06   0.10   0.15   0.30
    restitution e  0.91   0.82   0.72   0.63   0.41      (timeconst 0.15)

    timeconst      0.08   0.15   0.25
    sink at impact 1.7    3.0    4.7 cm                   (dampratio 0.06)

and dropped from 0.50 m, with the bed left soft deeper in (`solimp` width 0.25):

    timeconst      0.25   0.40   0.60   0.90
    sink at impact 3.4    5.6    8.2    11.6 cm            (dampratio 0.03, e 0.90)
    standing sink  0.0    0.1    0.3    0.6 cm             (2.5 kg at rest, no jitter)

Restitution barely depends on drop height (0.79 to 0.82 from 0.14 to 0.50 m), so
`dampratio` is the bounce dial and `timeconst` the depth dial, nearly independently.
The defaults below are placeholders for a real bed: measure the static sink and the
drop rebound of the bed you mean (docs/measurement.md), then pass the matching
numbers on the command line or in `.env`.

Replayed with the shipped `jumper.jump` policy (2026-10-09, native backend, 700
steps, five runs each), the bed's depth decided whether it stood after landing and
its friction did not:

    bed                         fell   rise      feet force at landing   rear-foot slide
    rigid (studio)              0/5    0.12-0.15 m   90-220 N (4-9 x weight)   5-7 cm
    shallow, tc 0.15, e 0.90    0/5    0.11-0.12 m   93-125 N                  30-55 cm
    deep,    tc 0.60, e 0.90    3/9    0.12-0.15 m   22-54 N                   34-68 cm
    deep,    friction 2.5       0/5    0.03 m        21-24 N                   70-82 cm

Friction 2.5 did not cure the slide; it pinned the feet so the legs sank instead
and the robot could not jump. The slide is the rigid-ground push pattern meeting
a floor that gives, not a friction shortage, and a flat compliant plane also
lacks the dish a real bed forms around a foot, so these slide figures are upper
bounds for comparing conditions, not predictions of a bed.

## What this is not

Not a membrane. The ground does not dish around the feet, has no mass or tension of
its own, and gives its energy back the instant the load comes off. A real bed
stores energy over a stroke of tens of centimetres and returns it over the same
stroke; this one stores it over a few centimetres of penetration. That is enough
to ask "does the policy survive a bouncy landing" and "how high does a rebound go",
and not enough to ask "how does the robot pump a bed" -- the latter needs a deck on
a spring joint, the way `scenes/swing.py` builds a prop with tendons.

`foot_friction` domain randomisation is **inert here**, as on `ice` and `rubber`
and for the same reason: the ground has outranked the feet, so every sample is
discarded before the solver sees it.
"""

from __future__ import annotations

import os
from typing import Any

from .registry import GROUND, Headlight, Scene, set_ground_contact

#: Environment variables the three dials fall back to, so a calibrated bed can live
#: in `.env` rather than on every command line.
BED_TIMECONST_ENV = "MJRL_BED_TIMECONST"
BED_DAMPRATIO_ENV = "MJRL_BED_DAMPRATIO"
BED_WIDTH_ENV = "MJRL_BED_WIDTH"
BED_FRICTION_ENV = "MJRL_BED_FRICTION"

#: Placeholder bed: restitution about 0.9 (the coach's call: a competition bed
#: gives most of the energy back) and a shallow stroke, about 3 cm of sink for
#: 2.5 kg from 0.50 m. Shallow is deliberate. Measured on 2026-10-09 with the
#: shipped `jumper.jump` policy, which was trained on rigid ground: on a deep bed
#: (timeconst 0.6, 8 cm of sink) it jumps and then falls on every landing, because
#: the landing sinks 5 cm and the rebound takes the posture away; on this shallow
#: bed it lands and stands, with the same restitution. A default a shipped policy
#: cannot stand on is the wrong default. The deep bed stays one flag away
#: (`--bed-timeconst 0.6`) as the curriculum's far end for a policy trained here.
DEFAULT_TIMECONST = 0.15
DEFAULT_DAMPRATIO = 0.03
#: `solimp` width: the penetration depth over which the contact stiffens from `d0`
#: to `dwidth`. `soft.py`'s shape, with a wider band so the bed stays soft through
#: the whole stroke rather than turning rigid a few millimetres in.
DEFAULT_WIDTH = 0.25
BED_SOLIMP_D0 = 0.5
BED_SOLIMP_DWIDTH = 0.95

#: Sliding friction of the bed. A trampoline bed is a tight nylon weave under
#: rubber feet, which sits at the grippy end of `rubber.py`'s band (1.2 to 1.8).
#: The coach watching the first replays saw the rear feet skate at the feet's own
#: 1.2, so the default is the top of that band. Torsional and rolling keep the
#: feet's proportion to sliding (1/120, as `rubber.py` argues).
DEFAULT_FRICTION = 1.8


def cli_args(group):
    """This scene's own command-line arguments; see `scenes.load_cli_args`.

    The pattern is `rough.py`'s. Values arrive as strings and are converted in
    `scene()`, so an empty string means "use the default", which is what the
    environment-variable fallback needs.
    """
    group.add_argument(
        "--bed-timeconst", default=os.environ.get(BED_TIMECONST_ENV, ""), metavar="S",
        help=f"contact time constant of the bed in seconds; the depth dial "
        f"(default {DEFAULT_TIMECONST})",
    )
    group.add_argument(
        "--bed-dampratio", default=os.environ.get(BED_DAMPRATIO_ENV, ""), metavar="R",
        help=f"contact damping ratio; the bounce dial, 1.0 is dead and 0.03 is "
        f"restitution about 0.9 (default {DEFAULT_DAMPRATIO})",
    )
    group.add_argument(
        "--bed-width", default=os.environ.get(BED_WIDTH_ENV, ""), metavar="M",
        help=f"solimp width in metres, how deep the bed stays soft "
        f"(default {DEFAULT_WIDTH})",
    )
    group.add_argument(
        "--bed-friction", default=os.environ.get(BED_FRICTION_ENV, ""), metavar="MU",
        help=f"sliding friction of the bed; the feet's own is 1.2, a coarse rubber "
        f"mat 1.8 (default {DEFAULT_FRICTION})",
    )
    return ("bed_timeconst", "bed_dampratio", "bed_width", "bed_friction")


def _number(value, default: float, name: str) -> float:
    if value is None or str(value).strip() == "":
        return default
    try:
        out = float(value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if out <= 0:
        raise ValueError(f"{name} must be positive, got {out}")
    return out


def _lay_the_bed(spec: Any, friction, solref, solimp) -> None:
    """The bed's contact on every ground geom, at the priority that makes it count."""
    set_ground_contact(spec, "trampoline", friction=friction, solref=solref, solimp=solimp)


def scene(bed_timeconst=None, bed_dampratio=None, bed_width=None, bed_friction=None) -> Scene:
    from mjlab.utils import spec_config as sc

    timeconst = _number(bed_timeconst, DEFAULT_TIMECONST, "--bed-timeconst")
    dampratio = _number(bed_dampratio, DEFAULT_DAMPRATIO, "--bed-dampratio")
    width = _number(bed_width, DEFAULT_WIDTH, "--bed-width")
    sliding = _number(bed_friction, DEFAULT_FRICTION, "--bed-friction")
    friction = (sliding, sliding / 120.0, sliding / 120.0)
    solref = (timeconst, dampratio)
    solimp = (BED_SOLIMP_D0, BED_SOLIMP_DWIDTH, width, 0.5, 2.0)

    return Scene(
        # Indoor, flat light from above, like `rubber`: a gym hall, not a field.
        headlight=Headlight(ambient=(0.22, 0.22, 0.24), diffuse=(0.42, 0.42, 0.44)),
        textures=(
            # A trampoline bed is woven and dark; a fine checker in two near blacks
            # reads as weave at this scale, and the seams the eye needs to judge a
            # jump height come from the lighter cell.
            sc.TextureCfg(
                name="trampoline_bed", type="2d", builtin="checker",
                rgb1=(0.07, 0.09, 0.16), rgb2=(0.12, 0.15, 0.26),
                width=512, height=512,
            ),
            sc.TextureCfg(
                name="trampoline_sky", type="skybox", builtin="gradient",
                rgb1=(0.62, 0.66, 0.72), rgb2=(0.30, 0.32, 0.36),
                width=256, height=256,
            ),
        ),
        materials=(
            sc.MaterialCfg(
                name="trampoline_bed", texture="trampoline_bed", texuniform=True,
                texrepeat=(40.0, 40.0), reflectance=0.05, geom_names_expr=GROUND,
            ),
        ),
        lights=(
            sc.LightCfg(name="overhead", type="directional", pos=(0.0, 0.0, 4.0),
                        dir=(-0.1, -0.1, -1.0), diffuse=(0.72, 0.72, 0.74)),
            sc.LightCfg(name="fill", type="directional", pos=(2.0, -2.0, 2.0),
                        dir=(-0.5, 0.5, -1.0), diffuse=(0.22, 0.22, 0.24),
                        castshadow=False),
        ),
        decorate=lambda spec: _lay_the_bed(spec, friction, solref, solimp),
    )
