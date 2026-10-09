# Measuring a real trampoline bed for `--scene trampoline`

The simulated bed (`scenes/trampoline.py`) has two dials that matter:
`--bed-dampratio` sets how much of the energy comes back (restitution) and
`--bed-timeconst` sets how deep it sinks. Both are placeholders until the bed
you mean has been measured. This is the procedure, as short as it can be, and
the record sheet. `tools/calibrate_bed.py` turns the two numbers into the dials.

*This file was reconstructed on 2026-10-09 from the project notes; the first
draft was lost with a scratch directory. The procedure is the same.*

## What you need

- A 2.5 kg weight with a flat bottom (a dumbbell plate, or a bag of rice on a
  board). The robot's simulated mass is 2.543 kg.
- A tape measure or a ruler held upright at the centre of the bed.
- A phone that records slow motion (120 fps or more), on a tripod or a chair,
  level with the bed, from the side.

## 1. Static sink: how stiff the bed is

1. With nothing on the bed, measure the height of the bed's centre above the
   floor. Write it down as the zero.
2. Place the 2.5 kg weight at the centre, let it settle, measure the height
   again. The difference is the static sink `x` in millimetres.
3. Repeat three times, lifting the weight off between readings.

Stiffness `k = m × 9.81 / x` in N/m (with `x` in metres). For 2.5 kg and a
10 mm sink that is 2450 N/m.

## 2. Drop rebound: how much energy comes back

1. Hold the weight so its bottom is exactly 0.50 m above the unloaded bed
   centre (measure to the bottom of the weight, not to your hand).
2. Start the slow-motion recording, let go cleanly, do not throw.
3. From the video, read two numbers: the height the weight bounces back to on
   its first rebound, `h_up` (bottom of the weight above the unloaded bed), and
   the deepest point the bed reaches under it, `x_dyn` (below the unloaded
   bed). A ruler in frame makes both readable.
4. Repeat three times.

Restitution `e = sqrt(h_up / 0.50)`; energy lost per bounce `1 − e²`. A
competition bed returns most of the energy (e around 0.9); a garden
trampoline less.

## 3. The weave

Measure the opening of the mesh (millimetres between threads) and the width of
a thread. These do not enter the simulation yet; they decide how the robot's
feet grip, and the real robot will tell us whether that matters.

## 4. From measurements to dials

    python tools/calibrate_bed.py --drop 0.5 --target-e <e> --target-sink-cm <x_dyn in cm>

prints a table and names the closest `--bed-timeconst / --bed-dampratio /
--bed-width`; refine with a finer grid around it. Then

    python scripts/play.py --task jumper.jump --checkpoint tasks/jumper/jump/out/example/model_3100.pt \
        --scene trampoline --bed-timeconst <tc> --bed-dampratio <dr> --bed-width <w>

or put the three in `.env` as `MJRL_BED_TIMECONST`, `MJRL_BED_DAMPRATIO`,
`MJRL_BED_WIDTH` so every run uses the measured bed.

## Record sheet

| date | bed (where, make) | zero height mm | static sink x (3 readings) mm | k N/m | drop m | h_up (3) m | e | x_dyn (3) cm | mesh opening mm | thread mm | notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| | | | | | 0.50 | | | | | | |

## What the simulation cannot reproduce

A real bed dishes around the feet and stores its energy over tens of
centimetres; the simulated one stores it over a few centimetres of contact
penetration and stays flat. Use the measured `e` and `x_dyn` to match the
rebound and the depth; do not expect the sideways grip of a real dish. The
docstring of `scenes/trampoline.py` has the full argument.
