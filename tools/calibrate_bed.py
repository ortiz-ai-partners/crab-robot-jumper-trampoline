#!/usr/bin/env python3
"""Turn a measured trampoline bed into `--scene trampoline` numbers.

Drops a stand-in for the robot (a 2.5 kg box on four foot-like pads that carry the
jumper's own foot contact parameters) onto a plane with candidate bed contact
parameters, and reports the sink at impact and the coefficient of restitution.
Needs only `mujoco` and `numpy`; no repository import.

    python tools/calibrate_bed.py                       # the table
    python tools/calibrate_bed.py --target-e 0.85 --target-sink-cm 4 --drop 0.5

With targets it also names the closest row. Measure the targets on the real bed
with docs/measurement.md: restitution from the drop test, sink from the slow-motion
frame at maximum depth, at the same drop height you pass here.
"""
from __future__ import annotations

import argparse
import itertools

import mujoco
import numpy as np

FOOT_SOLREF = (0.02, 1.0)                      # tasks/jumper/common/constants.py
FOOT_SOLIMP = (0.8, 0.90, 0.002, 0.5, 2.0)
FOOT_FRICTION = "1.2 0.01 0.01"


def drop_test(solref, solimp, drop=0.30, mass=2.543, seconds=2.5):
    pads = "".join(
        f'<geom name="f{i}" type="sphere" size="0.012" pos="{x} {y} -0.05" mass="0.05" '
        f'priority="1" solref="{FOOT_SOLREF[0]} {FOOT_SOLREF[1]}" '
        f'solimp="{" ".join(map(str, FOOT_SOLIMP))}" friction="{FOOT_FRICTION}" condim="3"/>'
        for i, (x, y) in enumerate([(0.15, 0.15), (-0.15, 0.15), (0.15, -0.15), (-0.15, -0.15)])
    )
    xml = f"""<mujoco><option timestep="0.002" integrator="implicitfast"/><worldbody>
    <geom name="bed" type="plane" size="5 5 0.1" priority="2" friction="{FOOT_FRICTION}"
          solref="{solref[0]} {solref[1]}" solimp="{" ".join(map(str, solimp))}"/>
    <body name="robot" pos="0 0 {drop + 0.05 + 0.012}"><freejoint/>
      <geom type="box" size="0.15 0.15 0.04" mass="{mass - 0.2}" contype="0" conaffinity="0"/>
      {pads}</body></worldbody></mujoco>"""
    m = mujoco.MjModel.from_xml_string(xml)
    d = mujoco.MjData(m)
    zs = []
    while d.time < seconds:
        mujoco.mj_step(m, d)
        zs.append(d.qpos[2])
    zs = np.array(zs)
    ground = 0.05 + 0.012                       # body height when the pads just touch
    sink = max(0.0, ground - zs.min())
    i0 = int(np.argmax(zs < ground))
    seg = zs[i0:]
    j = int(np.argmax(seg > ground)) if np.any(seg > ground) else 0
    apex = float(seg[j:].max() - ground) if j > 0 else 0.0
    e = float(np.sqrt(max(apex, 0.0) / drop))
    return sink, apex, e


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mass", type=float, default=2.543, help="kg; the jumper model is 2.543")
    p.add_argument("--drop", type=float, default=0.30, help="drop height in metres")
    p.add_argument("--timeconst", type=float, nargs="+", default=[0.08, 0.15, 0.25])
    p.add_argument("--dampratio", type=float, nargs="+", default=[0.03, 0.06, 0.10, 0.15, 0.30])
    p.add_argument("--width", type=float, nargs="+", default=[0.10])
    p.add_argument("--target-e", type=float, default=None, help="measured restitution")
    p.add_argument("--target-sink-cm", type=float, default=None, help="measured sink at impact")
    a = p.parse_args()

    rows = []
    print(f"mass {a.mass} kg, drop {a.drop} m")
    print(f"{'timeconst':>9} {'dampratio':>9} {'width':>6} | {'sink cm':>7} {'apex cm':>7} {'e':>5} {'loss':>5}")
    for tc, dr, w in itertools.product(a.timeconst, a.dampratio, a.width):
        solimp = (0.5, 0.95, w, 0.5, 2.0)
        sink, apex, e = drop_test((tc, dr), solimp, drop=a.drop, mass=a.mass)
        rows.append((tc, dr, w, sink, apex, e))
        print(f"{tc:9.3f} {dr:9.3f} {w:6.2f} | {sink * 100:7.1f} {apex * 100:7.1f} {e:5.2f} {1 - e * e:5.2f}")

    if a.target_e is not None or a.target_sink_cm is not None:
        def err(r):
            s = 0.0
            if a.target_e is not None:
                s += ((r[5] - a.target_e) / 0.1) ** 2
            if a.target_sink_cm is not None:
                s += ((r[3] * 100 - a.target_sink_cm) / 1.0) ** 2
            return s
        best = min(rows, key=err)
        print(f"\nclosest: --bed-timeconst {best[0]} --bed-dampratio {best[1]} --bed-width {best[2]}"
              f"  (sink {best[3] * 100:.1f} cm, e {best[5]:.2f})")
        print("Refine by passing a finer --timeconst / --dampratio grid around it.")


if __name__ == "__main__":
    main()
