#!/usr/bin/env python3
"""List the moments of the tricks in a probe_gymnast_v5 log (the CSV with gz, gx, gy), for picking video cuts.

    python trick_times.py take.csv [--slow 3]

Angles come from the projected gravity in the body frame (g = R^T (0, 0, -1)):
  pitch = atan2(gx, -gz)  forward tilt, + = nose down (0 upright, +-180 upside down), unwrapped over time
  roll  = atan2(gy, -gz)  sideways tilt, unwrapped
  tilt  = acos(-gz)       angle from upright, 0..180

  back drop (背落ち) : upside down (gz > --back-gz) at the bottom of a bounce (vz turning from down to up) with the
                       base low (z < --back-z): landed on the back and thrown up again
  somersault (宙返り): the pitch or the roll turns through --flip-deg or more within --flip-window s
                       (+ = forward / to the right, - = backward / to the left); notes whether the bed was
                       touched on the way (feet, or the base below --touch-z)
  donkey (ドンキー)  : head down, the nose --donkey-deg or more below the horizon (asin gx), with the base near
                       the bed (z < --donkey-z)

Times are simulation seconds; the video time is that times --slow (3 = a third of real speed).
"""
import csv
import math
import sys

args = sys.argv[1:]


def opt(name, default):
    return float(args[args.index(name) + 1]) if name in args else default


path = args[0]
SLOW = opt("--slow", 3.0)
BACK_GZ, BACK_Z = opt("--back-gz", 0.5), opt("--back-z", 0.06)
FLIP_DEG, FLIP_WIN = opt("--flip-deg", 330.0), opt("--flip-window", 2.0)
TOUCH_Z = opt("--touch-z", 0.03)
DONKEY_DEG, DONKEY_Z = opt("--donkey-deg", 60.0), opt("--donkey-z", 0.12)

sys.stdout.reconfigure(encoding="utf-8")
rows = list(csv.DictReader(open(path, encoding="utf-8")))
t = [float(r["t"]) for r in rows]
z = [float(r["base_z"]) for r in rows]
vz = [float(r["vz"]) for r in rows]
F = [float(r["foot_force_total"]) for r in rows]
gz = [float(r["gz"]) for r in rows]
gx = [float(r["gx"]) for r in rows]
gy = [float(r["gy"]) for r in rows]


def unwrap(a):
    out, prev, off = [], None, 0.0
    for v in a:
        if prev is not None:
            d = v - prev
            if d > 180:
                off -= 360
            elif d < -180:
                off += 360
        out.append(v + off)
        prev = v
    return out


pitch = unwrap([math.degrees(math.atan2(x, -c)) for x, c in zip(gx, gz)])
roll = unwrap([math.degrees(math.atan2(y, -c)) for y, c in zip(gy, gz)])
tilt = [math.degrees(math.acos(max(-1.0, min(1.0, -c)))) for c in gz]


def v(ts):
    return f"{ts:6.2f} s (video {ts * SLOW:6.2f} s)"


# back drops: the first bounce on the back after being the right way up; the bounces on the back that follow
# without getting up again are counted with it (an "upside-down spell" ends when gz < -0.3)
backs, last, spell = [], -9.0, None
for i in range(1, len(rows)):
    if gz[i] < -0.3 and spell is not None:
        spell[2] = i                                   # back the right way up
        backs.append(spell)
        spell = None
    if gz[i] > BACK_GZ and vz[i - 1] < 0 <= vz[i] and z[i] < BACK_Z and t[i] - last > 0.15:
        last = t[i]
        if spell is None:
            spell = [i, 1, None]
        else:
            spell[1] += 1
if spell is not None:
    backs.append(spell)

# somersaults: the first index j after i where pitch or roll has turned FLIP_DEG within FLIP_WIN
flips, i, n = [], 0, len(rows)
while i < n:
    hit = None
    j = i + 1
    while j < n and t[j] - t[i] <= FLIP_WIN:
        for name, a in (("pitch", pitch), ("roll", roll)):
            if abs(a[j] - a[i]) >= FLIP_DEG:
                hit = (name, j, a[j] - a[i])
                break
        if hit:
            break
        j += 1
    if hit:
        name, j, d = hit
        # tighten the start: the latest i' whose turn to j is still FLIP_DEG
        a = pitch if name == "pitch" else roll
        k = j
        while k > i and abs(a[j] - a[k - 1]) < FLIP_DEG:
            k -= 1
        k -= 1
        touched = any(F[q] > 1.0 or z[q] < TOUCH_Z for q in range(k, j + 1))
        flips.append((k, j, name, d, touched))
        i = j + 1
    else:
        i += 1

# donkeys
donkeys, cur = [], None
for i in range(n):
    # head down = the nose points at the bed: asin(gx) is the nose's angle below the horizon (90 = straight down).
    # (Not the pitch: lying on the back reads 180 deg of pitch with the nose level)
    on = math.degrees(math.asin(max(-1.0, min(1.0, gx[i])))) >= DONKEY_DEG and z[i] < DONKEY_Z
    if on and cur is None:
        cur = [i, i]
    elif on:
        cur[1] = i
    elif cur is not None:
        donkeys.append(cur)
        cur = None
if cur is not None:
    donkeys.append(cur)
donkeys = [d for d in donkeys if t[d[1]] - t[d[0]] >= 0.02]

print(f"# {path}")
print(f"## back drops (背落ち): {len(backs)} (the first bounce of each upside-down spell)")
for i, cnt, up in backs:
    tail = (f"; {cnt - 1} more bounces on the back" if cnt > 1 else "") +            (f", the right way up again at {t[up]:.2f} s (video {t[up] * SLOW:.2f} s)" if up is not None
            else ", still on the back at the end")
    print(f"  {v(t[i])}  upside down by {tilt[i]:.0f} deg{tail}")
print(f"## somersaults (宙返り): {len(flips)}")
for k, j, name, d, touched in flips:
    kind = {("pitch", True): "forward", ("pitch", False): "backward",
            ("roll", True): "to the right", ("roll", False): "to the left"}[(name, d > 0)]
    print(f"  {v(t[k])} -> {t[j]:6.2f} s (video {t[j] * SLOW:6.2f} s)  {abs(d):.0f} deg {kind} ({name})"
          f"{', touching the bed on the way' if touched else ', all in the air'}")
print(f"## donkeys (ドンキー, head down >= {DONKEY_DEG:.0f} deg, base below {DONKEY_Z} m): {len(donkeys)}")
for a, b in donkeys:
    mx = max(math.degrees(math.asin(max(-1.0, min(1.0, gx[q])))) for q in range(a, b + 1))
    zmin = min(z[q] for q in range(a, b + 1))
    print(f"  {v(t[a])} -> {t[b]:6.2f} s (video {t[b] * SLOW:6.2f} s)  nose down to {mx:.0f} deg, base down to {zmin:+.3f} m")
