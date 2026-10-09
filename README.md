# Jumper on a trampoline — 跳跳、トランポリンに乗る

An entry for the **Crab Robot Open-source Challenge 2026** (螃蟹机器人开源动作大赛)
by KingKong Robotics, built on the official toolkit
[KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper).

A gymnastics and trampoline coach, and the AI partners she works with, teach a
six-legged robot to bounce on a trampoline bed: land without falling, hold the
landing through the bed's compression, push from the bottom of the stroke, and
go higher each time. Everything here is simulation for now. The robot is the
prize.

**Status (2026-10-09 night): stage 0 done; a hand-written reflex layer already bounces on the deep bed; stage 1 written and launchable, not yet trained.**

| stage | what | state |
|---|---|---|
| 0 | a trampoline bed scene; the shipped jump replayed on it; measurements of landing force, slip and torque; recordings | done |
| 0.5 | no learning: the shipped posture policy driven by hand-written reflexes (push on the sign of the vertical speed, swing the high side's legs in the air, thrust the arms on the push, counter the yaw) keeps bouncing on the deep bed | done: 8 to 9 cm for 12 s without a fall; up to 31 cm with falls |
| 1 | `jumper.bounce`: a reinforcement-learning task that keeps bouncing, paid per bounce for the apex and its growth | written, launches on CPU, **untrained** |
| 2 | air poses, a somersault (front or side), skipping rope | planned |
| 3 | the real robot on a real bed, once we have one | planned |

## What is in here

This repository is an **overlay**: its files live at the same paths as in the
toolkit and are copied over a checkout of it. Nothing from the toolkit is
redistributed.

| path | what |
|---|---|
| `scenes/trampoline.py` | the bed, as under-damped soft contact on the ground: `--bed-dampratio` is the bounce, `--bed-timeconst` the depth, `--bed-width` how deep it stays soft, `--bed-friction` the grip. The docstring carries the measurements behind the defaults |
| `tools/calibrate_bed.py` | drops a 2.5 kg stand-in on candidate beds and reports sink and restitution; turns a measured real bed into the dials |
| `docs/measurement.md` | how to measure a real bed with a weight, a ruler and a phone |
| `tasks/jumper/bounce/` | stage 1: the bouncing task, built on the toolkit's `jumper.ref_free_jump` |
| `tools/probes/` | the one-off probes and recording scripts, all wrapping the toolkit's `scripts/play.py`: `probe_terminations.py` and `probe_slip.py` (stage 0: terminations, foot slip, ground force), `probe_drop.py` (drop tests and the "arrow" at touchdown), `probe_bounce_dance.py` (a sine wave on the posture policy's height command), `probe_reflex_bounce*.py` (the reflex layer, versions 2 to 4), `probe_air_reaction.py` and `probe_lift_calib2.py` (which way the body turns when a leg swings in the air), `render_sim.py` (draws a replay in simulation time, portrait 1080 x 1920, with a mask of the robot), `play_camera.py`, `record_play.ps1`, `repeat_probe.ps1`. Each names its jumper checkout in a `REPO` constant at the top; change it to yours |
| `tools/install_into_jumper.py` | copies the scene, the calibration tool and the task into a jumper checkout and adds the two registration lines |

## Use it

```bash
git clone https://github.com/KingKongRobotics/jumper.git
cd jumper && pip install -e ".[dev]"            # the toolkit's own setup; see its docs/AGENT_SETUP.md
python /path/to/crab-robot-jumper-trampoline/tools/install_into_jumper.py .
python scripts/train.py --list                   # expect "jumper.bounce" and the "trampoline" scene
```

Replay the shipped jump on the bed (CPU is enough):

```bash
# the deep bed, the project's main ground: restitution 0.9, about 8 cm of sink for 2.5 kg from 0.5 m
python scripts/play.py --task jumper.jump --checkpoint tasks/jumper/jump/out/example/model_3100.pt \
    --scene trampoline --bed-timeconst 0.6 --backend native --device cpu
# the shallow bed, same restitution, 3 cm of sink: the shipped jump lands and stands on this one
python scripts/play.py --task jumper.jump --checkpoint tasks/jumper/jump/out/example/model_3100.pt \
    --scene trampoline --bed-timeconst 0.15 --backend native --device cpu
```

Train stage 1 (GPU; the toolkit's `warp` backend):

```bash
python scripts/train.py --task jumper.bounce --scene trampoline --bed-timeconst 0.6 \
    --backend warp --device cuda:0 --num_envs 2048 --headless --max-iterations 50   # time one iteration first
python scripts/train.py --task jumper.bounce --scene trampoline --bed-timeconst 0.6 \
    --backend warp --device cuda:0 --num_envs 2048 --headless
```

Match the simulated bed to a real one: `docs/measurement.md`, then `python tools/calibrate_bed.py --drop 0.5 --target-e <e> --target-sink-cm <x>`.

## What we have measured so far

All in simulation, on the toolkit's native (CPU) backend, 2026-10-09.

**The shipped jump (`jumper.jump`, trained on rigid ground) replayed on the bed**, five runs a condition:

| ground | fell | rise of the base | feet force at landing | rear-foot slide in 1.5 s |
|---|---|---|---|---|
| rigid (`studio`) | 0/5 | 0.12 to 0.15 m | 90 to 220 N (4 to 9 x weight) | 5 to 7 cm |
| shallow bed, timeconst 0.15, e 0.90 | 0/5 | 0.11 to 0.12 m | 93 to 125 N | 30 to 55 cm |
| deep bed, timeconst 0.60, e 0.90 | 3/9 | 0.12 to 0.15 m | 22 to 54 N (1 to 2 x weight) | 34 to 68 cm |
| deep bed, friction 2.5 | 0/5 | 0.03 m | 21 to 24 N | 70 to 82 cm |

The bed cuts the landing force to a quarter. Friction does not cure the slide:
the rigid-ground push meets a floor that gives, and more grip only pins the feet
while the legs sink. Joint torque sits at the servo's plateau (1.746 N·m) at
take-off and landing on every ground.

**The bed itself** (2.5 kg stand-in, `tools/calibrate_bed.py` and a probe of its contact):

| bed | contact per bounce | time to the bottom | sink | peak force / weight |
|---|---|---|---|---|
| deep, from 0.14 m | 122 ms | 60 ms | 6.7 cm | 6 |
| deep, from 0.35 m | 106 ms | 50 ms | 9.4 cm | 11 |
| shallow, from 0.14 to 0.35 m | 31 ms | 15 ms | 1.6 to 2.5 cm | 18 to 28 |

The force peaks at the bottom of the stroke, within a millisecond. The coach's
rule for the push, "start rising when the load starts to fall", is therefore the
same instant as "the body stops going down": either signal will do as the
trigger, and the stand-in's numbers are an upper bound for a robot whose legs
give.

**A hand-written reflex layer on the shipped posture policy** (`tools/probes/probe_reflex_bounce_v4.py`, deep bed, 12 s runs, no learning; the coach gave the cues live and the numbers were measured the same evening):

| reflex | what it does | result |
|---|---|---|
| vertical-speed lock | the height command switches to "push" a little before the bottom of the stroke and to "crouch, hold" a little before the apex; no clock, no sine wave | 8 to 9 cm hops for 12 s, no fall; the bounce finds the bed's rhythm itself (0.36 to 0.45 s a hop) |
| push past the policy's command range | the same, commanding 0.20 m instead of 0.15 | up to 16 cm, 2 to 4 falls in 12 s |
| air righting (R1) | in flight, swing the high side's legs from the hip at full range with the knee extended; the body turns the other way | falls 3.75 to 2.5 per 12 s, jump up to 20 cm |
| arms on the push | thrust the front arms forward during the push only, back for the landing | up to 31 cm; more falls |
| yaw reflex | in flight, swing every hip yaw joint against the body's spin | the twist that the uneven slip of the rear feet builds (473 degrees in 12 s) held to 18 |

The first attempt at the push, a state machine that crouched from rest and pushed, never left the ground: the push has to come at the bottom of a *moving* body. Every take-off on the deep bed carries about 0.4 m/s of forward speed into the landing (0.05 on rigid ground); the falls are the landings that arrive tilting and turning. Those are the things stage 1 is asked to learn, and the heights are its bar.

**The recording fed forward with no policy** (the toolkit's `high_jump_flat.npz`,
8 environments, domain randomisation on): rises 0.10 to 0.13 m on rigid ground,
0.06 m on the shallow bed, 0.02 to 0.06 m on the deep bed, where 2 or 3 of 8
runs slide a metre within 3 s. The bed takes about half of an open-loop jump;
a closed-loop policy loses only 2 cm on the shallow bed. Learning to keep that
on the deep bed is stage 1.

## Stage 1 in one paragraph

`jumper.bounce` keeps the toolkit's `ref_free_jump` as it is (200 Hz, kp=20, the
recording fed forward and faded out over training, the actor observing only its
body) and changes what an episode is: 12 s on the bed, from a HOME spawn dropped
a few centimetres onto it or from a point along the recording. A bounce is a
flight the episode pushed off itself; its apex is paid as a stream for one
second after the touchdown, so the only way to go on being paid is to bounce
again and the only way to be paid more is to bounce higher. Around that, the
coach's landing: hold the touchdown pose through the compression (`absorb`),
all six feet down within 20 ms (`touchdown_sync`), come down level and still
(`air_upright`), keep the tuck in the air (`air_posture`), stay on the bed
(`horizontal_drift`, and an episode ends a metre from where it started). Every
weight is a starting value; none has been set against a trained policy. The
module docstrings carry the reasoning and the numbers.

## The story

*To be written by the coach: why a trampoline, what the eye of a gymnastics
coach sees in a robot's landing, and what the robot got wrong first.* The
contest asks for the failures as much as the result, and the recordings of
stage 0 are mostly failures: the shipped jump lands on the deep bed and goes
over on the second bounce, every time.

## Team

A gymnastics and trampoline coach who runs a gym (the author and the eye), and
the AI partners she works with: **Island Master**, a Claude Code agent on the
gym's reception PC, who builds, measures and records; **Verti**, a Claude agent
who holds the rules, the plan and the reward design; and the rest of the
Ortiz-AI Partners team. Every number in this repository was produced by one of
them and checked by another.

## License

Apache License 2.0, the toolkit's own; see `LICENSE` and `NOTICE`.
Entry for the Crab Robot Open-source Challenge 2026: #Jumper #CrabRobot #OpenSourceRobot.

---

## 日本語の要約

金鋼科技（KingKong Robotics）の螃蟹ロボット「跳跳（Jumper）」の二創コンテストへのエントリー。体操・トランポリンのコーチと、彼女と働くAIパートナーが、六本脚のロボットにトランポリンの上で跳ね続けることを教える。転ばずに着地し、ベッドが沈む間は姿勢を保ち、一番下から押し、毎回少しずつ高く。いまはすべてシミュレーション。実機は賞品だ。

- `scenes/trampoline.py`：跳ねる床。`--bed-dampratio` が跳ね、`--bed-timeconst` が沈み
- `tools/calibrate_bed.py` と `docs/measurement.md`：本物のベッドを重りと定規とスマホで測り、ダイヤルに変換する
- `tasks/jumper/bounce/`：段階1「漕いで跳ね続ける」学習課題。書き上げて発車確認済み、未学習
- `tools/probes/`：計測・反射・描画の道具。学習なしの手書き反射（底の少し前で押す、空中で上がった側の脚を付け根から振る、押しで腕を突き出す、ヨーを打ち消す）で、深い床を 8〜9 cm で転ばず跳ね続け、最大 31 cm に届いた（転倒あり）。`render_sim.py` はシミュレーション時間で縦動画とマスクを描く
- `tools/install_into_jumper.py`：本家のチェックアウトに上書きして登録行を足す

段階0の数字（公式ジャンプを床で再生）：着地の接地力は硬い床の 90〜220 N に対し、深い床で 22〜54 N。4分の1になる。深い床では9回中3回、二回目のバウンスで転ぶ。摩擦を上げても滑りは直らない。

物語の章はコーチが書く。
