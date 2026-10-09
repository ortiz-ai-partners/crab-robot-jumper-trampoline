# Jumper on a trampoline — 跳跳、トランポリンに乗る

An entry for the **Crab Robot Open-source Challenge 2026** (螃蟹机器人开源动作大赛)
by KingKong Robotics, built on the official toolkit
[KingKongRobotics/jumper](https://github.com/KingKongRobotics/jumper).

A gymnastics and trampoline coach, and the AI partners she works with, teach a
six-legged robot to bounce on a trampoline bed: land without falling, hold the
landing through the bed's compression, push from the bottom of the stroke, and
go higher each time. Everything here is simulation for now. The robot is the
prize.

**Status (2026-10-09): stage 0 done, stage 1 written and launchable, not yet trained.**

| stage | what | state |
|---|---|---|
| 0 | a trampoline bed scene; the shipped jump replayed on it; measurements of landing force, slip and torque; six recordings | done |
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
| `tools/probes/` | the one-off probes and recording scripts used for stage 0 (terminations, foot slip and ground force, camera placement, screen recording) |
| `tools/install_into_jumper.py` | copies the files above into a jumper checkout and adds the two registration lines |

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
- `tools/probes/`：段階0の計測と録画に使った道具
- `tools/install_into_jumper.py`：本家のチェックアウトに上書きして登録行を足す

段階0の数字（公式ジャンプを床で再生）：着地の接地力は硬い床の 90〜220 N に対し、深い床で 22〜54 N。4分の1になる。深い床では9回中3回、二回目のバウンスで転ぶ。摩擦を上げても滑りは直らない。

物語の章はコーチが書く。
