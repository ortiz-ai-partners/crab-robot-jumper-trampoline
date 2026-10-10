#!/usr/bin/env bash
# Re-render the lesson takes with faces (Velti's request, Notion "🎭 島へ：顔つきで撮り直し", 10/10).
#   faces_take.sh LESSON SEED [scout|final] [CX CY R]
# scout: 270x480, to read the face log, the falls and where the base wanders, about a minute or two a take.
# final: 1080x1920, no mask. Lesson 1 (R04, hard floor) without the net; lessons 2-8 on the net bed with no frame,
# following the crab; lesson 9 (R49) on the round trampoline (CX CY R; the take's own: 0.02 -0.42 1.2).
# The seeds the island chose on 10/11: 1 1, 2 2, 3 4, 4 5, 5 3, 6 4, 7 1, 8a 2, 8b 1, 9 3.
# Every take runs with the env's seed fixed (seeded.py, or the gymnast's own --seed), so scout and final match.
set -u
L=$1; SEED=$2; MODE=${3:-scout}
T=$(cd "$(dirname "$0")" && pwd)                 # this probes folder
cd "${JUMPER_REPO:-/d/IslandMaster/jumper}"      # the jumper checkout (the reception PC's by default)
PY=${PY:-./.venv/Scripts/python.exe}
export FACE=auto DISH_MODE=hex DISH_FALLOFF=0.25 DISH_GAIN=1.0 DISH_LIFT=0 ROUND_HEIGHT=0.25 ROUND_SAG=0.67
if [ "$MODE" = scout ]; then
  export RENDER_SIZE=270x480 ROUND_R=3.0 ROUND_CX=0 ROUND_CY=0; OUT=out/faces/scout
else
  unset RENDER_SIZE; export ROUND_CX=${4:-0} ROUND_CY=${5:-0} ROUND_R=${6:-1.2}; OUT=out/faces/final
fi
mkdir -p $OUT
POSTURE="--task jumper.posture --checkpoint tasks/jumper/posture/out/example/model_74800.pt --scene trampoline --backend native --device cpu --headless --steps 2400 --speed 0"
ZERO="--task jumper.posture --agent zero --scene trampoline --backend native --device cpu --headless --speed 0"
# 698, not 700: at step 700 the episode resets, and the jump of the base reads as a fall (> <) on the last frame
JUMP="--task jumper.jump --checkpoint tasks/jumper/jump/out/example/model_3100.pt --backend native --device cpu --headless --steps 698 --speed 0"
GYM="--amp 5.0 --arms 3.0 --vz-up -0.8 --flex 2.5 --nose-wave flight --slow 3"
N=$OUT/L${L}_s${SEED}
# Velti's latest (10/11): R04 (lesson 1, hard floor) without the net; lessons 2-8 on the net bed with no frame,
# following the crab (ROUND_R unset); only R49 (lesson 9) on the round trampoline. Finals without the mask.
if [ "$MODE" = final ] && [ "$L" != 9 ]; then unset ROUND_R ROUND_CX ROUND_CY; fi
NM=""; [ "$MODE" = final ] && NM="--no-mask"
case $L in
  1) export FACE_BASE=neutral RENDER_DISH=0
     cmd=($T/seeded.py $SEED $T/render_sim.py --out $N.mp4 --slow 3 $NM -- $JUMP --scene studio) ;;
  2) export FACE_BASE=neutral
     cmd=($T/seeded.py $SEED $T/render_sim.py --out $N.mp4 --slow 3 $NM -- $JUMP --scene trampoline) ;;
  3) cmd=($T/seeded.py $SEED $T/probe_bounce_dance.py --amp 0.04 --freq 3.0 --start 1.0 --slow 3 $NM --render $N.mp4 --out $N.csv -- $POSTURE) ;;
  4) cmd=($T/seeded.py $SEED $T/probe_reflex_bounce.py --mode vzlock --h-air 0.09 --h-push 0.15 --vz-up -0.2 --vz-down 0.1 --slow 3 $NM --render $N.mp4 --out $N.csv -- $POSTURE) ;;
  5) cmd=($T/seeded.py $SEED $T/probe_reflex_bounce_v3.py --mode vzlock --h-air 0.09 --h-push 0.20 --no-clamp --vz-up -0.2 --vz-down 0.1 --r1 --r1-bang --r1-kp 2.0 --r1-kd 0.3 --r1-umax 3.0 --r1-dead 0.1 --r1-extend --slow 3 $NM --render $N.mp4 --out $N.csv -- $POSTURE) ;;
  6) cmd=($T/seeded.py $SEED $T/probe_reflex_bounce_v4.py --mode vzlock --h-air 0.09 --h-push 0.20 --no-clamp --vz-up -0.2 --vz-down 0.1 --r1 --r1-bang --r1-kp 2.0 --r1-kd 0.3 --r1-umax 3.0 --r1-dead 0.1 --r1-extend --arms-forward 2.0 --arms-ext 2.0 --arms-air-off --slow 3 $NM --render $N.mp4 --out $N.csv -- $POSTURE) ;;
  7) cmd=($T/seeded.py $SEED $T/probe_squat.py --amp 3.0 --thigh 1.0 --freq 3.4 --slow 3 $NM --render $N.mp4 --out $N.csv -- $ZERO --steps 2400) ;;
  8a) cmd=($T/probe_gymnast_v5.py --seed $SEED $GYM --legacy-no-reset $NM --render $N.mp4 --out $N.csv -- $ZERO --steps 2400) ;;
  8b) cmd=($T/probe_gymnast_v5.py --seed $SEED $GYM --legacy-no-reset $NM --render $N.mp4 --out $N.csv -- $ZERO --bed-friction 2.7 --steps 3000) ;;
  9) cmd=($T/probe_gymnast_v5.py --seed $SEED $GYM --legacy-no-reset --elev -30 --dist 2.4 $NM --render $N.mp4 --out $N.csv -- $ZERO --bed-friction 3.6 --steps 2400) ;;
  *) echo "unknown lesson $L"; exit 2 ;;
esac
t0=$(date +%s)
$PY -I "${cmd[@]}" > $N.log 2>&1
echo "== L$L seed $SEED $MODE ($(( $(date +%s) - t0 )) s) -> $N.mp4"
grep -E "^\[render\] (base|face:)|apex|falls|fall|episode ends|terminat|exited|Error" $N.log | grep -v "^\[render\] bed" | cut -c1-400
