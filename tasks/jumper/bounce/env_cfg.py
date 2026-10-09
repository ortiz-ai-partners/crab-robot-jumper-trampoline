"""jumper.bounce -- keep bouncing on a trampoline bed, as high as it can.

Built on `jumper.ref_free_jump`'s config and kept as close to it as the task
allows: the same 200 Hz and kp=20, the same absolute action around HOME with
the recording fed forward and faded out, the same actor observation (body only,
no clock, no contact), the same prior cross-fade. What changes is what an
episode is and what it is paid for.

## The episode

Up to `EPISODE_S` on the bed, from one of two spawns: HOME, dropped from
`DROP_RANGE` above its stance so the first thing it meets is a landing on a bed
that gives (the coach's "start low, learn the air"); or, for `RSI_FRACTION` of
episodes, somewhere along the recording, so the recording's crouch, push and
landing are practised as in `ref_free_jump`. It ends when the robot falls
(`FELL_OVER_RAD`), leaves the bed (`DRIFT_OUT_M`), or time runs out.

## What it is paid for

Each counted bounce (a flight it pushed off itself, apex `min_rise` or more
above HOME, ended by a touchdown; `mdp/commands.py`) pays its apex as a stream
for `APEX_EXPIRE_S` after the touchdown (`bounce_apex`), its height in the air
while it is up there (`flight_height`), and how much higher it went than the
bounce before (`apex_growth`). The stream's expiry is a bounce period, so the
only way to go on being paid is to bounce again, and the only way to be paid
more is to bounce higher.

Around that, the coach's landing: hold the touchdown pose through the bed's
compression (`absorb`, charged), all six feet down within 20 ms of the first
(`touchdown_sync`, charged), come down level and not turning (`air_upright`),
keep the tuck in the air (`air_posture`), and stay where you started
(`horizontal_drift`). The penalties `ref_free_jump` carries -- a non-foot on
the ground, legs on each other, joint limits, action rate -- stay as they are.

## The bed is the scene's

This task owns no floor. Train it with `--scene trampoline` and the bed's dials
(`--bed-timeconst 0.6` is the deep bed the project set as its main ground on
2026-10-09, `0.15` the shallow one it compares against). Without a scene it is
a jump task on a hard floor, which is a control, not a mistake.

## On the numbers

Every weight below is a starting point, set against `ref_free_jump`'s budget
(an apex as high as the recording's pays 4 x 1.2 a step there) and against the
Island Master's replays of the shipped jump on the bed (2026-10-09, receiving
PC, native:cpu, five runs a condition): landing force 22 to 54 N on the deep bed
against 90 to 220 N rigid, rear feet sliding 34 to 68 cm in 1.5 s, falls on 3
of 9 second bounces. None has yet been set against a policy trained here.

What a hand-written reflex layer on the shipped posture policy found the same
evening (Island Master and the coach, 2026-10-09; `tools/probes/` in the
project repository) is the technique this task is meant to learn, and sets
the bar: switching the body-height command on the sign of the base's vertical
speed, a little before the bottom and a little before the apex, bounces 8 to
9 cm for 12 s on the deep bed without a fall; pushing past the posture
policy's command range reaches 16 cm and falls 2 to 4 times in 12 s; a
full-range hip swing with the knee extended in the air cuts those falls to
2.5 and lifts the jump to 20 cm; the front arms thrust forward during the
push only, 31 cm at best. Every take-off on the deep bed carries 0.4 m/s of
forward speed into the landing (0.05 on rigid ground), which is where the
slide comes from, and the yaw adds up from the two rear feet sliding by
different amounts. The reward terms stand as they were; `air_posture` lets
go of the tuck as the body tilts, and `air_upright` counts the yaw, both
because of these measurements.
"""

from __future__ import annotations

import math
from pathlib import Path

from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs import mdp as envs_mdp
from mjlab.managers import TerminationTermCfg
from mjlab.managers.curriculum_manager import CurriculumTermCfg
from mjlab.managers.event_manager import EventTermCfg
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.managers.reward_manager import RewardTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.tasks.velocity import mdp

from ..common.constants import GAIT_JOINTS
from ..ref_free_jump import env_cfg as ref_free
from ..ref_free_jump import mdp as ref_free_mdp
from . import mdp as bounce_mdp
from .rl_cfg import NUM_STEPS_PER_ENV

# ── The episode ────────────────────────────────────────────────────────────────
#: The coach's "jump until you fail": long enough for twenty bounces at the
#: 0.4 to 0.6 s a bounce period takes at 0.1 to 0.35 m of apex.
EPISODE_S = 12.0

#: Metres above its stance a HOME spawn is dropped from, at iteration 0 and
#: from `DROP_END_ITERATIONS` on. The shipped jump's apex is 0.12 to 0.15 m and
#: it falls on the deep bed from there, so the start is below that and the end
#: is the apex the project is after.
DROP_RANGE_START = (0.03, 0.08)
DROP_RANGE_END = (0.10, 0.30)
DROP_HOLD_ITERATIONS = 1000
DROP_END_ITERATIONS = 4000
#: The drop in replay: fixed, so every attempt starts the same.
DROP_PLAY = (0.10, 0.10)

#: Share of episodes spawned on the recording, as `ref_free_jump`.
RSI_FRACTION = 0.5
RSI_PHASE_RANGE = (0.0, 0.9)

#: For a dropped spawn, the go comes this long after its first touchdown
#: (`mdp/commands.py`): the deep bed's stroke, there and back. Measured with
#: the 2.5 kg stand-in of tools/calibrate_bed.py (timeconst 0.6, dampratio
#: 0.03, width 0.25): 100 to 122 ms in contact per bounce from 0.14 to 0.50 m,
#: 48 to 60 ms of it going down; the shallow bed (0.15) 30 ms and 15 ms.
GO_AFTER_LANDING_S = 0.10

# ── The objective ──────────────────────────────────────────────────────────────
#: Metres per unit of height reward; `ref_free_jump`'s.
HEIGHT_SCALE = 0.10
#: Seconds the apex stream lasts after a touchdown: about one bounce period
#: at the heights above, with margin. Longer, and standing still after a good
#: jump is paid; shorter, and the stream ends before the next push.
APEX_EXPIRE_S = 1.0
#: End weights of the three height terms, from `PRIOR_END_ITERATIONS` on; 0
#: until `PRIOR_HOLD_ITERATIONS`, as `ref_free_jump`'s cross-fade.
APEX_WEIGHT = 4.0
FLIGHT_WEIGHT = 2.0
GROWTH_WEIGHT = 1.0
#: Metres of growth that pay 1 at full weight, and the clip on it, so one
#: lucky bounce cannot buy more than a steady climb.
GROWTH_CLIP = 1.0

# ── The landing ────────────────────────────────────────────────────────────────
#: `absorb`: (mean squared rad from the touchdown pose) / sigma_sq, capped.
#: 0.1 rad of give costs 1 a step at this weight, against the 4.8 a step a
#: recording-height bounce pays.
ABSORB_WEIGHT = -1.0
ABSORB_SIGMA_SQ = 0.01
ABSORB_CAP = 4.0
#: Seconds after a touchdown in which `absorb` is charged: the deep bed's
#: whole contact (above), so the downstroke is covered at every depth and a
#: robot standing on the bed afterwards is not.
ABSORB_WINDOW_S = 0.12
#: `touchdown_sync`: per foot still up, per step of the window.
SYNC_WEIGHT = -1.0
SYNC_WINDOW_STEPS = 4
#: `air_upright`, `air_posture`: `ref_free_jump`'s kernels for the same
#: quantities (upright 0.10, home 0.05).
AIR_UPRIGHT_WEIGHT = 1.0
AIR_UPRIGHT_SIGMA_SQ = 0.10
AIR_OMEGA_SIGMA_SQ = 2.0
AIR_POSTURE_WEIGHT = 1.0
AIR_POSTURE_SIGMA_SQ = 0.05
#: The tuck is asked for while the body is level, and let go of as it tilts,
#: so the legs are free to right it (`mdp/rewards.py::air_posture`). The
#: upright kernel's width.
AIR_POSTURE_LEVEL_SIGMA_SQ = 0.10
#: Upright throughout, as `ref_free_jump`.
UPRIGHT_WEIGHT = 1.0
UPRIGHT_SIGMA_SQ = 0.10
#: Drift: 0.30 m costs what upright pays. Wider than `ref_free_jump`'s 0.10
#: because twenty landings on a bed will wander, and `drifted_off` is the
#: hard edge.
DRIFT_WEIGHT = -1.0
DRIFT_SCALE = 0.30

#: Newtons on a part that is not a foot before `illegal_contact` (weight -5,
#: `ref_free_jump`'s) counts it. `ref_free_jump` uses 1 N on rigid ground. On
#: the bed the ground gives under the feet and the six lower leg links rest
#: on it at 2 to 3 N whenever the robot stands, so at 1 N the term read 85% of
#: steps on either bed with the recording fed forward (8 envs, 4 s, DR on,
#: native:cpu). Measured on the same runs, the largest non-foot force per step:
#: shallow bed, standing, never over 5 N; deep bed, standing, over 5 N in 26%
#: of steps, over 10 N in 10%, over 20 N in none; over the whole run 1% of
#: steps over 20 N on either bed, in the recording's crouch. 20 N is most of
#: the robot's weight (25 N): a part carrying the body, not one brushing the
#: bed.
ILLEGAL_CONTACT_FORCE_N = 20.0

# ── Terminations ───────────────────────────────────────────────────────────────
FELL_OVER_RAD = math.radians(50.0)
#: Half a competition bed is 1.07 m (4.28 x 2.14 m); a metre from the centre
#: is the edge, with the robot's own half-length as margin. An assumption
#: about the bed the project films on, to be replaced by its measurement.
DRIFT_OUT_M = 1.0


def env_cfg(asset: Path | None = None, play: bool = False) -> ManagerBasedRlEnvCfg:
    """Bouncing on a bed, from `ref_free_jump`'s config.

    Args:
        asset: path to the model XML, from `--model`. None uses the default jumper.xml.
        play: replay mode -- noise and pushes off, HOME spawns dropped from
            `DROP_PLAY`, no schedule.
    """
    cfg = ref_free.env_cfg(asset=asset, play=play)
    gait_cfg = SceneEntityCfg("robot", joint_names=list(GAIT_JOINTS))

    # ── Command: the go, with bounce bookkeeping ──────────────────────────────
    base = cfg.commands["jump"]
    cfg.commands["jump"] = bounce_mdp.commands.BounceCommandCfg(
        entity_name=base.entity_name,
        motion_file=base.motion_file,
        command_delay_range=base.command_delay_range,
        rsi_fraction=0.0 if play else RSI_FRACTION,
        rsi_phase_range=RSI_PHASE_RANGE,
        go_after_landing_s=GO_AFTER_LANDING_S,
    )

    # ── Observations: the critic sees the bookkeeping ─────────────────────────
    critic = cfg.observations["critic"].terms
    critic.pop("jump_state", None)
    critic["bounce_state"] = ObservationTermCfg(
        func=bounce_mdp.observations.bounce_state,
        params={"command_name": "jump", "scale": HEIGHT_SCALE, "mirror_kind": "scalar"},
    )

    # ── Rewards ───────────────────────────────────────────────────────────────
    keep = {
        name: cfg.rewards[name]
        for name in (
            "prior_joint_pos", "prior_joint_vel", "illegal_contact", "leg_collision",
            "dof_pos_limits", "action_rate_l2",
        )
    }
    keep["illegal_contact"].params = {
        **keep["illegal_contact"].params, "force_threshold": ILLEGAL_CONTACT_FORCE_N,
    }
    cfg.rewards = {
        **keep,
        "bounce_apex": RewardTermCfg(
            func=bounce_mdp.rewards.bounce_apex, weight=0.0,
            params={"command_name": "jump", "scale": HEIGHT_SCALE,
                    "expire_s": APEX_EXPIRE_S},
        ),
        "flight_height": RewardTermCfg(
            func=bounce_mdp.rewards.flight_height, weight=0.0,
            params={"command_name": "jump", "scale": HEIGHT_SCALE},
        ),
        "apex_growth": RewardTermCfg(
            func=bounce_mdp.rewards.apex_growth, weight=0.0,
            params={"command_name": "jump", "scale": HEIGHT_SCALE,
                    "clip": GROWTH_CLIP, "expire_s": APEX_EXPIRE_S},
        ),
        "absorb": RewardTermCfg(
            func=bounce_mdp.rewards.absorb, weight=ABSORB_WEIGHT,
            params={"asset_cfg": gait_cfg, "command_name": "jump",
                    "sigma_sq": ABSORB_SIGMA_SQ, "cap": ABSORB_CAP,
                    "window_s": ABSORB_WINDOW_S},
        ),
        "touchdown_sync": RewardTermCfg(
            func=bounce_mdp.rewards.touchdown_sync, weight=SYNC_WEIGHT,
            params={"command_name": "jump", "window_steps": SYNC_WINDOW_STEPS},
        ),
        "air_upright": RewardTermCfg(
            func=bounce_mdp.rewards.air_upright, weight=AIR_UPRIGHT_WEIGHT,
            params={"command_name": "jump", "sigma_sq": AIR_UPRIGHT_SIGMA_SQ,
                    "omega_sigma_sq": AIR_OMEGA_SIGMA_SQ},
        ),
        "air_posture": RewardTermCfg(
            func=bounce_mdp.rewards.air_posture, weight=AIR_POSTURE_WEIGHT,
            params={"asset_cfg": gait_cfg, "command_name": "jump",
                    "sigma_sq": AIR_POSTURE_SIGMA_SQ,
                    "level_sigma_sq": AIR_POSTURE_LEVEL_SIGMA_SQ},
        ),
        "upright": RewardTermCfg(
            func=ref_free_mdp.rewards.upright, weight=UPRIGHT_WEIGHT,
            params={"sigma_sq": UPRIGHT_SIGMA_SQ},
        ),
        "horizontal_drift": RewardTermCfg(
            func=ref_free_mdp.rewards.horizontal_drift, weight=DRIFT_WEIGHT,
            params={"scale": DRIFT_SCALE},
        ),
    }

    # ── Curriculum: the prior comes down, the heights and the drop go up ──────
    def _cross_fade(term: str, start: float, end: float) -> CurriculumTermCfg:
        return CurriculumTermCfg(
            func=bounce_mdp.curriculum.anneal_reward_weight,
            params={
                "term_name": term, "start_weight": start, "end_weight": end,
                "start_step": NUM_STEPS_PER_ENV * ref_free.PRIOR_HOLD_ITERATIONS,
                "end_step": NUM_STEPS_PER_ENV * ref_free.PRIOR_END_ITERATIONS,
            },
        )

    cfg.curriculum = {
        "prior_joint_pos_weight": _cross_fade(
            "prior_joint_pos", ref_free.PRIOR_POS_WEIGHT, 0.0),
        "prior_joint_vel_weight": _cross_fade(
            "prior_joint_vel", ref_free.PRIOR_VEL_WEIGHT, 0.0),
        "bounce_apex_weight": _cross_fade("bounce_apex", 0.0, APEX_WEIGHT),
        "flight_height_weight": _cross_fade("flight_height", 0.0, FLIGHT_WEIGHT),
        "apex_growth_weight": _cross_fade("apex_growth", 0.0, GROWTH_WEIGHT),
        "drop_height": CurriculumTermCfg(
            func=bounce_mdp.curriculum.anneal_drop_height,
            params={
                "term_name": "drop",
                "start_range": DROP_RANGE_START, "end_range": DROP_RANGE_END,
                "start_step": NUM_STEPS_PER_ENV * DROP_HOLD_ITERATIONS,
                "end_step": NUM_STEPS_PER_ENV * DROP_END_ITERATIONS,
            },
        ),
        "feedforward": CurriculumTermCfg(func=bounce_mdp.curriculum.report_feedforward),
    }

    # ── Terminations: no "back home"; the episode is the bouncing ─────────────
    cfg.terminations = {
        "time_out": TerminationTermCfg(func=envs_mdp.time_out, time_out=True),
        "fell_over": TerminationTermCfg(
            func=mdp.bad_orientation, params={"limit_angle": FELL_OVER_RAD}
        ),
        "drifted_off": TerminationTermCfg(
            func=bounce_mdp.terminations.drifted_off, params={"limit": DRIFT_OUT_M}
        ),
    }

    # ── Events: HOME, then the drop, then RSI over its share ──────────────────
    drop = EventTermCfg(
        func=envs_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"z": DROP_PLAY if play else DROP_RANGE_START},
            "velocity_range": {},
        },
    )
    events = {}
    for name, term in cfg.events.items():
        events[name] = term
        if name == "reset_to_default":
            events["drop"] = drop
    assert "drop" in events, "ref_free_jump's reset_to_default event has moved"
    events["reset_from_reference"].params["rsi_fraction"] = 0.0 if play else RSI_FRACTION
    cfg.events = events

    # ── Episode and viewer ────────────────────────────────────────────────────
    cfg.episode_length_s = EPISODE_S
    cfg.viewer.distance = 1.2
    cfg.viewer.elevation = -10.0

    if play:
        cfg.curriculum = {}

    return cfg


__all__ = ["env_cfg"]
