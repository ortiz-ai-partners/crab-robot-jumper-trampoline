"""PPO hyperparameters for jumper.bounce: `ref_free_jump`'s, with a longer run.

Nothing here has been set against a policy trained on this task. They are
`ref_free_jump`'s because the action, the rate, the observation and the prior's
schedule are its, and a bounce period (0.4 to 0.6 s) is inside the 0.5 s
horizon its discount buys at 200 Hz.
"""

from __future__ import annotations

from mjlab.rl import RslRlOnPolicyRunnerCfg

from ..common.ppo import jumper_ppo_baseline

#: Environment steps per policy update; `ref_free_jump`'s 0.24 s at 200 Hz.
#: A module constant because `env_cfg.py` writes its schedules in iterations.
NUM_STEPS_PER_ENV = 48


def agent_cfg() -> RslRlOnPolicyRunnerCfg:
    return jumper_ppo_baseline(
        experiment_name="jumper.bounce",
        actor_hidden_dims=(512, 256, 128, 64),
        critic_hidden_dims=(512, 256, 128, 64),
        init_std=0.3,
        entropy_coef=0.001,
        gamma=0.99**0.25,
        lam=0.95**0.25,
        num_steps_per_env=NUM_STEPS_PER_ENV,
        # The prior is gone by 3000 and the drop reaches its end range by 4000
        # (env_cfg.py); the rest is the objective on its own, on the full drop.
        max_iterations=8000,
    )
