import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-Lite-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.LiteFlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiteFlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Lite-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.LiteFlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiteFlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Lite-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.LiteFlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:LiteFlatLowFreqPPORunnerCfg",
    },
)
