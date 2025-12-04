import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Tracking-Flat-Lite12-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Lite12FlatEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite12FlatPPORunnerCfg",
    },
)

gym.register(
    id="Tracking-Flat-Lite12-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Lite12FlatWoStateEstimationEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite12FlatPPORunnerCfg",
    },
)


gym.register(
    id="Tracking-Flat-Lite12-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": flat_env_cfg.Lite12FlatLowFreqEnvCfg,
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Lite12FlatLowFreqPPORunnerCfg",
    },
)
