from isaaclab.utils import configclass

from whole_body_tracking.robots.Lite12 import Lite12_ACTION_SCALE, Lite12_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.Lite12.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class Lite12FlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Lite12_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Lite12_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "left_hip_pitch_link",
            "left_hip_roll_link",
            "left_hip_yaw_link",
            "left_knee_link",
            "left_ankle_pitch_link",
            "left_ankle_roll_link",
            "right_hip_pitch_link",
            "right_hip_roll_link",
            "right_hip_yaw_link",
            "right_knee_link",
            "right_ankle_pitch_link",
            "right_ankle_roll_link",
        ]


@configclass
class Lite12FlatWoStateEstimationEnvCfg(Lite12FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class Lite12FlatLowFreqEnvCfg(Lite12FlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
