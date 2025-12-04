from isaaclab.utils import configclass

from whole_body_tracking.robots.Lite import Lite_ACTION_SCALE, Lite_CYLINDER_CFG
from whole_body_tracking.tasks.tracking.config.Lite.agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE
from whole_body_tracking.tasks.tracking.tracking_env_cfg import TrackingEnvCfg


@configclass
class LiteFlatEnvCfg(TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = Lite_CYLINDER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = Lite_ACTION_SCALE
        self.commands.motion.anchor_body_name = "base_link"
        self.commands.motion.body_names = [
            "base_link",
            "link_left_hip_pitch",
            "link_left_hip_roll",
            "link_left_hip_yaw",
            "link_left_knee",
            "link_left_ankle_pitch",
            "link_left_ankle_roll",
            "link_right_hip_pitch",
            "link_right_hip_roll",
            "link_right_hip_yaw",
            "link_right_knee",
            "link_right_ankle_pitch",
            "link_right_ankle_roll",
        ]


@configclass
class LiteFlatWoStateEstimationEnvCfg(LiteFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class LiteFlatLowFreqEnvCfg(LiteFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
