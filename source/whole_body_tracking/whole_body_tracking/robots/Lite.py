import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_A10020 = 0.07031  # 0.000488265*12**2    # hip_pitch , knee
ARMATURE_A8116 = 0.05963   # 0.000184053*18**2     # hip_roll
ARMATURE_A6408 = 0.03891   # 0.000062254*25**2     # hip_yaw ,ankle_pitch ,ankle_roll

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

# STIFFNESS_A10020 = ARMATURE_A10020 * NATURAL_FREQ**2
# STIFFNESS_A6408 = ARMATURE_A6408 * NATURAL_FREQ**2
# STIFFNESS_A8116 = ARMATURE_A8116 * NATURAL_FREQ**2
STIFFNESS_A10020 = 280
STIFFNESS_A8116 = 240
STIFFNESS_A6408 = 150

DAMPING_A10020 = 14
DAMPING_A8116 = 12
DAMPING_A6408 = 10

Lite_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/miniloong/urdf/miniloong_fixed.urdf",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            "joint_.*_hip_pitch": 0.39,
            "joint_.*_hip_roll":0.,
            "joint_.*_hip_yaw":-0.12,
            "joint_.*_knee": 0.74,
            "joint_.*_ankle_pitch": 0.36,
            "joint_.*_ankle_roll": 0.,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                "joint_.*_hip_pitch",
                "joint_.*_hip_roll",
                "joint_.*_hip_yaw",                
                "joint_.*_knee",
            ],
            effort_limit_sim={
                "joint_.*_hip_pitch": 150.0,
                "joint_.*_hip_roll": 60,
                "joint_.*_hip_yaw": 150.0,
                "joint_.*_knee": 150.0,
            },
            velocity_limit_sim={
                "joint_.*_hip_pitch": 18.0,
                "joint_.*_hip_roll": 18.0,
                "joint_.*_hip_yaw": 18.0,
                "joint_.*_knee": 18.0,
            },
            stiffness={
                "joint_.*_hip_pitch": STIFFNESS_A10020,
                "joint_.*_hip_roll": STIFFNESS_A8116,
                "joint_.*_hip_yaw": STIFFNESS_A6408,
                "joint_.*_knee": STIFFNESS_A10020,
            },
            damping={
                "joint_.*_hip_pitch": DAMPING_A10020,
                "joint_.*_hip_roll": DAMPING_A8116,
                "joint_.*_hip_yaw": DAMPING_A6408,
                "joint_.*_knee": DAMPING_A10020,
            },
            armature={
                "joint_.*_hip_pitch": ARMATURE_A10020,
                "joint_.*_hip_roll": ARMATURE_A8116,
                "joint_.*_hip_yaw": ARMATURE_A6408,
                "joint_.*_knee": ARMATURE_A10020,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim= 60.0,
            velocity_limit_sim= 18.0,
            joint_names_expr= ["joint_.*_ankle_pitch", "joint_.*_ankle_roll"],
            stiffness= STIFFNESS_A6408,
            damping= DAMPING_A6408,
            armature= ARMATURE_A6408,
        ),
    },
)

Lite_ACTION_SCALE = {}
for a in Lite_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            Lite_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
