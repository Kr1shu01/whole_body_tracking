import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from whole_body_tracking.assets import ASSET_DIR

ARMATURE_P2_24=476.222*1e-6*24*24    #hip pitch 0.274303872      
ARMATURE_P1_12=488.265*1e-6*12*12    #hip roll and knee 0.07031016
ARMATURE_P1_18H=192.82*1e-6*18*18     #hip yaw 0.06247368
ARMATURE_P1_18=146.69*1e-6*18*18      #ankle pitch and roll 0.04752756
ARMATURE_P2_36=25.5*1e-6*36*36        #arm 0.033048

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

Lite12_CYLINDER_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ASSET_DIR}/lite1_2/lite1_2.urdf",
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
        pos=(0.0, 0.0, 0.97),
        joint_pos={
            ".*_hip_pitch_joint": -0.1,
            ".*_knee_joint": 0.2,
            ".*_ankle_pitch_joint": -0.1,
            ".*_elbow_joint": 1.0,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 130.0,
                ".*_hip_roll_joint": 150.0,
                ".*_hip_pitch_joint": 330.0,
                ".*_knee_joint": 150.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": 280,
                ".*_hip_roll_joint": 240,
                ".*_hip_yaw_joint": 150,
                ".*_knee_joint": 280,
            },
            damping={
                ".*_hip_pitch_joint": 14,
                ".*_hip_roll_joint": 12,
                ".*_hip_yaw_joint": 10,
                ".*_knee_joint": 14,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_P2_24,
                ".*_hip_roll_joint": ARMATURE_P1_12,
                ".*_hip_yaw_joint": ARMATURE_P1_18H,
                ".*_knee_joint": ARMATURE_P1_12,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=90.0,
            velocity_limit_sim=16.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=50,
            damping=5,
            armature=ARMATURE_P1_18,
        ),
        # "waist": ImplicitActuatorCfg(
        #     effort_limit_sim=50,
        #     velocity_limit_sim=37.0,
        #     joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
        #     stiffness=2.0 * STIFFNESS_5020,
        #     damping=2.0 * DAMPING_5020,
        #     armature=2.0 * ARMATURE_5020,
        # ),
        # "waist_yaw": ImplicitActuatorCfg(
        #     effort_limit_sim=88,
        #     velocity_limit_sim=32.0,
        #     joint_names_expr=["joint_waist_yaw"],
        #     stiffness=STIFFNESS_7520_14,
        #     damping=DAMPING_7520_14,
        #     armature=ARMATURE_7520_14,
        # ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 97.0,
                ".*_shoulder_roll_joint": 97.0,
                ".*_shoulder_yaw_joint": 27.0,
                ".*_elbow_joint": 27.0,
                ".*_wrist_roll_joint": 7.0,
                ".*_wrist_pitch_joint": 7.0,
                ".*_wrist_yaw_joint": 7.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 6.28,
                ".*_shoulder_roll_joint": 6.28,
                ".*_shoulder_yaw_joint": 10.47,
                ".*_elbow_joint": 10.47,
                ".*_wrist_roll_joint": 20.94,
                ".*_wrist_pitch_joint": 20.94,
                ".*_wrist_yaw_joint": 20.94,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 80,
                ".*_shoulder_roll_joint": 80,
                ".*_shoulder_yaw_joint": 80,
                ".*_elbow_joint": 80,
                ".*_wrist_roll_joint": 16,
                ".*_wrist_pitch_joint": 16,
                ".*_wrist_yaw_joint": 16,
            },
            damping={
                ".*_shoulder_pitch_joint": 8,
                ".*_shoulder_roll_joint": 8,
                ".*_shoulder_yaw_joint": 8,
                ".*_elbow_joint": 8,
                ".*_wrist_roll_joint": 1.6,
                ".*_wrist_pitch_joint": 1.6,
                ".*_wrist_yaw_joint": 1.6,
            },
            armature={
                ".*_shoulder_pitch_joint": 0.03,
                ".*_shoulder_roll_joint": 0.03,
                ".*_shoulder_yaw_joint": 0.003,
                ".*_elbow_joint": 0.003,
                ".*_wrist_roll_joint": 0.0005,
                ".*_wrist_pitch_joint": 0.0005,
                ".*_wrist_yaw_joint": 0.0005,
            },
        ),
    },
)

Lite12_ACTION_SCALE = {}
for a in Lite12_CYLINDER_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            Lite12_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
