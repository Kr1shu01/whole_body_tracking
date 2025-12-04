import os
import time
import argparse
import onnx
import onnxruntime as ort
import mujoco
import mujoco.viewer
import numpy as np
import yaml
import pinocchio as pin
# from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

LEGGED_GYM_ROOT_DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
# config_file = f"{LEGGED_GYM_ROOT_DIR}/deploy/config/g1.yaml"
config_file = f"{LEGGED_GYM_ROOT_DIR}/deploy/config/lite.yaml"
# parser = argparse.ArgumentParser()
# parser.add_argument("--config_file", type=str, help="config file name in the config folder")
# args = parser.parse_args()

# 默认顺序到实际顺序的索引映射
mujoco_to_policy_idx =  [
    0, 6, 1, 7, 2, 8,
    3, 9, 4, 10, 5, 11]

# 反向映射：实际顺序到默认顺序
policy_to_mujoco_idx = np.argsort(mujoco_to_policy_idx)

def convert_joint_order(joint, to_policy=True):
    if to_policy:
        return joint[mujoco_to_policy_idx]
    else:
        return joint[policy_to_mujoco_idx]

def pd_control(target_q, q, kp, target_dq, kd):
    """Calculates torques from position commands"""
    return (target_q - q) * kp - target_dq * kd

# ========== 四元数工具 ==========
def quat_inv(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q)
    w, x, y, z = np.moveaxis(q, -1, 0)
    norm2 = w*w + x*x + y*y + z*z
    return np.stack([w, -x, -y, -z], axis=-1) / norm2[..., None]

def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    w1, x1, y1, z1 = np.moveaxis(q1, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(q2, -1, 0)
    w = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z = w1*z2 + x1*y2 - y1*x2 + z1*w2
    return np.stack([w, x, y, z], axis=-1)

def relative_quat(q01: np.ndarray, q02: np.ndarray | None = None) -> np.ndarray:
    q10 = quat_inv(q01)
    return quat_mul(q10, q02) if q02 is not None else q10

def matrix_from_quat(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    w, x, y, z = np.moveaxis(q, -1, 0)
    N = w*w + x*x + y*y + z*z
    s = 2.0 / N
    wx, wy, wz = s*w*x, s*w*y, s*w*z
    xx, xy, xz = s*x*x, s*x*y, s*x*z
    yy, yz, zz = s*y*y, s*y*z, s*z*z
    mat = np.stack([
        1.0-(yy+zz), xy-wz, xz+wy,
        xy+wz, 1.0-(xx+zz), yz-wx,
        xz-wy, yz+wx, 1.0-(xx+yy)
    ], axis=-1)
    return mat.reshape(q.shape[:-1] + (3, 3))
def quat_from_axis_angle(axis, angle):
    axis = np.asarray(axis, dtype=np.float64)
    axis = axis / np.linalg.norm(axis)
    half = angle * 0.5
    w = np.cos(half)
    xyz = axis * np.sin(half)
    return np.array([w, xyz[0], xyz[1], xyz[2]], dtype=np.float64)

def compute_pad(q_robot: np.ndarray, q_anchor: np.ndarray) -> np.ndarray:
    q_rel = relative_quat(q_robot, q_anchor)
    R = matrix_from_quat(q_rel)
    return R[..., :2].reshape(-1, 6).squeeze(0)

def compute_ori_b(root_quat_w: np.ndarray, anchor_quat_w: np.ndarray, 
                  q_robot: np.ndarray) -> np.ndarray:
    # yaw   = float(q_robot[12])
    # roll  = float(q_robot[13])
    # pitch = float(q_robot[14])
    yaw   = 0.0
    roll  = 0.0
    pitch = 0.0

    q_yaw   = quat_from_axis_angle([0, 0, 1], yaw)
    q_roll  = quat_from_axis_angle([1, 0, 0], roll)
    q_pitch = quat_from_axis_angle([0, 1, 0], pitch)

    quat_rel   = quat_mul(quat_mul(q_yaw, q_roll), q_pitch)
    robot_quat_w = quat_mul(root_quat_w, quat_rel)
    
    return compute_pad(robot_quat_w, anchor_quat_w).astype(np.float32)


def add_noise_to_array(data_array, noise_std=0.0, noise_type='gaussian'):
    """
    为任意numpy数组添加噪声
    
    Args:
        data_array: 输入的numpy数组
        noise_std: 噪声标准差 (默认: 0.0，即无噪声)
        noise_type: 噪声类型 (默认: 'gaussian'，高斯噪声)
    
    Returns:
        添加噪声后的numpy数组，与输入数组形状相同
    """
    # 确保输入是numpy数组
    data_array = np.asarray(data_array, dtype=np.float32)
    
    if noise_std <= 0.0:
        # 如果噪声标准差为0或负数，直接返回原始数据
        return data_array.copy()
    
    # 根据噪声类型添加噪声
    if noise_type == 'gaussian':
        # 添加高斯白噪声
        noise = np.random.normal(0, noise_std, data_array.shape)
        noisy_data = data_array + noise
    elif noise_type == 'uniform':
        # 添加均匀分布噪声
        noise = np.random.uniform(-noise_std, noise_std, data_array.shape)
        noisy_data = data_array + noise
    else:
        raise ValueError(f"不支持的噪声类型: {noise_type}")
    
    return noisy_data.astype(np.float32)

if __name__ == "__main__":
    # get config file name from command line
    # config_file = args.config_file
    with open(f"{config_file}", "r") as flie:
        config = yaml.load(flie, Loader=yaml.FullLoader)
        policy_path = config["policy_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        xml_path = config["xml_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        urdf_path = config["urdf_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        motion_path = config["motion_path"].replace("{LEGGED_GYM_ROOT_DIR}", LEGGED_GYM_ROOT_DIR)
        ref_motion = np.load(motion_path)
        print("当前参考动作的key: ",ref_motion.files)

        simulation_duration = config["simulation_duration"]
        simulation_dt = config["simulation_dt"]
        control_decimation = config["control_decimation"]

        kps = np.array(config["kps"], dtype=np.float32)
        kds = np.array(config["kds"], dtype=np.float32)

        default_pos = np.array(config["default_pos"], dtype=np.float32)
        default_pos_in_policy = convert_joint_order(default_pos)  
        action_scale = np.array(config["action_scale"])
        action_scale = convert_joint_order(action_scale)
        num_actions = config["num_actions"]
        num_obs = config["num_obs"]

        motor_noise_switch = config["MOTOR_NOISE"]
        imu_noise_switch = config["IMU_NOISE"]


    # define context variables
    action = np.zeros(num_actions, dtype=np.float32)
    target_dof_pos = default_pos.copy()
    obs = np.zeros(num_obs, dtype=np.float32)

    counter = 0

    # Load robot model
    model = mujoco.MjModel.from_xml_path(xml_path)
    model.opt.timestep = simulation_dt
    data = mujoco.MjData(model)
    data.qpos[7:] = default_pos

    torso_ref_index = 9
    torso_link_name = 16


    # Load ONNX model using ONNX Runtime
    ort_session = ort.InferenceSession(policy_path)  # Load ONNX model

    with mujoco.viewer.launch_passive(model, data) as viewer:
        # Close the viewer automatically after simulation_duration wall-seconds.
        start = time.time()
        while viewer.is_running() and time.time() - start < simulation_duration:
            step_start = time.time()
            tau = pd_control(target_dof_pos, data.qpos[7:], kps, data.qvel[6:], kds)
            data.ctrl[:] = tau
            # mj_step can be replaced with code that also evaluates
            # a policy and applies a control signal before stepping the physics.
            mujoco.mj_step(model, data)

            counter += 1
            if counter % control_decimation == 0:
                #create observation
                frame_idx = counter // control_decimation
                # 修改为循环模式而不是重置为0
                if frame_idx >= ref_motion["joint_pos"].shape[0]:
                    frame_idx = frame_idx % ref_motion["joint_pos"].shape[0]

                if imu_noise_switch:
                    pelvis_quat = add_noise_to_array(data.qpos[3:7], noise_std=0.02, noise_type='gaussian')
                    pelvis_ang_vel = add_noise_to_array(data.qvel[3:6], noise_std=0.1, noise_type='gaussian')
                else:
                    pelvis_quat = data.qpos[3:7]
                    pelvis_ang_vel = data.qvel[3:6]

                if motor_noise_switch:
                    motor_cur_pos = add_noise_to_array(data.qpos[7:], noise_std=0.01, noise_type='gaussian')
                    motor_cur_vel = add_noise_to_array(data.qvel[6:], noise_std=0.1, noise_type='gaussian')
                else:
                    motor_cur_pos = data.qpos[7:]
                    motor_cur_vel = data.qvel[6:]

                pelvis_quat = data.qpos[3:7]
                anchor_quat_w = ref_motion["body_quat_w"][frame_idx, 9]
                motion_anchor_ori_b = compute_ori_b(pelvis_quat, anchor_quat_w, data.qpos[7:])

                obs[0:12]  = ref_motion["joint_pos"][frame_idx]
                obs[12:24] = ref_motion["joint_vel"][frame_idx]
                obs[24:30] = motion_anchor_ori_b
                # obs[64:67] = data.qvel[3:6]
                obs[30:33] = pelvis_ang_vel
                # obs[67:96] = convert_joint_order(data.qpos[7:])-default_pos_in_policy
                # obs[96:125] = convert_joint_order(data.qvel[6:])
                obs[33:45] = convert_joint_order(motor_cur_pos)-default_pos_in_policy
                obs[45:57] = convert_joint_order(motor_cur_vel)
                obs[57:69] = action
   
                # # policy inference
                t = np.array([[0]], dtype=np.float32)
                # obs = build_obs(ref_motion, frame_idx, data, action, default_pos_in_policy, motion_anchor_ori_b)
                obs_out = np.asarray(obs, dtype=np.float32).reshape(1, -1)  # (1,154)
                t = np.array([[frame_idx]], dtype=np.float32)                   # 固定 time_step=0
                ort_outs = ort_session.run(["actions"], {"obs": obs_out, "time_step": t})[0]
                action = ort_outs[0].squeeze()  # 从 ONNX 输出中获取动作
                # transform action to target_dof_pos
                target_dof_from_policy = action * action_scale + default_pos_in_policy
                target_dof_pos = convert_joint_order(target_dof_from_policy,False)

            
            # Pick up changes to the physics state, apply perturbations, update options from GUI.
            viewer.sync()

            # Rudimentary time keeping, will drift relative to wall clock.
            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
