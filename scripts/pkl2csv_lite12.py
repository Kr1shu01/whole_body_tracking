import os
import math
import pickle
import joblib
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R


# # 获取当前脚本所在目录
# current_dir = os.path.dirname(os.path.abspath(__file__))
# print("当前文件路径：", current_dir)
# # 构建相对于当前脚本位置的文件路径
# file_path = os.path.join(current_dir, "../dataest/kun_dance/TeShanKao.pkl")     # 输入 pkl 文件路径
# csv_path = os.path.join(current_dir, "../dataest/kun_dance/TeShanKao.csv")         # 输出 csv 文件路径
# pkl_output_path = os.path.join(current_dir, "../dataest/kun_dance/TeShanKao_process.pkl")  # 输出处理后的 pkl 文件路径


file_path = "/home/kr1shu/motion_data/motion_bvh/dunqi_f0_lite12.pkl"     # 输入 pkl 文件路径
csv_path = "/home/kr1shu/motion_data/motion_bvh/dunqi_f0_lite12.csv"      # 输出 csv 文件路径
# pkl_output_path = "/home/kr1shu/motion_data/motion_lafan/lafan1_lite12/jump11_processed.pkl"  # 输出处理后的 pkl 文件路径


default_joint_pos_old = np.array([ 
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,      # 左腿    
    -0.1, 0.0, 0.0, 0.2, -0.1, 0.0,      # 右腿
    0.0, 0.2, 0.0, 1.0, 0.0, 0.0, 0.0,     # 左臂
    0.0, -0.2, 0.0, 1.0, 0.0, 0.0, 0.0 
], dtype=np.float32)


def align_Ground(quat):
    """
    输入与输出均为 [w, x, y, z]。
    """
    w, x, y, z = quat
    # 归一化
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n == 0.0:
        raise ValueError("Zero-norm quaternion")
    w, x, y, z = w/n, x/n, y/n, z/n

    # 提取 yaw
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y*y + z*z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    # 构造只含 yaw 的四元数
    cy = math.cos(0.5 * yaw)
    sy = math.sin(0.5 * yaw)
    return (cy, 0.0, 0.0, sy)

def correct_rot(root_rot,root_trans,dof):
    origin_root_rot = root_rot[0]
    q_correction = R.from_quat(origin_root_rot).inv()
    correct_root_rot = (q_correction * R.from_quat(root_rot)).as_quat()
    correct_root_trans = np.dot(root_trans, q_correction.as_matrix().T)

    # correct_root_trans = root_trans.copy()  # 直接使用原始位置
    # correct_root_rot = root_rot.copy()  # 直接使用原始旋转
    
    euler_initial = R.from_quat(origin_root_rot).as_euler('xyz',degrees=False)
    theta = euler_initial[1]

    print('correct theta: ',theta)
    
    dof[:,[0,6,14]] += theta

    
    # 只对初始的 x 和 y 进行去偏置处理
    correct_root_trans[:, 0] -= correct_root_trans[0, 0]  # 去除初始 x 偏置
    correct_root_trans[:, 1] -= correct_root_trans[0, 1]  # 去除初始 y 偏置
    correct_root_trans[:, 2] -= correct_root_trans[0, 2] - 0.97  # 去除初始 z 偏置

    return correct_root_trans,correct_root_rot,dof

def main():
    with open(file_path, "rb") as f:
        data = joblib.load(f)

    # 确保包含所需的 key
    required_keys = ["root_pos", "root_rot", "dof_pos"]
    for k in required_keys:
        if k not in data:
            raise KeyError(f"缺少关键字段: {k}")

    # 依次取出数据并拼接
    # 假设每个 key 对应的是 (帧数, 特征数) 的数组
    fps = data["fps"]
    local_body_pos = None
    link_body_list = None
    ori_root_pos = np.array(data["root_pos"])[2:]
    ori_root_rot = np.array(data["root_rot"])[2:]
    ori_dof_pos  = np.array(data["dof_pos"])[2:]

    root_pos,root_rot,dof_pos = correct_rot(ori_root_rot,ori_root_pos,ori_dof_pos)
    # root_pos,root_rot,dof_pos = ori_root_rot,ori_root_pos,ori_dof_pos)

    pre_root_pos = np.array([0, 0, 0.97])
    pre_root_rot = np.array([0, 0, 0, 1])

    final_root_pos = root_pos[-1].copy()
    print("Final root pos:", final_root_pos)
    final_root_pos[2] = 0.97
    final_root_rot = align_Ground(root_rot[-1])

    pre_root_pos_frames = np.tile(pre_root_pos, (60, 1))
    pre_root_rot_frames = np.tile(pre_root_rot, (60, 1))
    pre_dof_pos_frames = np.tile(default_joint_pos_old, (60, 1))
    
    post_root_pos_frames = np.tile(final_root_pos, (60, 1))
    post_root_rot_frames = np.tile(final_root_rot, (60, 1))
    post_dof_pos_frames = np.tile(default_joint_pos_old, (60, 1))
    
    # 拼接所有帧数据：前30帧默认姿态 + 原始动作数据 + 后60帧结束姿态
    combined_root_pos = np.vstack([pre_root_pos_frames, root_pos, post_root_pos_frames])
    combined_root_rot = np.vstack([pre_root_rot_frames, root_rot, post_root_rot_frames])
    combined_dof_pos = np.vstack([pre_dof_pos_frames, dof_pos, post_dof_pos_frames])

    # 检查帧数是否一致
    n_frames = root_pos.shape[0]
    assert root_rot.shape[0] == n_frames and dof_pos.shape[0] == n_frames, "帧数不一致"

    # 拼接 (沿第二维度)
    all_data = np.hstack([combined_root_pos, combined_root_rot, combined_dof_pos])

    # 转换成 DataFrame 并保存
    df = pd.DataFrame(all_data)
    df.to_csv(csv_path, index=False, header=False)

        # 重新打包成pkl格式
    processed_data = {
        "fps": fps,
        "root_pos": combined_root_pos,
        "root_rot": combined_root_rot,
        "dof_pos": combined_dof_pos,
        "local_body_pos": local_body_pos,
        "link_body_list": link_body_list
    }
    
    # 保存处理后的pkl文件
    # with open(pkl_output_path, "wb") as f:
    #     pickle.dump(processed_data, f)
    # print(f"成功保存到 {csv_path}, 共 {n_frames} 帧，{all_data.shape[1]} 个特征")
    print("处理前 z 值范围:", ori_root_pos[:, 2].min(), "~", ori_root_pos[:, 2].max())
    print("处理后 z 值范围:", root_pos[:, 2].min(), "~", root_pos[:, 2].max())

if __name__ == "__main__":
    main()