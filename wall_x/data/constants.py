# 动作键范围配置 - 所有可能的action key及其范围
ACTION_KEY_RANGES = {
    # ARX系列 - 末端位置控制
    "follow_left_ee_cartesian_pos": {
        "min_range": [-0.1, -0.5, -0.5],
        "max_range": [0.5, 0.5, 0.5],
        "shape": 3,
    },
    "follow_right_ee_cartesian_pos": {
        "min_range": [-0.1, -0.5, -0.5],
        "max_range": [0.5, 0.5, 0.5],
        "shape": 3,
    },
    "master_left_ee_cartesian_pos": {
        "min_range": [-0.1, -0.5, -0.5],
        "max_range": [0.5, 0.5, 0.5],
        "shape": 3,
    },
    "master_right_ee_cartesian_pos": {
        "min_range": [-0.1, -0.5, -0.5],
        "max_range": [0.5, 0.5, 0.5],
        "shape": 3,
    },
    # ARX系列 - 末端旋转控制
    "follow_left_ee_rotation": {
        "min_range": [-3.0, -3.0, -3.0],
        "max_range": [3.0, 3.0, 3.0],
        "shape": 3,
    },
    "follow_right_ee_rotation": {
        "min_range": [-3.0, -3.0, -3.0],
        "max_range": [3.0, 3.0, 3.0],
        "shape": 3,
    },
    "master_left_ee_rotation": {
        "min_range": [-3.0, -3.0, -3.0],
        "max_range": [3.0, 3.0, 3.0],
        "shape": 3,
    },
    "master_right_ee_rotation": {
        "min_range": [-3.0, -3.0, -3.0],
        "max_range": [3.0, 3.0, 3.0],
        "shape": 3,
    },
    # ARX系列 - 夹爪控制
    "follow_left_gripper": {"min_range": [-1], "max_range": [4.5], "shape": 1},
    "follow_right_gripper": {"min_range": [-1], "max_range": [4.5], "shape": 1},
    "master_left_gripper": {"min_range": [-1], "max_range": [4.5], "shape": 1},
    "master_right_gripper": {"min_range": [-1], "max_range": [4.5], "shape": 1},
    # ARX系列 - 相对控制
    "follow_left_ee_cartesian_pos_relative": {
        "min_range": [-0.6, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0],
        "shape": 3,
    },
    "follow_right_ee_cartesian_pos_relative": {
        "min_range": [-0.6, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0],
        "shape": 3,
    },
    "master_left_ee_cartesian_pos_relative": {
        "min_range": [-0.6, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0],
        "shape": 3,
    },
    "master_right_ee_cartesian_pos_relative": {
        "min_range": [-0.6, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0],
        "shape": 3,
    },
    "follow_left_ee_rotation_relative": {
        "min_range": [-3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14],
        "shape": 3,
    },
    "follow_right_ee_rotation_relative": {
        "min_range": [-3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14],
        "shape": 3,
    },
    "master_left_ee_rotation_relative": {
        "min_range": [-3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14],
        "shape": 3,
    },
    "master_right_ee_rotation_relative": {
        "min_range": [-3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14],
        "shape": 3,
    },
    # ARX系列 - 6D旋转表示
    "follow_left_ee_rotation_6D": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    "follow_right_ee_rotation_6D": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    "master_left_ee_rotation_6D": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    "master_right_ee_rotation_6D": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    # ARX系列 - 相对6D旋转表示
    "follow_left_ee_rotation_6D_relative": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    "follow_right_ee_rotation_6D_relative": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    "master_left_ee_rotation_6D_relative": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    "master_right_ee_rotation_6D_relative": {
        "min_range": [-1.0, -1.0, -1.0, -1.0, -1.0, -1.0],
        "max_range": [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        "shape": 6,
    },
    # ARX系列 - 关节控制
    "follow_left_arm_joint_pos": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    "follow_right_arm_joint_pos": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    "master_left_arm_joint_pos": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    "master_right_arm_joint_pos": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    "follow_left_arm_joint_dev": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    "follow_right_arm_joint_dev": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    "master_left_arm_joint_dev": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    "master_right_arm_joint_dev": {
        "min_range": [-3.14, -3.14, -3.14, -3.14, -3.14, -3.14],
        "max_range": [3.14, 3.14, 3.14, 3.14, 3.14, 3.14],
        "shape": 6,
    },
    # 乌龟+移动支架系列
    "velocity_decomposed": {
        "min_range": [-0.3, -0.3, -0.4],
        "max_range": [0.3, 0.3, 0.4],
        "shape": 3,
    },  # vx, vy, vaw (前进、侧向、角速度)
    "car_pose": {
        "min_range": [-10.0, -10.0, -3.14159],
        "max_range": [10.0, 10.0, 3.14159],
        "shape": 3,
    },  # x, y, theta (用于历史数据计算)
    "height": {"min_range": [0.0], "max_range": [1.0], "shape": 1},  # 机器人高度
    "head_rotation": {
        "min_range": [-1.5, -1.5],
        "max_range": [1.5, 1.5],
        "shape": 2,
    },  # head_pitch, head_yaw
    # ARX系列 - 关节力矩控制 #TODO：需要复查
    "follow_left_arm_joint_cur": {
        "min_range": [-20, -20, -20, -20, -20, -20, -9],
        "max_range": [20, 20, 20, 20, 20, 20, 9],
    },
    "follow_right_arm_joint_cur": {
        "min_range": [-20, -20, -20, -20, -20, -20, -9],
        "max_range": [20, 20, 20, 20, 20, 20, 9],
    },
    "master_left_arm_joint_cur": {
        "min_range": [-20, -20, -20, -20, -20, -20, -9],
        "max_range": [20, 20, 20, 20, 20, 20, 9],
    },
    "master_right_arm_joint_cur": {
        "min_range": [-20, -20, -20, -20, -20, -20, -9],
        "max_range": [20, 20, 20, 20, 20, 20, 9],
    },
    # ARX系列 - 夹爪力矩控制
    "follow_left_gripper_cur": {"min_range": [-4], "max_range": [4]},
    "follow_right_gripper_cur": {"min_range": [-4], "max_range": [4]},
    # JAKA系列 - 末端位置控制
    "follow_left_ee_cartesian_pos_jaka": {
        "min_range": [0.3, -0.5, 0.1],
        "max_range": [0.7, 0.5, 0.5],
    },
    "follow_right_ee_cartesian_pos_jaka": {
        "min_range": [0.3, -0.5, 0.1],
        "max_range": [0.7, 0.5, 0.5],
    },
    "follow_left_ee_rotation_jaka": {
        "min_range": [-1.7, -1.7, -1.7],
        "max_range": [1.7, 1.7, 1.7],
    },
    "follow_right_ee_rotation_jaka": {
        "min_range": [-1.7, -1.7, -1.7],
        "max_range": [1.7, 1.7, 1.7],
    },
    # JAKA系列 - 关节控制
    "follow_left_arm_joint_pos_jaka": {
        "min_range": [-6.29, -2.19, -2.27, -6.29, -2.10, -6.29],
        "max_range": [6.29, 2.19, 2.27, 6.29, 2.10, 6.29],
    },
    "follow_right_arm_joint_pos_jaka": {
        "min_range": [-6.29, -2.19, -2.27, -6.29, -2.10, -6.29],
        "max_range": [6.29, 2.19, 2.27, 6.29, 2.10, 6.29],
    },
    # 手部关节控制
    "follow_left_hand_joint_pos": {
        "min_range": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "max_range": [
            2.1,
            1.85,
            1.6,
            1.6,
            0.6,
            1.6,
            1.6,
            1.6,
            1.6,
            1.6,
            1.6,
            0.6,
            1.6,
            1.6,
            1.6,
            0.6,
            1.6,
            1.6,
            1.6,
        ],
    },
    "follow_right_hand_joint_pos": {
        "min_range": [
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        ],
        "max_range": [
            2.1,
            1.85,
            1.6,
            1.6,
            0.6,
            1.6,
            1.6,
            1.6,
            1.6,
            1.6,
            1.6,
            0.6,
            1.6,
            1.6,
            1.6,
            0.6,
            1.6,
            1.6,
            1.6,
        ],
    },
}


# def get_action_ranges_by_keys(action_keys):
#     """
#     根据action_keys列表获取对应的最小最大范围

#     Args:
#         action_keys: 动作键列表，按照预期的顺序

#     Returns:
#         tuple: (min_ranges, max_ranges) 按照action_keys顺序排列的范围列表
#     """
#     min_ranges = []
#     max_ranges = []

#     for key in action_keys:
#         if key not in ACTION_KEY_RANGES:
#             raise ValueError(f"Action key '{key}' not found in ACTION_KEY_RANGES")

#         config = ACTION_KEY_RANGES[key]
#         min_ranges.extend(config["min_range"])
#         max_ranges.extend(config["max_range"])

#     return min_ranges, max_ranges


# def get_combined_action_keys(predict_action_keys, obs_action_keys):
#     """
#     合成所有需要的action keys，去重并保持顺序

#     Args:
#         predict_action_keys: 预测动作键列表
#         obs_action_keys: 观测动作键列表

#     Returns:
#         list: 合成的action keys列表，去重并保持顺序
#     """
#     # 使用dict来保持顺序去重（Python 3.7+）
#     combined_keys = dict.fromkeys(predict_action_keys + obs_action_keys)
#     return list(combined_keys.keys())
