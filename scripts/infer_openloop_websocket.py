import asyncio
import websockets
import msgpack
import msgpack_numpy as m
import numpy as np
import cv2
import os
import json
import base64
import yaml
# from x2robot_dataset.common.data_preprocessing import _ARX_MAPPING
# from x2robot_dataset.common.data_utils import convert_euler_to_6D
from wall_x.data.utils import euler_to_rotation_6d
from scipy.spatial.transform import Rotation
from matplotlib import pyplot as plt

_ARX_MAPPING = {
    "follow_left_ee_cartesian_pos": "follow_left_position",
    "follow_left_ee_rotation": "follow_left_rotation",
    "follow_left_ee_rotation_6D": "follow_left_rotation",
    "follow_left_gripper": "follow_left_gripper",
    "follow_right_ee_cartesian_pos": "follow_right_position",
    "follow_right_ee_rotation": "follow_right_rotation",
    "follow_right_ee_rotation_6D": "follow_right_rotation",
    "follow_right_gripper": "follow_right_gripper",
    "master_left_ee_cartesian_pos": "master_left_position",
    "master_left_ee_rotation_6D": "master_left_rotation",
    "master_left_gripper": "master_left_gripper",
    "master_right_ee_cartesian_pos": "master_right_position",
    "master_right_ee_rotation_6D": "master_right_rotation",
    "master_right_gripper": "master_right_gripper",
    "velocity_decomposed": "velocity_decomposed",
    "height": "lifting_mechanism_position",
    "head_actions": "head_rotation",
}

# 注册 msgpack-numpy 扩展
m.patch()

def plot_openloop(action_pred_list, action_gt_list, save_path):
    assert len(action_pred_list) == len(
        action_gt_list
    ), "Predicted action and ground truth action must have the same shape."

    dim = len(action_pred_list[0])
    plt.figure(figsize=(12, 4 * dim))

    for i in range(dim):
        plt.subplot(dim, 1, i + 1)

        # plot every 10th action
        plt.xticks(np.arange(0, sum(len(gt) for gt in action_gt_list), step=10))
        # for j in range(len(action_gt_list)):
        gt_action = np.array(action_gt_list)
        predict_action = np.array(action_pred_list)

        plt.plot(gt_action[:, i], label="Ground Truth", color="blue")
        plt.plot(predict_action[:, i], label="Model Output", color="orange")

        plt.title(f"Action Dimension {i + 1}")
        plt.xlabel("Time Step")
        plt.ylabel("Action Value")
        plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path}.jpg", dpi=200)
    print(f"Saved plot to {save_path}.jpg")


# --- 数据加载部分 ---
def load_vla_case_data(case_path, sample_name, config):
    """读取单个 case 下的轨迹和视频帧"""
    data = {
        "trajectory": None,
        "face_frames": [],
        "left_frames": [],
        "right_frames": [],
    }
    follow2_pos_keys = [
        "follow_right_ee_cartesian_pos",
        "follow_right_ee_rotation_6D",
        "follow_right_ee_rotation",
        "follow_right_gripper",
    ]
    follow1_pos_keys = [
        "follow_left_ee_cartesian_pos",
        "follow_left_ee_rotation_6D",
        "follow_left_ee_rotation",
        "follow_left_gripper",
    ]
    other_pos_keys = [
        "velocity_decomposed",
        "height",
        "head_actions",
    ]
    # 加载轨迹数据
    follow1_pos_list = []
    follow2_pos_list = []
    other_pos_list = []           
    follow1_pos_list_gt = []
    follow2_pos_list_gt = []
    other_pos_list_gt = [] 
    traj_path = os.path.join(case_path, f"{sample_name}.json")
    if os.path.exists(traj_path):
        with open(traj_path, "r") as f:
            data["trajectory"] = json.load(f)["data"]
            for traj in data["trajectory"]:
                follow1_pos = []
                follow2_pos = []
                other_pos = []
                for obs_k in config["data"]["obs_action_keys"]:
                    if obs_k in follow1_pos_keys:
                        act = traj[_ARX_MAPPING[obs_k]]
                        if "rotation_6D" in obs_k and np.asarray(act).shape[-1] == 3:
                            act = euler_to_rotation_6d(np.array(act, dtype=np.float64).reshape(-1, 3)).flatten().tolist()
                        if isinstance(act, list):
                            follow1_pos.extend(act)
                        else:
                            follow1_pos.append(act)
                    elif obs_k in follow2_pos_keys:
                        act = traj[_ARX_MAPPING[obs_k]]
                        if "rotation_6D" in obs_k and np.asarray(act).shape[-1] == 3:
                            act = euler_to_rotation_6d(np.array(act, dtype=np.float64).reshape(-1, 3)).flatten().tolist()
                        if isinstance(act, list):
                            follow2_pos.extend(act)
                        else:
                            follow2_pos.append(act)
                    elif obs_k in other_pos_keys:   # 新增
                        default_dims = {"velocity_decomposed": 3, "height": 1, "head_actions": 2}
                        act = [0.0] * default_dims.get(obs_k, 1)
                        other_pos.extend(act)
                        
                follow1_pos_list.append(follow1_pos)
                follow2_pos_list.append(follow2_pos)
                other_pos_list.append(other_pos)

                follow1_pos = []
                follow2_pos = []
                other_pos = []
                for obs_k in config["data"]["obs_action_keys"]:
                    if obs_k in follow1_pos_keys:
                        act = traj[_ARX_MAPPING[obs_k]]
                        if "rotation_6D" in obs_k and np.asarray(act).shape[-1] == 3:
                            act = euler_to_rotation_6d(np.array(act, dtype=np.float64).reshape(-1, 3)).flatten().tolist()
                        if isinstance(act, list):
                            follow1_pos.extend(act)
                        else:
                            follow1_pos.append(act)
                    elif obs_k in follow2_pos_keys:
                        act = traj[_ARX_MAPPING[obs_k]]
                        if "rotation_6D" in obs_k and np.asarray(act).shape[-1] == 3:
                            act = euler_to_rotation_6d(np.array(act, dtype=np.float64).reshape(-1, 3)).flatten().tolist()
                        if isinstance(act, list):
                            follow2_pos.extend(act)
                        else:
                            follow2_pos.append(act)
                    elif obs_k in other_pos_keys:   # 新增
                        default_dims = {"velocity_decomposed": 3, "height": 1, "head_actions": 2}
                        act = [0.0] * default_dims.get(obs_k, 1)
                        other_pos.extend(act)

                follow1_pos_list_gt.append(follow1_pos)
                follow2_pos_list_gt.append(follow2_pos)
                other_pos_list_gt.append(other_pos)

    data["follow1_pos_list"] = follow1_pos_list
    data["follow2_pos_list"] = follow2_pos_list
    data["other_pos_list"] = other_pos_list
    data["follow1_pos_gt_list"] = follow1_pos_list_gt
    data["follow2_pos_gt_list"] = follow2_pos_list_gt
    data["other_pos_gt_list"] = other_pos_list_gt

    if data["trajectory"] is not None:
        data["action_gt_relative"] = compute_relative_action_gt(data["trajectory"], config)

    # 视频映射
    video_map = {
        "faceImg.mp4": "face_frames",
        "leftImg.mp4": "left_frames",
        "rightImg.mp4": "right_frames",
    }

    for video_name, key in video_map.items():
        video_path = os.path.join(case_path, video_name)
        if not os.path.exists(video_path):
            continue

        cap = cv2.VideoCapture(video_path)
        frames = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            # VLA 通常需要 RGB
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        data[key] = np.array(frames)

    return data


def compute_relative_action_gt(trajectory, config):
    """Compute relative action GT from trajectory based on predict_action_keys."""
    gt_list = []
    for t in range(len(trajectory) - 1):
        traj_t = trajectory[t]
        traj_t1 = trajectory[t + 1]
        gt_step = []

        for pred_k in config["data"]["predict_action_keys"]:
            raw_key = _ARX_MAPPING.get(pred_k, pred_k)
            val_t = traj_t.get(raw_key)
            val_t1 = traj_t1.get(raw_key)

            if val_t is None or val_t1 is None:
                default_dims = {
                    "velocity_decomposed": 3, "height": 1, "head_actions": 2,
                }
                dim = default_dims.get(pred_k, 1)
                gt_step.extend([0.0] * dim)
                continue

            val_t = np.atleast_1d(np.array(val_t, dtype=np.float64))
            val_t1 = np.atleast_1d(np.array(val_t1, dtype=np.float64))

            if "rotation_6D_relative" in pred_k:
                R_t = Rotation.from_euler("xyz", val_t.flatten())
                R_t1 = Rotation.from_euler("xyz", val_t1.flatten())
                R_rel = R_t1 * R_t.inv()
                mat = R_rel.as_matrix()
                rot_6d = np.concatenate([mat[:, 0], mat[:, 1]])
                gt_step.extend(rot_6d.tolist())
            elif "relative" in pred_k:
                diff = (val_t1 - val_t).flatten()
                gt_step.extend(diff.tolist())
            else:
                val_out = val_t1.flatten()
                if "rotation_6D" in pred_k:
                    val_out = euler_to_rotation_6d(val_out.reshape(-1, 3)).flatten()
                gt_step.extend(val_out.tolist())

        gt_list.append(gt_step)
    return gt_list


# --- 通信与评估部分 ---


def encode_image(image):
    """压缩并编码图像"""
    # 如果 server 接受 msgpack 里的 raw numpy，可以直接传 image
    # 但由于你第一份代码用了 encode_image，这里保留 JPEG 压缩以节省带宽
    _, buffer = cv2.imencode(".jpg", image)
    return base64.b64encode(buffer).decode("utf-8")


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    config["data"]["model_type"] = config.get("model_type")
    return config


async def run_vla_evaluation(uri, data_dir, config_path, save_dir, sample_nums=1):

    config = load_config(config_path)

    # 读取任务列表
    report_path = os.path.join(data_dir, "report.json")
    with open(report_path, "r") as f:
        samples = json.load(f)["sample_name"]

    async with websockets.connect(uri) as websocket:
        print(f"🚀 Connected to VLA Server")

        for sample in samples[:sample_nums]:
            sample_path = os.path.join(data_dir, sample)
            save_path = os.path.join(save_dir, sample)
            print(f"\n📂 Processing: {sample}")

            case_data = load_vla_case_data(sample_path, sample, config)

            action_pred = []
            action_gt = []

            # 确保帧数和轨迹步数匹配（以最小值为准防止溢出）
            num_steps = min(
                len(case_data["follow1_pos_list"]), len(case_data["face_frames"]), len(case_data["other_pos_list"])
            )
            idx = int(0.1 * num_steps)
            while len(action_pred) < num_steps - 10:
                # idx = len(action_pred)
                # if idx!=0:
                #     idx-=1
                payload = {
                    # "state": {
                    #     # "follow1_pos": case_data["follow1_pos_list"][
                    #     #     idx
                    #     # ],
                    #     # "follow2_pos": case_data["follow2_pos_list"][idx],
                    # },

                    "state": case_data["follow1_pos_list"][idx] + case_data["follow2_pos_list"][idx] + case_data["other_pos_list"][idx],
                    # "views": {
                    #     "camera_front": encode_image(case_data["face_frames"][idx]),
                    #     "camera_left": encode_image(case_data["left_frames"][idx]),
                    #     "camera_right": encode_image(case_data["right_frames"][idx]),
                    # },
                    "camera_front": case_data["face_frames"][idx],
                    "camera_left": case_data["left_frames"][idx],
                    "camera_right": case_data["right_frames"][idx],
                    "prompt": "Put the cup on the plate.",  # 或者从 json 中读取 task
                    "dataset_names": "x2_normal"
                }
                # 2. 发送请求
                result = None
                while True:
                    binary_data = msgpack.packb(payload, use_bin_type=True)
                    await websocket.send(binary_data)

                    # 3. 接收预测结果
                    response = await websocket.recv()
                    if isinstance(response, str):
                        raise RuntimeError(f"Server error: {response}")
                    result = msgpack.unpackb(response, raw=False)

                    # if "follow1_pos" in result:
                    if "predict_action" in result:
                        break

                predict_action = np.array(result["predict_action"])  # (1, pred_horizon, action_dim)
                action_dim = predict_action.shape[-1]
                action_step = predict_action[0, 0, :action_dim].tolist()

                if idx > 0:
                    idx += 1

                gt_step = case_data["action_gt_relative"][idx]
                action_gt.append(gt_step)
                action_pred.append(action_step)

                print("len(action_pred)", len(action_pred), len(action_pred[-1]))
                print("len(action_gt)", len(action_gt), len(action_gt[-1]))

                # idx+=1

                if len(action_pred) > 150:
                    break

            ml = min(len(action_pred), len(action_gt))
            plot_openloop(action_pred[:ml], action_gt[:ml], save_path)


if __name__ == "__main__":
    try:
        uri = "ws://localhost:42100"
        data_dir = "/x2robot_data/zhengwei/10053/20250526-day-pick-up_cup-train"
        # config_path = "/x2robot_v2/share/bus2602/pretrain_vq_delta_6d_eef_xloss_448_0214/ckpt_ddp_1/0_200000/config.yml"
        config_path = "/x2robot_v2/share/jerry1/ckpt/open-wallx/pick_up_cup_lerobot_bs256_ep10_448/3/config.yml"
        save_dir = (
            "/mnt/data/x2robot_v2/jerry1/open-wallx/logs/openloop/pick_up_cup_bus2602_bs256_ep3"
        )

        # data_dir = "/x2robot_data/zhengwei/10162/20260202-night-folding_clothes/"
        # config_path = "/x2robot_v2/share/yangping/tmp/0130/folding_cloths/11_530000/config.yml"
        # save_dir = "/mnt/data/x2robot_v2/yangping/gitlab/wall-x/run_scripts/visualize/folding_clothes"

        # data_dir = "/x2robot_data/zhengwei/10053/20250526-day-pick-up_cup-train/"
        # config_path = "/x2robot_v2/share/liangyuxin/share_ckpts/multitask_0130/9/config.yml"
        # save_dir = "/mnt/data/x2robot_v2/yangping/gitlab/wall-x/run_scripts/visualize/pick-up_cup-train"

        sample_nums = 1
        asyncio.run(
            run_vla_evaluation(uri, data_dir, config_path, save_dir, sample_nums)
        )
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
