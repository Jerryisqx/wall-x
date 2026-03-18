import asyncio
import websockets
import msgpack
import msgpack_numpy as m
import numpy as np
import cv2
import os
import json
import base64
from matplotlib import pyplot as plt

m.patch()

_ACTION_LABELS = [
    "left_pos_x", "left_pos_y", "left_pos_z",
    "left_rot_r", "left_rot_p", "left_rot_y",
    "left_gripper",
    "right_pos_x", "right_pos_y", "right_pos_z",
    "right_rot_r", "right_rot_p", "right_rot_y",
    "right_gripper",
]


def plot_openloop(action_pred_list, action_gt_list, save_path, dim_labels=None):
    assert len(action_pred_list) == len(action_gt_list)

    predict_action = np.array(action_pred_list)
    gt_action = np.array(action_gt_list)
    dim = min(predict_action.shape[1], gt_action.shape[1])

    plt.figure(figsize=(12, 4 * dim))
    for i in range(dim):
        plt.subplot(dim, 1, i + 1)
        plt.xticks(np.arange(0, len(predict_action), step=10))
        plt.plot(predict_action[:, i], label="Model Output", color="orange")
        plt.plot(gt_action[:, i], label="Ground Truth", color="blue")

        label = dim_labels[i] if dim_labels and i < len(dim_labels) else f"dim {i}"
        plt.title(label)
        plt.xlabel("Time Step")
        plt.ylabel("Action Value")
        plt.legend()

    plt.tight_layout(rect=[0, 0, 1, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(f"{save_path}.jpg", dpi=200)
    plt.close()
    print(f"Saved plot to {save_path}.jpg")


def _build_follow_pos(traj, side):
    """Build 7D follow pos [x,y,z,r,p,y,gripper] from trajectory step."""
    pos = traj.get(f"follow_{side}_position", [0, 0, 0])
    rot = traj.get(f"follow_{side}_rotation", [0, 0, 0])
    grip = traj.get(f"follow_{side}_gripper", 0)
    if not isinstance(grip, list):
        grip = [grip]
    return np.array(list(pos) + list(rot) + list(grip), dtype=np.float32)


def _encode_image(image_rgb: np.ndarray) -> str:
    """Encode RGB numpy image to base64 JPEG string (x2robot_client format)."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def load_vla_case_data(case_path, sample_name):
    """Load trajectory and video frames for a single case."""
    data = {
        "trajectory": None,
        "face_frames": [],
        "left_frames": [],
        "right_frames": [],
    }

    traj_path = os.path.join(case_path, f"{sample_name}.json")
    if os.path.exists(traj_path):
        with open(traj_path, "r") as f:
            data["trajectory"] = json.load(f)["data"]

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
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        cap.release()
        data[key] = np.array(frames)

    return data


async def run_vla_evaluation(uri, data_dir, save_dir, sample_nums=1):

    report_path = os.path.join(data_dir, "report.json")
    with open(report_path, "r") as f:
        samples = json.load(f)["sample_name"]

    instruction_path = os.path.join(data_dir, "instruction.json")
    instruction_dict = {}
    if os.path.exists(instruction_path):
        with open(instruction_path, "r") as f:
            instruction_dict = json.load(f)

    async with websockets.connect(uri, max_size=None) as websocket:
        metadata = msgpack.unpackb(await websocket.recv())
        print(f"Connected to VLA Server, metadata: {metadata}")

        for sample in samples[:sample_nums]:

            sample_instruction = instruction_dict.get(sample, {})
            if isinstance(sample_instruction, dict):
                prompt = sample_instruction.get("instruction", "")
            else:
                prompt = str(sample_instruction) if sample_instruction else ""
            print(f"Prompt: {prompt}")

            sample_path = os.path.join(data_dir, sample)
            save_path = os.path.join(save_dir, sample)
            print(f"\nProcessing: {sample}")

            case_data = load_vla_case_data(sample_path, sample)
            trajectory = case_data["trajectory"]
            if trajectory is None:
                print("No trajectory data, skipping")
                continue

            action_pred = []
            action_gt = []

            num_steps = min(len(trajectory), len(case_data["face_frames"]))
            idx = int(0.1 * num_steps)

            while len(action_pred) < num_steps - 10:
                traj = trajectory[idx]

                payload = {
                    "state": {
                        "follow1_pos": _build_follow_pos(traj, "left"),
                        "follow2_pos": _build_follow_pos(traj, "right"),
                    },
                    "views": {
                        "camera_front": _encode_image(case_data["face_frames"][idx]),
                        "camera_left": _encode_image(case_data["left_frames"][idx]),
                        "camera_right": _encode_image(case_data["right_frames"][idx]),
                    },
                    "instruction": prompt,
                }

                binary_data = msgpack.packb(payload, use_bin_type=True)
                await websocket.send(binary_data)

                response = await websocket.recv()
                if isinstance(response, str):
                    raise RuntimeError(f"Server error: {response}")
                result = msgpack.unpackb(response, raw=False)

                if "follow1_pos" not in result:
                    raise RuntimeError(
                        f"Server did not return follow1_pos, got keys: {list(result.keys())}"
                    )

                follow1 = np.array(result["follow1_pos"])
                follow2 = np.array(result["follow2_pos"])
                action_step = np.concatenate([follow1[1], follow2[1]], axis=0).tolist()

                if idx > 0:
                    idx += 1

                gt_traj = trajectory[min(idx, len(trajectory) - 1)]
                gt_left = _build_follow_pos(gt_traj, "left")
                gt_right = _build_follow_pos(gt_traj, "right")
                gt_step = np.concatenate([gt_left, gt_right]).tolist()
                action_gt.append(gt_step)
                action_pred.append(action_step)

                print(f"step {len(action_pred)}: pred_dim={len(action_step)}, gt_dim={len(gt_step)}")

                if len(action_pred) > 150:
                    break

            ml = min(len(action_pred), len(action_gt))
            plot_openloop(action_pred[:ml], action_gt[:ml], save_path, _ACTION_LABELS)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Open-loop evaluation via websocket")
    parser.add_argument("--uri", default="ws://localhost:42100", help="Server websocket URI")
    parser.add_argument("--data-dir", required=True, help="Path to evaluation data directory")
    parser.add_argument("--save-dir", required=True, help="Path to save plots")
    parser.add_argument("--sample-nums", type=int, default=1, help="Number of samples to evaluate")
    args = parser.parse_args()

    try:
        asyncio.run(
            run_vla_evaluation(args.uri, args.data_dir, args.save_dir, args.sample_nums)
        )
    except KeyboardInterrupt:
        print("\nEvaluation stopped by user.")
