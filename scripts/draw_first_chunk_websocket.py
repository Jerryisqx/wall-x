"""Draw chunk comparisons: for each sample, send observations at evenly spaced
positions, get full action horizon, plot predicted vs GT trajectories."""

import asyncio
import websockets
import msgpack
import msgpack_numpy as m
import numpy as np
import cv2
import os
import json
import base64
import argparse
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


def _build_follow_pos(traj, side):
    pos = traj.get(f"follow_{side}_position", [0, 0, 0])
    rot = traj.get(f"follow_{side}_rotation", [0, 0, 0])
    grip = traj.get(f"follow_{side}_gripper", 0)
    if not isinstance(grip, list):
        grip = [grip]
    return np.array(list(pos) + list(rot) + list(grip), dtype=np.float32)


def _encode_image(image_rgb: np.ndarray) -> str:
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buffer = cv2.imencode(".jpg", img_bgr)
    return base64.b64encode(buffer).decode("utf-8")


def load_vla_case_data(case_path, sample_name):
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


def plot_chunk(pred_chunk, gt_chunk, start_idx, chunk_idx, save_path, dim_labels=None):
    """Plot one predicted chunk vs GT chunk."""
    horizon = min(len(pred_chunk), len(gt_chunk))
    pred_chunk = np.array(pred_chunk[:horizon])
    gt_chunk = np.array(gt_chunk[:horizon])
    dim = min(pred_chunk.shape[1], gt_chunk.shape[1])

    fig, axes = plt.subplots(dim, 1, figsize=(14, 3.5 * dim))
    if dim == 1:
        axes = [axes]
    t = np.arange(horizon)

    for i in range(dim):
        ax = axes[i]
        ax.plot(t, pred_chunk[:, i], "o-", color="orange", markersize=3, label="Predicted")
        ax.plot(t, gt_chunk[:, i], "s-", color="blue", markersize=3, label="Ground Truth")
        label = dim_labels[i] if dim_labels and i < len(dim_labels) else f"dim {i}"
        ax.set_title(label)
        ax.set_xlabel("Step in chunk")
        ax.set_ylabel("Value")
        ax.legend(loc="upper right", fontsize=7)
        ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Chunk {chunk_idx} (obs@step {start_idx}, horizon={horizon})", fontsize=13
    )
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=200)
    plt.close()
    print(f"  Saved: {save_path}")


async def infer_one_chunk(websocket, case_data, trajectory, start_idx, instruction):
    """Send one observation and return (pred_chunk, gt_chunk) as numpy arrays."""
    traj = trajectory[start_idx]
    payload = {
        "state": {
            "follow1_pos": _build_follow_pos(traj, "left"),
            "follow2_pos": _build_follow_pos(traj, "right"),
        },
        "views": {
            "camera_front": _encode_image(case_data["face_frames"][start_idx]),
            "camera_left": _encode_image(case_data["left_frames"][start_idx]),
            "camera_right": _encode_image(case_data["right_frames"][start_idx]),
        },
        "instruction": instruction,
    }

    await websocket.send(msgpack.packb(payload, use_bin_type=True))
    response = await websocket.recv()
    if isinstance(response, str):
        raise RuntimeError(f"Server error: {response}")
    result = msgpack.unpackb(response, raw=False)

    if "follow1_pos" not in result:
        raise RuntimeError(f"No follow1_pos in result, keys: {list(result.keys())}")

    follow1 = np.array(result["follow1_pos"])  # (T+1, 7)
    follow2 = np.array(result["follow2_pos"])
    pred_chunk = np.concatenate([follow1[1:], follow2[1:]], axis=1)

    horizon = len(follow1) - 1
    gt_rows = []
    for t in range(horizon):
        gt_idx = min(start_idx + 1 + t, len(trajectory) - 1)
        gt_traj = trajectory[gt_idx]
        gt_rows.append(np.concatenate([
            _build_follow_pos(gt_traj, "left"),
            _build_follow_pos(gt_traj, "right"),
        ]))
    gt_chunk = np.array(gt_rows)

    return pred_chunk, gt_chunk


async def run_chunk_eval(uri, data_dir, save_dir, sample_nums=1, num_chunks=3):
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
        pred_horizon = metadata.get("pred_horizon", 32)
        print(f"Connected. metadata={metadata}, pred_horizon={pred_horizon}")

        for sample in samples[:sample_nums]:
            sample_instruction = instruction_dict.get(sample, {})
            if isinstance(sample_instruction, dict):
                instruction = sample_instruction.get("instruction", "")
            else:
                instruction = str(sample_instruction) if sample_instruction else ""

            sample_path = os.path.join(data_dir, sample)
            print(f"\nSample: {sample}  instruction: {instruction}")

            case_data = load_vla_case_data(sample_path, sample)
            trajectory = case_data["trajectory"]
            if trajectory is None:
                print("  No trajectory, skipping")
                continue

            num_steps = min(len(trajectory), len(case_data["face_frames"]))
            if num_steps < pred_horizon + 1:
                print(f"  Trajectory too short ({num_steps}), skipping")
                continue

            max_start = num_steps - pred_horizon - 1
            if num_chunks == 1:
                start_indices = [0]
            else:
                start_indices = np.linspace(0, max_start, num_chunks, dtype=int).tolist()

            print(f"  num_steps={num_steps}, drawing {len(start_indices)} chunks at {start_indices}")

            for ci, start_idx in enumerate(start_indices):
                pred_chunk, gt_chunk = await infer_one_chunk(
                    websocket, case_data, trajectory, start_idx, instruction
                )
                print(
                    f"  chunk {ci}: obs@step={start_idx}, "
                    f"pred={pred_chunk.shape}, gt={gt_chunk.shape}"
                )
                save_path = os.path.join(
                    save_dir, f"{sample}_chunk{ci}_step{start_idx}.jpg"
                )
                plot_chunk(
                    pred_chunk, gt_chunk, start_idx, ci, save_path, _ACTION_LABELS
                )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Draw chunk-level predicted vs GT trajectory comparisons"
    )
    parser.add_argument("--uri", default="ws://localhost:42100")
    parser.add_argument("--data-dir", required=True, help="Data directory with report.json")
    parser.add_argument("--save-dir", required=True, help="Directory to save plots")
    parser.add_argument("--sample-nums", type=int, default=1, help="Number of samples")
    parser.add_argument("--num-chunks", type=int, default=3,
                        help="Number of chunks to draw per sample (evenly spaced from start to end)")
    args = parser.parse_args()

    try:
        asyncio.run(
            run_chunk_eval(
                args.uri, args.data_dir, args.save_dir,
                args.sample_nums, args.num_chunks,
            )
        )
    except KeyboardInterrupt:
        print("\nStopped by user.")
