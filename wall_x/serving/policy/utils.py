from typing import Dict, List
import logging
import numpy as np
from wall_x.data.utils import preprocesser_call
from qwen_vl_utils.vision_process import smart_resize
import torch
from PIL import Image
from transformers import BatchFeature

import numpy as np
from scipy.signal import savgol_filter
from scipy.spatial.transform import Rotation as R  # TODO：转换成numba的函数
# from collections import deque
# import threading

# from wall_x.infer.logger import InferLogger

logger = logging.getLogger(__name__)


def prepare_batch(
    obs: Dict,
    processor,
    normalizer_propri,
    camera_key: List[str],
    agent_pos_dim,
    action_dim,
    pred_horizon,
    fixed_action_dim,
    max_length,
    image_factor: int,
    min_pixels: int,
    max_pixels: int,
    predict_mode: str = "fast",
    device: str = "cuda",
    use_state_string_representation: bool = False,
    state_bins: int = 512,
) -> BatchFeature:
    """Prepare observation into model input format.

    Args:
        obs: Dictionary containing:
            - 'camera_key_0' : image 0
            - 'camera_key_1' : image 1
            ...
            - 'prompt': Text prompt
            - 'state': Robot state/proprioception
            - 'dataset_names': Dataset names

    Returns:
        BatchFeature object ready for model input
    """
    # Handle images - can be single image, list of images, or dict of images
    images = []
    images = [obs[key] for key in camera_key]
    # Convert numpy arrays to PIL Images
    processed_images = []
    for img in images:
        if isinstance(img, np.ndarray):
            # Debug: Log the shape and dtype
            logger.debug(f"Image shape: {img.shape}, dtype: {img.dtype}")

            # Handle unexpected dimensions - squeeze if needed
            if img.ndim > 3:
                logger.warning(
                    f"Image has {img.ndim} dimensions, squeezing extra dimensions"
                )
                img = np.squeeze(img)

            # Verify shape is valid for PIL
            if img.ndim == 2:
                # Grayscale image
                pass
            elif img.ndim == 3:
                # Check if channel dimension is first or last
                if img.shape[0] == 3 or img.shape[0] == 1:
                    # Channels first, transpose to channels last
                    img = np.transpose(img, (1, 2, 0))
                elif img.shape[2] == 3 or img.shape[2] == 1:
                    # Already channels last
                    pass
                else:
                    raise ValueError(
                        f"Unexpected image shape: {img.shape}. Expected (H, W, C) or (C, H, W)"
                    )
            else:
                raise ValueError(
                    f"Invalid image dimensions: {img.ndim}. Expected 2 or 3 dimensions, got shape {img.shape}"
                )

            # Convert to PIL Image
            if img.dtype == np.uint8:
                img = Image.fromarray(img)
            else:
                img = Image.fromarray((img * 255).astype(np.uint8))
        processed_images.append(img)

    # print("processed_images:",processed_images)
    # Apply smart resize to images
    resized_images = process_images(
        processed_images, image_factor, min_pixels, max_pixels
    )

    # Handle robot state/proprioception
    propri_string = None
    state = None
    if "state" in obs:
        state = obs["state"]
        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        if state.dim() == 1:
            state = state.unsqueeze(0)
        if state.dim() == 2:
            state = state.unsqueeze(1)  # [batch, 1, state_dim]

        agent_pos_mask = (~torch.isnan(state)).float()
        state = torch.nan_to_num(state, nan=0.0)

        state = normalizer_propri.normalize_data(state, [obs["dataset_names"]] * state.shape[0])

        if use_state_string_representation:
            norm_np = state.cpu().numpy()
            discretized = np.digitize(norm_np, bins=np.linspace(-1, 1, state_bins + 1)[:-1]) - 1
            discretized = discretized[:, 0, :]
            mask_np = agent_pos_mask[:, 0, :].cpu().numpy().astype(bool) if isinstance(agent_pos_mask, torch.Tensor) else agent_pos_mask[:, 0, :].astype(bool)
            propri_string = " ".join(map(str, discretized[0, mask_np[0]]))

    # Handle text prompt - format with vision tokens
    instruction = obs["prompt"]
    formatted_text = format_text_with_vision_tokens(
        instruction, camera_key, predict_mode, pred_horizon, propri_string=propri_string
    )
    # print("formatted_text:",formatted_text)
    # Use processor to prepare inputs
    inputs = preprocesser_call(
        processor=processor,
        text=[formatted_text],
        images=[resized_images],
        videos=None,
        padding=True,
        truncation=True,
        return_tensors="pt",
        max_length=max_length,
    )

    action_token_id = processor.tokenizer.convert_tokens_to_ids("<|action|>")
    moe_token_types = inputs.input_ids == action_token_id
    inputs["moe_token_types"] = torch.tensor(moe_token_types)
    # decoded_text = processor.tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=False)
    # print(f"decoded text: {decoded_text}")

    if state is not None and not use_state_string_representation:
        inputs["proprioception"] = state
        inputs["agent_pos_mask"] = agent_pos_mask

    # Add dataset name (required by model)
    batch_size = state.shape[0] if state is not None else 1
    inputs["dataset_names"] = [obs["dataset_names"]] * batch_size

    # Move all tensors to device
    for key in inputs:
        if isinstance(inputs[key], torch.Tensor):
            inputs[key] = inputs[key].to(device)

    dof_mask = torch.ones([state.shape[0], pred_horizon, fixed_action_dim])
    dof_mask[:, :, action_dim:] = 0

    inputs["dof_mask"] = dof_mask

    inputs["dof_mask"][:,:,20:] = 0
    # inputs["agent_pos_mask"][:,:,20:] = 0

    # Convert to BatchFeature to maintain consistency with training pipeline
    return BatchFeature(data=dict(inputs)).to(device)


def process_images(
    images: List[Image.Image], image_factor: int, min_pixels: int, max_pixels: int
) -> List[Image.Image]:
    """Process images with smart resize following the data loading pattern.

    Args:
        images: List of PIL Images

    Returns:
        List of resized PIL Images
    """
    resized_images = []
    for img_pil in images:

        orig_width, orig_height = img_pil.size
        target_size = 256
        if target_size != -1:
            # Maintain aspect ratio logic
            if orig_width > orig_height:  # Landscape image
                new_width = target_size
                new_height = int(target_size * orig_height / orig_width)
            else:  # Portrait image
                new_height = target_size
                new_width = int(target_size * orig_width / orig_height)
            img_pil = img_pil.resize((new_width, new_height))

        # Apply smart scaling (Qwen logic)
        current_width, current_height = img_pil.size
        resized_height, resized_width = smart_resize(
            current_height,
            current_width,
            factor=image_factor,
            min_pixels=min_pixels,
            max_pixels=max_pixels,
        )

        resized_img = img_pil.resize((resized_width, resized_height))
        resized_images.append(resized_img)

    return resized_images


def format_text_with_vision_tokens(
    instruction: str,
    camera_key: List[str],
    predict_mode: str = "diffusion",
    pred_horizon: int = 32,
    propri_string: str | None = None,
) -> str:
    """Format text prompt with vision tokens for the model.

    Args:
        instruction: Task instruction text
        camera_key: List of camera names

    Returns:
        Formatted text with special tokens
    """
    # Special tokens for formatting
    role_start_symbol = "<|im_start|>"
    role_end_symbol = "<|im_end|>"
    vision_start_symbol = "<|vision_start|>"
    vision_end_symbol = "<|vision_end|>"
    image_pad_symbol = "<|image_pad|>"
    propri_symbol = "<|propri|>"
    action_symbol = "<|action|>"
    action_fast_symbol = "<|action_fast|>"

    # Camera name mapping
    camera_name_mapping = {
        "front_view": "front view",
        "face_view": "front view",
        "left_wrist_view": "left wrist view",
        "right_wrist_view": "right wrist view",
        "top_view": "top view",
        "wall_view": "wall view",
    }

    # System prologue
    prologue = (
        f"{role_start_symbol}system\nYou are a helpful assistant.{role_end_symbol}\n"
    )

    # User request with observation
    user_request = f"{role_start_symbol}user\nObservation:"
    if camera_key:
        for cam_name in camera_key:
            view_name = camera_name_mapping.get(cam_name, cam_name)
            user_request += f" {view_name}: {vision_start_symbol}{image_pad_symbol}{vision_end_symbol}"
    user_request += "\nInstruction:"

    propri_content = propri_string if propri_string is not None else propri_symbol
    text_prompt = (
        f"\nPredict the next action in robot action.\nProprioception: {propri_content}\n"
    )
    user_message = f"{user_request} {instruction}{text_prompt}{role_end_symbol}\n"
    assistant_output = (
        f"{role_start_symbol}assistant\n{action_fast_symbol}{role_end_symbol}\n"
    )
    # if predict_mode in ("diffusion", "slow"):
    assistant_output = f"{role_start_symbol}assistant\n{action_symbol * pred_horizon}{role_end_symbol}\n"
    complete_text = prologue + user_message + assistant_output

    return complete_text



# 机械臂轨迹参数
ARM_MAX_VELOCITY = 0.02
ARM_EXECUTION_HZ = 20
ARM_MIN_EXECUTION_TIME = 5.0
ARM_MAX_EXECUTION_TIME = 15.0



class UnifiedTrajectoryProcessor:
    """统一轨迹处理器"""

    @staticmethod
    def interpolate_trajectory_batch(trajectories, target_length, smooth=True):
        """
        批量插值多个轨迹到统一长度
        Args:
            trajectories: list of np.array, 每个数组shape为(N, D)
            target_length: int, 目标长度
            smooth: bool, 是否平滑
        Returns:
            list of np.array, 插值后的轨迹
        """
        if not trajectories:
            return []

        results = []
        for traj in trajectories:
            if len(traj) == 0:
                results.append(np.zeros((target_length, traj.shape[1])))
                continue

            if len(traj) == target_length:
                results.append(traj)
                continue

            # 向量化插值
            original_indices = np.linspace(0, len(traj) - 1, len(traj))
            target_indices = np.linspace(0, len(traj) - 1, target_length)

            # 处理不同类型的数据
            if traj.shape[1] == 7:  # 机械臂数据 [x,y,z,rx,ry,rz,gripper]
                interpolated = UnifiedTrajectoryProcessor._interpolate_arm_trajectory(
                    traj, original_indices, target_indices, target_length
                )
            else:  # 其他数据(高度、电流等)
                interpolated = np.zeros((target_length, traj.shape[1]))
                for i in range(traj.shape[1]):
                    interpolated[:, i] = np.interp(
                        target_indices, original_indices, traj[:, i]
                    )

            # 平滑处理
            if smooth and len(interpolated) >= 5:
                interpolated = UnifiedTrajectoryProcessor._smooth_trajectory(
                    interpolated
                )

            results.append(interpolated)

        return results

    @staticmethod
    def _interpolate_arm_trajectory(
        traj, original_indices, target_indices, target_length
    ):
        """优化的机械臂轨迹插值"""
        interpolated = np.zeros((target_length, 7))

        # 向量化插值位置和夹爪
        for i in [0, 1, 2, 6]:  # x, y, z, gripper
            interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])

        # 四元数插值(向量化)
        quaternions = R.from_euler("xyz", traj[:, 3:6]).as_quat()
        interpolated_quats = np.zeros((target_length, 4))
        for i in range(4):
            interpolated_quats[:, i] = np.interp(
                target_indices, original_indices, quaternions[:, i]
            )

        # 批量归一化
        norms = np.linalg.norm(interpolated_quats, axis=1, keepdims=True)
        interpolated_quats = interpolated_quats / norms

        # 批量转换回欧拉角
        interpolated[:, 3:6] = R.from_quat(interpolated_quats).as_euler("xyz")

        return interpolated

    @staticmethod
    def _interpolate_position_trajectory(
        traj, original_indices, target_indices, target_length
    ):
        """优化的位置轨迹插值"""
        interpolated = np.zeros((target_length, 3))
        for i in range(3):
            interpolated[:, i] = np.interp(target_indices, original_indices, traj[:, i])
        return interpolated

    @staticmethod
    def _smooth_trajectory(trajectory):
        """向量化平滑处理"""
        if len(trajectory) < 5:
            return trajectory

        try:
            # 批量平滑所有维度
            smoothed = np.zeros_like(trajectory)
            for dim in range(trajectory.shape[1]):
                smoothed[:, dim] = savgol_filter(
                    trajectory[:, dim],
                    min(
                        5,
                        (
                            len(trajectory)
                            if len(trajectory) % 2 == 1
                            else len(trajectory) - 1
                        ),
                    ),
                    3,
                    mode="nearest",
                )
            return smoothed
        except Exception:
            return trajectory

    @staticmethod
    def calculate_optimal_trajectory_length(left_traj, right_traj):
        """计算最优轨迹长度"""

        # 向量化距离计算
        def calc_distance(traj):
            if len(traj) < 2:
                return 0.0
            pos_diff = traj[1:, :3] - traj[:-1, :3]
            return np.sum(np.linalg.norm(pos_diff, axis=1))

        distances = [calc_distance(left_traj), calc_distance(right_traj)]
        max_distance = max(distances)

        if max_distance > 1e-6:
            execution_time = np.clip(
                max_distance / ARM_MAX_VELOCITY,
                ARM_MIN_EXECUTION_TIME,
                ARM_MAX_EXECUTION_TIME,
            )
        else:
            execution_time = ARM_MIN_EXECUTION_TIME

        return max(int(execution_time * ARM_EXECUTION_HZ), len(left_traj))


