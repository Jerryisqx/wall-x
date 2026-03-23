import logging
from typing import Dict, Any, List
import torch
import copy
import numpy as np
import base64
import copy
import cv2
import time
from wall_x.serving.websocket_policy_server import BasePolicy
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.serving.policy.utils import prepare_batch
from wall_x.model.model_utils import load_wallx_processors, register_normalizers
from wall_x.data.utils import maybe_expand_rotation_to_6d, convert_6D_to_euler, convert_euler_to_6D
from wall_x.data.data_utils import euler_to_matrix_zyx_6d_nb,compose_state_and_delta_to_abs_6d,so3_to_euler_zyx_batch_nb
from wall_x.serving.policy.utils import UnifiedTrajectoryProcessor
import time
import os
from matplotlib import pyplot as plt
# 注册 msgpack-numpy 扩展
import msgpack_numpy as m
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

logger = logging.getLogger(__name__)


def _decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 string to numpy image array (RGB format)."""
    if base64_str is None:
        return np.zeros((480, 640, 3), dtype=np.uint8)
    try:
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return np.zeros((480, 640, 3), dtype=np.uint8)
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return np.zeros((480, 640, 3), dtype=np.uint8)
    

class WallXPolicy(BasePolicy):
    """Policy wrapper for Wall-X model that implements the BasePolicy interface."""

    def __init__(
        self,
        model_path: str,
        train_config: dict,
        action_tokenizer_path: str | None,
        action_dim: int,
        agent_pos_dim: int,
        pred_horizon: int,
        camera_key: List[str],
        device: str = "cuda",
        dtype: str = "bfloat16",
        predict_mode: str = "diffusion",
        default_prompt: str | None = None,
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
        image_factor: int = 28,
        max_length: int = 2048,
    ):
        """Initialize the Wall-X policy.

        Args:
            model_path: Path to the pretrained model checkpoint
            action_tokenizer_path: Path to the action tokenizer
            action_dim: Dimension of action space
            pred_horizon: Prediction horizon for actions
            device: Device to run model on ('cuda' or 'cpu')
            dtype: Data type for model ('bfloat16', 'float16', or 'float32')
            predict_mode: Prediction mode ('fast' or 'slow')
            default_prompt: Default text prompt for the model
            min_pixels: Minimum pixels for image resizing
            max_pixels: Maximum pixels for image resizing
            image_factor: Factor for smart resize
            max_length: Maximum sequence length for text
        """
        logger.info(f"Loading Wall-X model from {model_path}")

        self.normalizer_action, self.normalizer_propri = register_normalizers(
            train_config, model_path
        )

        self.model = Qwen2_5_VLMoEForAction.from_pretrained(
            model_path,
            train_config=train_config,
            action_tokenizer_path=action_tokenizer_path,
        )
        self.model.set_normalizer(
            copy.deepcopy(self.normalizer_action), copy.deepcopy(self.normalizer_propri)
        )
        self.model.eval()
        self.model = self.model.to(device)
        self.model.to_bfloat16_for_selected_params()

        # hard code the action dim to 20 for align to wall-x configuration
        self.fixed_action_dim = 26

        self.action_dim = action_dim
        self.agent_pos_dim = action_dim
        self.pred_horizon = pred_horizon
        self.device = device
        self.predict_mode = predict_mode
        self.default_prompt = default_prompt
        self.camera_key = camera_key

        # Image preprocessing config
        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_factor = image_factor
        self.max_length = max_length

        self.use_state_string_representation = train_config.get("data", {}).get(
            "use_state_string_representation", train_config.get("use_state_string_representation", False)
        )
        self.state_bins = train_config.get("data", {}).get("state_bins", 512)
        self._obs_action_keys = train_config.get("data", {}).get("obs_action_keys", [])
        self._agent_pos_config = train_config.get("agent_pos_config", {})


        print("predict_mode", predict_mode)
        print("camera_key", camera_key)
        print("use_state_string_representation", self.use_state_string_representation)
        print("state_bins", self.state_bins)

        # Load processor
        logger.info("Loading processor and tokenizer...")

        processors_dict = load_wallx_processors(train_config)
        self.processor = processors_dict["processor"]

        # Action buffer for multi-step predictions
        self.action_buffer = []
        self.buffer_index = 0

        logger.info(
            f"Model loaded successfully. Device: {device}, Action dim: {action_dim}, Horizon: {pred_horizon}"
        )
        if "x2_normal" in self.normalizer_propri.min:
            print(f"normalizer_propri x2_normal min: {self.normalizer_propri.min['x2_normal'].data}")
            print(f"normalizer_propri x2_normal delta: {self.normalizer_propri.delta['x2_normal'].data}")
            print(f"normalizer_propri x2_normal dim: {self.normalizer_propri.min['x2_normal'].shape}")

        if "x2_normal" in self.normalizer_action.min:
            print(f"normalizer_action x2_normal min: {self.normalizer_action.min['x2_normal'].data}")
            print(f"normalizer_action x2_normal delta: {self.normalizer_action.delta['x2_normal'].data}")
            print(f"normalizer_action x2_normal dim: {self.normalizer_action.min['x2_normal'].shape}")

    @property
    def metadata(self) -> Dict[str, Any]:
        """Return metadata about the policy."""
        return {
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "device": self.device,
            "predict_mode": self.predict_mode,
        }

    def reset(self) -> None:
        """Reset the policy state."""
        self.action_buffer = []
        self.buffer_index = 0
        logger.debug("Policy reset")

    def _convert_x2robot_obs(self, obs: Dict) -> Dict:
        """Convert x2robot client format to Wall-X format.
        
        x2robot format:
        {
            "state": {"follow1_pos": [...], "follow2_pos": [...], ...},
            "views": {"camera_front": base64, "camera_left": base64, "camera_right": base64},
            "instruction": ["task description"]
        }
        
        Wall-X format:
        {
            "face_view": np.ndarray,
            "left_wrist_view": np.ndarray,
            "right_wrist_view": np.ndarray,
            "prompt": str,
            "state": flat_array,
            "dataset_names": ["x2"]
        }
        """
        # Check if this is x2robot format (has "views" key)
        if "views" not in obs:
            # Already in Wall-X format or unknown format
            return obs
        
        logger.debug("Converting x2robot observation format to Wall-X format")
        
        converted = {}
        
        # 1. Convert images from base64 to numpy arrays
        views = obs.get("views", {})
        camera_mapping = {
            "camera_front": "face_view",
            "camera_left": "left_wrist_view", 
            "camera_right": "right_wrist_view",
        }
        
        for src_key, dst_key in camera_mapping.items():
            img_data = views.get(src_key)
            if isinstance(img_data, str):
                # base64 encoded
                img = _decode_base64_image(img_data)
            elif isinstance(img_data, np.ndarray):
                img = img_data
            # else:
            #     img = np.zeros((256, 256, 3), dtype=np.uint8)
            
            # Resize image to match training resolution
            target_height = 256
            # img = _resize_image_by_height(img, target_height)
            converted[dst_key] = img
        
        # 2. Convert state from nested dict to flat array
        state_dict = obs.get("state", {})
        
        # Extract state components in order (matching training data format)
        # Order: follow1_pos(7) + follow2_pos(7) = 14 dims for dual-arm
        follow1_pos = state_dict.get("follow1_pos", np.zeros(1, dtype=np.float32))
        follow2_pos = state_dict.get("follow2_pos", np.zeros(1, dtype=np.float32))
        
        # Flatten to 1D arrays if needed
        if hasattr(follow1_pos, 'flatten'):
            follow1_pos = follow1_pos.flatten()
        if hasattr(follow2_pos, 'flatten'):
            follow2_pos = follow2_pos.flatten()
        
        # Concatenate state
        state_flat = np.concatenate([
            np.array(follow1_pos, dtype=np.float32),
            np.array(follow2_pos, dtype=np.float32),
            np.array([0]*6, dtype=np.float32),
        ])
        
        converted["state"] = state_flat
        
        # 3. Convert instruction to prompt
        instruction = obs.get("instruction", [""])
        if isinstance(instruction, np.ndarray):
            instruction = instruction.tolist()
        if isinstance(instruction, list) and len(instruction) > 0:
            prompt = str(instruction[0])
        elif isinstance(instruction, str):
            prompt = instruction
        else:
            prompt = self.default_prompt or "Execute the task."
        
        converted["prompt"] = prompt
        
        # 4. Add dataset_names (use the dataset name from training config)
        # converted["dataset_names"] = "pick_up_cup_with_certain_color"
        converted["dataset_names"] ="ex_normal"
        
        logger.debug(f"Converted obs keys: {list(converted.keys())}, state shape: {converted['state'].shape}")
        
        return converted


    def infer(self, obs: Dict) -> Dict:
        """Infer action from observation.

        Args:
            obs: Dictionary containing:
                - 'image': Image observation (numpy array or PIL Image)
                - 'prompt': Optional text prompt
                - 'state': Optional robot state
                - Other modality-specific observations

        Returns:
            Dictionary containing:
                - 'action': Predicted action (numpy array)
                - Additional metadata
        """

        obs_copy = copy.deepcopy(obs)
        print("obs_copy: ",obs_copy['state'])
        if "state" in obs and self._obs_action_keys:
            obs["state"]["follow1_pos"] = np.array(obs["state"]["follow1_pos"]).reshape(1, -1)
            obs["state"]["follow2_pos"] = np.array(obs["state"]["follow2_pos"]).reshape(1, -1)

            rotation1 = euler_to_matrix_zyx_6d_nb(obs["state"]["follow1_pos"][:,3:6])[0]
            if rotation1.shape[0] == 1:
                rotation1 = rotation1[0]
            # print("rotation1 shape: ",rotation1.shape)
            # print(obs["state"]["follow1_pos"].shape)
            obs["state"]["follow1_pos"] = np.concatenate([obs["state"]["follow1_pos"][0,:3],rotation1,obs["state"]["follow1_pos"][0,6:7]],axis=0)

            rotation2 = euler_to_matrix_zyx_6d_nb(obs["state"]["follow2_pos"][:,3:6])[0]
            if rotation2.shape[0] == 1:
                rotation2 = rotation2[0]
            obs["state"]["follow2_pos"] = np.concatenate([obs["state"]["follow2_pos"][0,:3],rotation2,obs["state"]["follow2_pos"][0,6:7]],axis=0)
        
        obs = self._convert_x2robot_obs(obs)

        state = obs.get("state")

        try:
            # Need to predict new actions
            input_batch = prepare_batch(
                obs,
                self.processor,
                self.normalizer_propri,
                self.camera_key,
                self.agent_pos_dim,
                self.action_dim,
                self.pred_horizon,
                self.fixed_action_dim,
                self.max_length,
                self.image_factor,
                self.min_pixels,
                self.max_pixels,
                self.predict_mode,
                self.device,
                self.use_state_string_representation,
                self.state_bins,
            )

            with torch.no_grad():
                outputs = self.model(
                    **input_batch,
                    action_dim=(
                        self.action_dim
                        if self.predict_mode == "fast"
                        else self.fixed_action_dim
                    ),
                    action_horizon=self.pred_horizon,
                    mode="predict",
                    predict_mode=self.predict_mode,
                )

            if outputs["predict_action"] is None:
                predicted_actions = np.zeros(
                    [1, self.pred_horizon, self.action_dim]
                ).astype(np.float32)

            predicted_actions = (
                outputs["predict_action"][:, :, : self.action_dim]
                .detach()
                .cpu()
                .to(torch.float32)
                .numpy()
            )
            actions = predicted_actions[0]  # (T, action_dim)

            # print("state.shape:",state[None,:].shape)
            # actions = state[None,:]+actions/
            
            # result = {"action": predicted_actions}
            result = {}

            state_pos_left = state[None,:][:,:3]
            state_rotation_left = state[None,:][:,3:9]
            state_gripper_left = state[None,:][:,9:10]
            state_pos_right = state[None,:][:,10:13]
            state_rotation_right = state[None,:][:,13:19]
            state_gripper_right = state[None,:][:,19:20]


            action_pos_left = state_pos_left+actions[:, :3]
            action_rotation_left = compose_state_and_delta_to_abs_6d(actions[:, 3:9], state_rotation_left[0])
            action_gripper_left = actions[:, 9:10]
            action_pos_right = state_pos_right+actions[:, 10:13]
            action_rotation_right = compose_state_and_delta_to_abs_6d(actions[:, 13:19],state_rotation_right[0])
            action_gripper_right = actions[:, 19:20]

            # action_pos_left = actions[:, :3]
            # action_rotation_left = actions[:, 3:9]
            # action_gripper_left = actions[:, 9:10]
            # action_pos_right = actions[:, 10:13]
            # action_rotation_right = actions[:, 13:19]
            # action_gripper_right = actions[:, 19:20]

            # print(result)

            rotation1 = so3_to_euler_zyx_batch_nb(action_rotation_left)
            result["follow1_pos"] = np.concatenate([action_pos_left,rotation1,action_gripper_left],axis=1)
            result["follow1_pos"] = np.concatenate([np.array(obs_copy["state"]["follow1_pos"]).reshape(1, -1),result["follow1_pos"]])  #[32,7]
            result["follow1_pos"] = result["follow1_pos"]
            rotation2 = so3_to_euler_zyx_batch_nb(action_rotation_right)
            result["follow2_pos"] = np.concatenate([action_pos_right,rotation2,action_gripper_right],axis=1)
            result["follow2_pos"] = np.concatenate([np.array(obs_copy["state"]["follow2_pos"]).reshape(1, -1),result["follow2_pos"]])  #[32,7]
            result["follow2_pos"] = result["follow2_pos"]


            robot_action_interpolate_multiplier = 2
            target_length = robot_action_interpolate_multiplier * len(result["follow1_pos"])
            result["follow1_pos"], result["follow2_pos"] = (
                UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
                    [result["follow1_pos"], result["follow2_pos"]], target_length
                )
            )

            result["follow1_pos"]=result["follow1_pos"][:64,:].tolist()
            result["follow2_pos"]=result["follow2_pos"][:64,:].tolist()

            print("result:",result)
            b=result
            gt_b=np.concatenate([np.array(b["follow1_pos"]),np.array(b["follow2_pos"])],axis=1)
            tt=int(time.time())
            sp=f"/x2robot_v2/yangping/github/jerry_git/wall-x/workspace/server/visualize/b-10-141-parcel-joint2-{tt}.png"
            plot_openloop(gt_b,gt_b,sp)
            # time.sleep(3)
            
            return result

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
