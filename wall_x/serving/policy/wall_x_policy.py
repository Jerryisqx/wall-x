import logging
import re
from typing import Dict, Any, List
import torch
import base64
import cv2
import copy
import numpy as np
from wall_x.serving.websocket_policy_server import BasePolicy
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.serving.policy.utils import prepare_batch
from wall_x.model.model_utils import load_wallx_processors, register_normalizers
from wall_x.infer.base_dataclass import RobotStateActionData
from wall_x.infer.utils import UnifiedTrajectoryProcessor

logger = logging.getLogger(__name__)

ROBOT_ACTION_KEY_MAPPING = {
    "follow_left_ee_cartesian_pos": "follow1_pos[:3]",
    "follow_left_ee_rotation": "follow1_pos[3:6]",
    "follow_left_gripper": "follow1_pos[6:7]",
    "follow_right_ee_cartesian_pos": "follow2_pos[:3]",
    "follow_right_ee_rotation": "follow2_pos[3:6]",
    "follow_right_gripper": "follow2_pos[6:7]",
    "head_actions": "head_pos",
    "head_rotation": "head_pos",
    "height": "lift",
    "velocity_decomposed": "velocity_decomposed",
}


def _decode_base64_image(base64_str: str) -> np.ndarray:
    """Decode base64 JPEG string to numpy RGB array. Returns None on failure."""
    if base64_str is None:
        return None
    try:
        img_bytes = base64.b64decode(base64_str)
        nparr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    except Exception as e:
        logger.warning(f"Failed to decode base64 image: {e}")
        return None


class WallXPolicy(BasePolicy):
    """Policy wrapper for Wall-X model.

    Accepts x2robot_client input format:
        state: dict with follow1_pos, follow2_pos, head_pos, lift, ...
        views: dict with camera base64 or numpy images
        instruction / prompt: task description
    Returns x2robot_client output format:
        follow1_pos / follow2_pos: np.float32 (T+1, 7), row 0 = current state
    """

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
        dataset_name: str = "ex_normal",
        min_pixels: int = 4 * 28 * 28,
        max_pixels: int = 16384 * 28 * 28,
        image_factor: int = 28,
        max_length: int = 2048,
    ):
        logger.info(f"Loading Wall-X model from {model_path}")
        self.config = train_config

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

        self.fixed_action_dim = action_dim
        self.action_dim = action_dim
        self.agent_pos_dim = action_dim
        self.pred_horizon = pred_horizon
        self.device = device
        self.predict_mode = predict_mode
        self.default_prompt = default_prompt
        self.dataset_name = dataset_name
        self.camera_key = camera_key

        self.min_pixels = min_pixels
        self.max_pixels = max_pixels
        self.image_factor = image_factor
        self.max_length = max_length

        self.use_state_string_representation = train_config.get("data", {}).get(
            "use_state_string_representation",
            train_config.get("use_state_string_representation", False),
        )
        self.state_bins = train_config.get("data", {}).get("state_bins", 512)
        self._obs_action_keys = train_config.get("data", {}).get("obs_action_keys", [])
        self._predict_action_keys = train_config.get("data", {}).get("predict_action_keys", [])
        self._agent_pos_config = train_config.get("agent_pos_config", {})
        self._dof_config = train_config.get("dof_config", {})

        logger.info("Loading processor and tokenizer...")
        processors_dict = load_wallx_processors(train_config)
        self.processor = processors_dict["processor"]

        self.action_buffer = []
        self.buffer_index = 0
        self.action_start_ratio = 0.0
        self.action_end_ratio = 1.0
        self.action_interpolate_multiplier = 2

        logger.info(
            f"Model loaded. Device: {device}, Action dim: {action_dim}, "
            f"Horizon: {pred_horizon}, Dataset: {dataset_name}"
        )
        logger.info(f"predict_mode={predict_mode}, camera_key={camera_key}")
        logger.info(f"obs_action_keys={self._obs_action_keys}")
        logger.info(f"predict_action_keys={self._predict_action_keys}")

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "device": self.device,
            "predict_mode": self.predict_mode,
            "dataset_name": self.dataset_name,
        }

    def reset(self) -> None:
        self.action_buffer = []
        self.buffer_index = 0

    def _build_robot_state(self, state_dict) -> RobotStateActionData:
        """Build RobotStateActionData from a client state dict.

        Parses keys like follow1_pos, follow2_pos using ROBOT_ACTION_KEY_MAPPING.
        """
        rsad = RobotStateActionData()
        for key, state_key_str in ROBOT_ACTION_KEY_MAPPING.items():
            value = None
            match = re.match(r"(\w+)\[(.*)\]", state_key_str)
            if match:
                base_key = match.group(1)
                if base_key in state_dict:
                    slicing_str = match.group(2)
                    slice_parts = slicing_str.split(":")
                    slice_args = [(int(p) if p.strip() else None) for p in slice_parts]
                    s = slice(*slice_args)
                    value = np.asarray(state_dict[base_key])[s]
            else:
                if state_key_str in state_dict:
                    value = state_dict[state_key_str]
            if value is not None:
                rsad.save_state_data_with_key(np.asarray(value)[None], key)
        return rsad

    def _postprocess_with_rsad(self, predicted_actions, rsad):
        """Convert model output to x2robot_client format via RSAD auto-conversion.

        Returns:
        - follow1_pos: np.float32 (T+1, 7) with row 0 = current state
        - follow2_pos: np.float32 (T+1, 7) with row 0 = current state
        - predict_action: raw model output
        """
        rsad.save_action_data(predicted_actions, self._predict_action_keys)

        result = {"predict_action": predicted_actions}

        for side in ("left", "right"):
            pos = rsad.data.get(f"action_{side}_ee_cartesian_pos")
            rot = rsad.data.get(f"action_{side}_ee_rotation")
            grip = rsad.data.get(f"action_{side}_gripper")
            parts = [p for p in (pos, rot, grip) if p is not None]
            if not parts:
                continue

            arm_key = "follow1_pos" if side == "left" else "follow2_pos"
            arm_action = np.concatenate(parts, axis=1)

            state_pos = rsad.data.get(f"state_{side}_ee_cartesian_pos")
            state_rot = rsad.data.get(f"state_{side}_ee_rotation")
            state_grip = rsad.data.get(f"state_{side}_gripper")
            state_parts = [p for p in (state_pos, state_rot, state_grip) if p is not None]
            if state_parts:
                state_row = np.concatenate(state_parts, axis=1)
                arm_action = np.concatenate([state_row, arm_action], axis=0)

            result[arm_key] = arm_action.astype(np.float32)

        # print(result)
        return result

    def _serialize_actions(
        self,
        result: Dict,
        interpolate_multiplier: int = None,
        start_ratio: float = None,
        end_ratio: float = None,
    ) -> Dict:
        """Trim and interpolate action trajectories.

        Each follow*_pos has shape (T+1, 7) where row 0 is current state.
        1. Skip row 0 (state), trim action rows by [start_ratio, end_ratio]
        2. Interpolate to target_length if multiplier > 1
        3. Prepend state row back (for client interpolation starting point)
        """
        if interpolate_multiplier is None:
            interpolate_multiplier = self.action_interpolate_multiplier
        if start_ratio is None:
            start_ratio = self.action_start_ratio
        if end_ratio is None:
            end_ratio = self.action_end_ratio

        arm_keys = [k for k in ("follow1_pos", "follow2_pos") if k in result]
        if not arm_keys:
            return result

        trimmed = {}
        state_rows = {}
        for key in arm_keys:
            traj = np.asarray(result[key])
            state_rows[key] = traj[:1]
            actions = traj[1:]
            actual_len = len(actions)
            s = int(start_ratio * actual_len)
            e = int(end_ratio * actual_len)
            trimmed[key] = actions[s:e]

        if interpolate_multiplier >= 1 and len(trimmed[arm_keys[0]]) > 0:
            target_length = interpolate_multiplier * len(trimmed[arm_keys[0]])
            interpolated = UnifiedTrajectoryProcessor.interpolate_trajectory_batch(
                [trimmed[k] for k in arm_keys], target_length
            )
            for i, key in enumerate(arm_keys):
                trimmed[key] = interpolated[i]

        for key in arm_keys:
            result[key] = np.concatenate(
                [state_rows[key], trimmed[key]], axis=0
            ).astype(np.float32)

        return result

    def _normalize_client_input(self, obs: Dict) -> None:
        """Normalize x2robot_client input in-place for the model pipeline.

        - views (base64/numpy) → flatten to obs top-level
        - instruction → prompt
        - dataset_names from self.dataset_name
        """
        if "views" in obs:
            for cam_key, img_data in obs["views"].items():
                if isinstance(img_data, str):
                    obs[cam_key] = _decode_base64_image(img_data)
                elif isinstance(img_data, np.ndarray):
                    obs[cam_key] = img_data
            del obs["views"]

        if "instruction" not in obs:
            instr = obs.pop("prompt", None) or self.default_prompt
            if instr is not None:
                obs["instruction"] = str(instr)
        else:
            instr = obs["instruction"]
            if isinstance(instr, np.ndarray):
                instr = instr.flat[0]
            if isinstance(instr, (list, tuple)):
                instr = instr[0] if instr else ""
            obs["instruction"] = str(instr)

        if "dataset_names" not in obs:
            obs["dataset_names"] = self.dataset_name

    def infer(self, obs: Dict) -> Dict:
        """Infer action from x2robot_client observation.

        Input (x2robot_client format):
            state: dict with follow1_pos(7D), follow2_pos(7D), head_pos, lift, ...
            views: dict with camera base64/numpy images
            instruction / prompt: task text

        Output (x2robot_client format):
            follow1_pos: np.float32 (T+1, 7), row 0 = current state, 3D euler
            follow2_pos: np.float32 (T+1, 7), row 0 = current state, 3D euler
            predict_action: raw model output
        """
        # print(obs.keys())
        # print(obs["state"].keys())
        # print(obs["views"].keys())
        # print(obs["prompt"])
        self._normalize_client_input(obs)

        state_dict = obs.get("state", {})
        if not isinstance(state_dict, dict):
            raise ValueError(
                "Expected obs['state'] to be a dict (x2robot_client format), "
                f"got {type(state_dict).__name__}"
            )

        rsad = self._build_robot_state(state_dict)
        agent_pos = rsad.get_agent_pos(self._obs_action_keys)
        if agent_pos.ndim == 3:
            agent_pos = agent_pos.squeeze(0)
        obs["state"] = agent_pos

        try:
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
            else:
                predicted_actions = (
                    outputs["predict_action"][:, :, : self.action_dim]
                    .detach()
                    .cpu()
                    .to(torch.float32)
                    .numpy()
                )

            model_output = self._postprocess_with_rsad(predicted_actions, rsad)

            return self._serialize_actions(
                model_output,
                interpolate_multiplier=obs.get("robot_action_interpolate_multiplier"),
                start_ratio=obs.get("robot_action_start_ratio"),
                end_ratio=obs.get("robot_action_end_ratio"),
            )
        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
