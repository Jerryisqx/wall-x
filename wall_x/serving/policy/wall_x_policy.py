import logging
import re
from typing import Dict, Any, List
import torch
import copy
import numpy as np
from wall_x.serving.websocket_policy_server import BasePolicy
from wall_x.model.qwen2_5_based.modeling_qwen2_5_vl_act import Qwen2_5_VLMoEForAction
from wall_x.serving.policy.utils import prepare_batch
from wall_x.model.model_utils import load_wallx_processors, register_normalizers
from wall_x.data.utils import maybe_expand_rotation_to_6d, infer_present_keys, pad_tensor_with_nan
from wall_x.infer.base_dataclass import RobotStateActionData

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

        self.fixed_action_dim = action_dim
        self.action_dim = action_dim
        self.agent_pos_dim = action_dim
        self.pred_horizon = pred_horizon
        self.device = device
        self.predict_mode = predict_mode
        self.default_prompt = default_prompt
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

        print("predict_mode", predict_mode)
        print("camera_key", camera_key)
        print("obs_action_keys", self._obs_action_keys)
        print("predict_action_keys", self._predict_action_keys)

        logger.info("Loading processor and tokenizer...")
        processors_dict = load_wallx_processors(train_config)
        self.processor = processors_dict["processor"]

        self.action_buffer = []
        self.buffer_index = 0

        logger.info(
            f"Model loaded. Device: {device}, Action dim: {action_dim}, Horizon: {pred_horizon}"
        )
        if "x2_normal" in self.normalizer_propri.min:
            print(f"normalizer_propri x2_normal dim: {self.normalizer_propri.min['x2_normal'].shape}")
        if "x2_normal" in self.normalizer_action.min:
            print(f"normalizer_action x2_normal dim: {self.normalizer_action.min['x2_normal'].shape}")

    @property
    def metadata(self) -> Dict[str, Any]:
        return {
            "action_dim": self.action_dim,
            "pred_horizon": self.pred_horizon,
            "device": self.device,
            "predict_mode": self.predict_mode,
        }

    def reset(self) -> None:
        self.action_buffer = []
        self.buffer_index = 0

    def _build_robot_state(self, state_dict) -> RobotStateActionData:
        """Build RobotStateActionData from a client state dict.

        Parses keys like follow1_pos, follow2_pos using ROBOT_ACTION_KEY_MAPPING,
        exactly like wall-x's robot preprocessor.
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

    def _prepare_state_for_model(self, obs):
        """Prepare state tensor for model input.

        Handles 3D→6D rotation expansion and NaN padding for missing keys.
        Works with both structured dict state and flat array state.
        """
        state = obs.get("state")
        if state is None:
            return

        if isinstance(state, np.ndarray):
            state = torch.from_numpy(state).float()
        elif not isinstance(state, torch.Tensor):
            state = torch.tensor(state, dtype=torch.float32)

        if self._obs_action_keys and self._agent_pos_config:
            present_keys = infer_present_keys(state.shape[-1], self._obs_action_keys, self._agent_pos_config)
            state = maybe_expand_rotation_to_6d(state, present_keys, self._agent_pos_config)
            target_dim = sum(self._agent_pos_config[k] for k in self._obs_action_keys)
            state = pad_tensor_with_nan(state, target_dim)

        obs["state"] = state

    def _postprocess_with_rsad(self, predicted_actions, rsad):
        """Use RobotStateActionData to convert model output to absolute 3D euler.

        Saves predicted action into rsad, then reads back absolute values
        via ComputedDict auto-conversion (relative→abs, 6D→3D).
        """
        rsad.save_action_data(predicted_actions, self._predict_action_keys)

        result = {"predict_action": predicted_actions}

        for side in ("left", "right"):
            pos = rsad.data.get(f"action_{side}_ee_cartesian_pos")
            rot = rsad.data.get(f"action_{side}_ee_rotation")
            grip = rsad.data.get(f"action_{side}_gripper")
            parts = [p for p in (pos, rot, grip) if p is not None]
            if parts:
                arm_key = "follow1_pos" if side == "left" else "follow2_pos"
                arm_action = np.concatenate(parts, axis=1)
                state_pos = rsad.data.get(f"state_{side}_ee_cartesian_pos")
                state_rot = rsad.data.get(f"state_{side}_ee_rotation")
                state_grip = rsad.data.get(f"state_{side}_gripper")
                state_parts = [p for p in (state_pos, state_rot, state_grip) if p is not None]
                if state_parts:
                    state_row = np.concatenate(state_parts, axis=1)
                    arm_action = np.concatenate([state_row, arm_action], axis=0)
                result[arm_key] = arm_action.tolist()

        return result

    def infer(self, obs: Dict) -> Dict:
        """Infer action from observation.

        Supports two state input modes:
        1. Structured dict: obs["state"] = {"follow1_pos": [...], "follow2_pos": [...], ...}
           → Uses RobotStateActionData for full conversion (6D, relative→abs)
        2. Flat array: obs["state"] = [x, y, z, ...]
           → Direct flat tensor path with rotation expansion + NaN padding

        Output always includes "predict_action" (raw model output).
        In structured mode, also includes "follow1_pos"/"follow2_pos" (absolute 3D euler).
        """
        state_input = obs.get("state")
        rsad = None

        if isinstance(state_input, dict):
            rsad = self._build_robot_state(state_input)
            agent_pos = rsad.get_agent_pos(self._obs_action_keys)
            if agent_pos.ndim == 3:
                agent_pos = agent_pos.squeeze(0)
            obs["state"] = agent_pos
        else:
            self._prepare_state_for_model(obs)

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

            if rsad is not None:
                return self._postprocess_with_rsad(predicted_actions, rsad)

            return {"predict_action": predicted_actions}

        except Exception as e:
            logger.error(f"Error during inference: {e}")
            raise
