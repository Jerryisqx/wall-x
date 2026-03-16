import torch
import numpy as np

from scipy import stats
from scipy.signal import savgol_filter
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation
from x2robot_dataset.common.constants import ACTION_KEY_RANGES
from numba import jit, prange


@jit(nopython=True, parallel=True)
def euler_to_matrix_zyx_6d_nb(eulers):
    """
    Numba 版本：欧拉角 (N, 3) → 前两行展平 (N, 6)
    """
    N = eulers.shape[0]
    R6 = np.empty((N, 6), dtype=np.float64)
    for i in prange(N):
        roll = eulers[i, 0]
        pitch = eulers[i, 1]
        yaw = eulers[i, 2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        r00 = cy * cp
        r01 = cy * sp * sr - sy * cr
        r02 = cy * sp * cr + sy * sr

        r10 = sy * cp
        r11 = sy * sp * sr + cy * cr
        r12 = sy * sp * cr - cy * sr

        R6[i, 0] = r00
        R6[i, 1] = r01
        R6[i, 2] = r02
        R6[i, 3] = r10
        R6[i, 4] = r11
        R6[i, 5] = r12
    return R6


@jit(nopython=True, parallel=True)
def euler_to_matrix_zyx_batch_nb(eulers):
    N = eulers.shape[0]
    R = np.empty((N, 3, 3), dtype=np.float64)
    for i in prange(N):
        roll = eulers[i, 0]
        pitch = eulers[i, 1]
        yaw = eulers[i, 2]

        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)

        R[i, 0, 0] = cy * cp
        R[i, 0, 1] = cy * sp * sr - sy * cr
        R[i, 0, 2] = cy * sp * cr + sy * sr

        R[i, 1, 0] = sy * cp
        R[i, 1, 1] = sy * sp * sr + cy * cr
        R[i, 1, 2] = sy * sp * cr - cy * sr

        R[i, 2, 0] = -sp
        R[i, 2, 1] = cp * sr
        R[i, 2, 2] = cp * cr
    return R


def so3_to_euler_zyx_batch_nb(batch_so3):
    matrix = so3_to_matrix_batch_nb(batch_so3)
    eulers = matrix_to_euler_zyx_batch_nb(matrix)
    return canonicalize_euler_zyx_batch_nb(eulers)


@jit(nopython=True, parallel=True)
def matrix_to_euler_zyx_batch_nb(Rs):
    """
    R = Rz(yaw) * Ry(pitch) * Rx(roll)
    提取：
      pitch = asin(-R[2,0])
      roll  = atan2(R[2,1], R[2,2])
      yaw   = atan2(R[1,0], R[0,0])
    """
    N = Rs.shape[0]
    eulers = np.empty((N, 3), dtype=np.float64)
    for i in prange(N):
        r00 = Rs[i, 0, 0]
        # r01 = Rs[i, 0, 1]
        # r02 = Rs[i, 0, 2]
        r10 = Rs[i, 1, 0]
        # r11 = Rs[i, 1, 1]
        # r12 = Rs[i, 1, 2]
        r20 = Rs[i, 2, 0]
        r21 = Rs[i, 2, 1]
        r22 = Rs[i, 2, 2]

        # 数值稳定：夹到 [-1, 1]
        x = -r20
        if x > 1.0:
            x = 1.0
        elif x < -1.0:
            x = -1.0

        pitch = np.arcsin(x)
        roll = np.arctan2(r21, r22)
        yaw = np.arctan2(r10, r00)

        eulers[i, 0] = roll
        eulers[i, 1] = pitch
        eulers[i, 2] = yaw
    return eulers


@jit(nopython=True, parallel=True)
def so3_to_matrix_batch_nb(batch_so3):
    N = batch_so3.shape[0]
    R_all = np.empty((N, 3, 3), dtype=np.float64)
    eps = 1e-12
    for i in prange(N):
        r1x, r1y, r1z = batch_so3[i, 0], batch_so3[i, 1], batch_so3[i, 2]
        r2x, r2y, r2z = batch_so3[i, 3], batch_so3[i, 4], batch_so3[i, 5]

        # normalize r1
        n1 = np.sqrt(r1x * r1x + r1y * r1y + r1z * r1z) + eps
        r1x /= n1
        r1y /= n1
        r1z /= n1

        # orthogonalize r2 to r1, then normalize
        dot12 = r1x * r2x + r1y * r2y + r1z * r2z
        r2x -= dot12 * r1x
        r2y -= dot12 * r1y
        r2z -= dot12 * r1z
        n2 = np.sqrt(r2x * r2x + r2y * r2y + r2z * r2z) + eps
        r2x /= n2
        r2y /= n2
        r2z /= n2

        # r3 = r1 x r2
        r3x = r1y * r2z - r1z * r2y
        r3y = r1z * r2x - r1x * r2z
        r3z = r1x * r2y - r1y * r2x

        R_all[i, 0, 0] = r1x
        R_all[i, 0, 1] = r1y
        R_all[i, 0, 2] = r1z
        R_all[i, 1, 0] = r2x
        R_all[i, 1, 1] = r2y
        R_all[i, 1, 2] = r2z
        R_all[i, 2, 0] = r3x
        R_all[i, 2, 1] = r3y
        R_all[i, 2, 2] = r3z
    return R_all


@jit(nopython=True, parallel=True)
def compute_delta_from_state_and_abs_rot(rotations, state):
    if rotations.shape[-1] == 3:
        rotations_matrix = euler_to_matrix_zyx_batch_nb(rotations)
        out_is_euler = True
    elif rotations.shape[-1] == 6:
        rotations_matrix = so3_to_matrix_batch_nb(rotations)
        out_is_euler = False
    else:
        raise ValueError(
            f"Only support 3D euler angle or 6D rotation, but got {rotations.shape[-1]}D"
        )

    if state.shape[-1] == 3:
        state_matrix = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]
    elif state.shape[-1] == 6:
        state_matrix = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]
    else:
        raise ValueError(
            f"Only support 3D euler angle or 6D rotation, but got {state.shape[-1]}D"
        )

    ST = np.empty((3, 3), dtype=np.float64)
    ST[0, 0] = state_matrix[0, 0]
    ST[0, 1] = state_matrix[1, 0]
    ST[0, 2] = state_matrix[2, 0]
    ST[1, 0] = state_matrix[0, 1]
    ST[1, 1] = state_matrix[1, 1]
    ST[1, 2] = state_matrix[2, 1]
    ST[2, 0] = state_matrix[0, 2]
    ST[2, 1] = state_matrix[1, 2]
    ST[2, 2] = state_matrix[2, 2]
    N = rotations_matrix.shape[0]
    R_rel = np.empty((N, 3, 3), dtype=np.float64)
    for i in prange(N):
        A00 = rotations_matrix[i, 0, 0]
        A01 = rotations_matrix[i, 0, 1]
        A02 = rotations_matrix[i, 0, 2]
        A10 = rotations_matrix[i, 1, 0]
        A11 = rotations_matrix[i, 1, 1]
        A12 = rotations_matrix[i, 1, 2]
        A20 = rotations_matrix[i, 2, 0]
        A21 = rotations_matrix[i, 2, 1]
        A22 = rotations_matrix[i, 2, 2]

        R_rel[i, 0, 0] = A00 * ST[0, 0] + A01 * ST[1, 0] + A02 * ST[2, 0]
        R_rel[i, 0, 1] = A00 * ST[0, 1] + A01 * ST[1, 1] + A02 * ST[2, 1]
        R_rel[i, 0, 2] = A00 * ST[0, 2] + A01 * ST[1, 2] + A02 * ST[2, 2]
        R_rel[i, 1, 0] = A10 * ST[0, 0] + A11 * ST[1, 0] + A12 * ST[2, 0]
        R_rel[i, 1, 1] = A10 * ST[0, 1] + A11 * ST[1, 1] + A12 * ST[2, 1]
        R_rel[i, 1, 2] = A10 * ST[0, 2] + A11 * ST[1, 2] + A12 * ST[2, 2]
        R_rel[i, 2, 0] = A20 * ST[0, 0] + A21 * ST[1, 0] + A22 * ST[2, 0]
        R_rel[i, 2, 1] = A20 * ST[0, 1] + A21 * ST[1, 1] + A22 * ST[2, 1]
        R_rel[i, 2, 2] = A20 * ST[0, 2] + A21 * ST[1, 2] + A22 * ST[2, 2]

    if out_is_euler:
        d_euler = matrix_to_euler_zyx_batch_nb(R_rel)
        return canonicalize_euler_zyx_batch_nb(d_euler)
    else:
        out6 = np.empty((N, 6), dtype=np.float64)
        for i in prange(N):
            out6[i, 0] = R_rel[i, 0, 0]
            out6[i, 1] = R_rel[i, 0, 1]
            out6[i, 2] = R_rel[i, 0, 2]
            out6[i, 3] = R_rel[i, 1, 0]
            out6[i, 4] = R_rel[i, 1, 1]
            out6[i, 5] = R_rel[i, 1, 2]
        return out6


@jit(nopython=True, parallel=True)
def canonicalize_euler_zyx_batch_nb(rpy_batch):
    """
    批量 ZYX 欧拉角规范化（并行版）
    输入:  rpy_batch (N, 3)  [roll, pitch, yaw]（弧度）
    输出:  out       (N, 3)  约束到同一分支且每分量在 (-π, π]
    规则:
      1) 先逐分量 wrap 到 (-π, π]
      2) 若 p >  π/2:  p =  π - p; r += π; y += π
         若 p <= -π/2: p = -π - p; r += π; y += π
      3) 最后再逐分量 wrap 到 (-π, π]
    """
    N = rpy_batch.shape[0]
    out = np.empty_like(rpy_batch)
    two_pi = 2.0 * np.pi

    for i in prange(N):
        r = rpy_batch[i, 0]
        p = rpy_batch[i, 1]
        y = rpy_batch[i, 2]

        # 第一次 wrap
        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        # 分支规范
        if p > np.pi / 2.0:
            p = np.pi - p
            r = r + np.pi
            y = y + np.pi
        elif p <= -np.pi / 2.0:
            p = -np.pi - p
            r = r + np.pi
            y = y + np.pi

        # 最终 wrap
        r = (r + np.pi) % two_pi - np.pi
        p = (p + np.pi) % two_pi - np.pi
        y = (y + np.pi) % two_pi - np.pi

        out[i, 0] = r
        out[i, 1] = p
        out[i, 2] = y

    return out


@jit(nopython=True, parallel=True)
def compose_state_and_delta_to_abs_rpy(delta, state):
    """
    输入:
      delta: (N,3)  -> Δrpy(ZYX)   或 (N,6) -> Δ6D(前两行展平)
      state: (3,)   -> rpy(ZYX)    或 (6,)  -> 6D(前两行展平)
    输出:
      abs_rpy: (N,3) 绝对姿态的 rpy(ZYX, 弧度制)，已归一到 (-π, π]
    """
    if delta.shape[-1] == 3:
        R_delta = euler_to_matrix_zyx_batch_nb(delta)  # (N,3,3)
    elif delta.shape[-1] == 6:
        R_delta = so3_to_matrix_batch_nb(delta)  # (N,3,3)
    else:
        raise ValueError(f"delta last dim must be 3 or 6, got {delta.shape[-1]}")

    if state.shape[-1] == 3:
        R_state = euler_to_matrix_zyx_batch_nb(state[np.newaxis, :])[0]  # (3,3)
    elif state.shape[-1] == 6:
        R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]  # (3,3)
    else:
        raise ValueError(f"state last dim must be 3 or 6, got {state.shape[-1]}")

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    # 预先复制 R_state 以避免视图问题
    S00 = R_state[0, 0]
    S01 = R_state[0, 1]
    S02 = R_state[0, 2]
    S10 = R_state[1, 0]
    S11 = R_state[1, 1]
    S12 = R_state[1, 2]
    S20 = R_state[2, 0]
    S21 = R_state[2, 1]
    S22 = R_state[2, 2]

    for i in prange(N):
        A00 = R_delta[i, 0, 0]
        A01 = R_delta[i, 0, 1]
        A02 = R_delta[i, 0, 2]
        A10 = R_delta[i, 1, 0]
        A11 = R_delta[i, 1, 1]
        A12 = R_delta[i, 1, 2]
        A20 = R_delta[i, 2, 0]
        A21 = R_delta[i, 2, 1]
        A22 = R_delta[i, 2, 2]

        R_abs[i, 0, 0] = A00 * S00 + A01 * S10 + A02 * S20
        R_abs[i, 0, 1] = A00 * S01 + A01 * S11 + A02 * S21
        R_abs[i, 0, 2] = A00 * S02 + A01 * S12 + A02 * S22

        R_abs[i, 1, 0] = A10 * S00 + A11 * S10 + A12 * S20
        R_abs[i, 1, 1] = A10 * S01 + A11 * S11 + A12 * S21
        R_abs[i, 1, 2] = A10 * S02 + A11 * S12 + A12 * S22

        R_abs[i, 2, 0] = A20 * S00 + A21 * S10 + A22 * S20
        R_abs[i, 2, 1] = A20 * S01 + A21 * S11 + A22 * S21
        R_abs[i, 2, 2] = A20 * S02 + A21 * S12 + A22 * S22

    abs_rpy = matrix_to_euler_zyx_batch_nb(R_abs)
    abs_rpy = canonicalize_euler_zyx_batch_nb(abs_rpy)

    return abs_rpy


@jit(nopython=True, parallel=True)
def compose_state_and_delta_to_abs_6d(delta, state):
    """
    输入:
      delta: (N,6)  -> Δ6D(前两行展平) 6D旋转增量
      state: (6,)   -> 6D(前两行展平) 当前状态6D旋转
    输出:
      abs_6d: (N,6) 绝对姿态的 6D 表示(前两行展平)
    """
    R_delta = so3_to_matrix_batch_nb(delta)  # (N,3,3)
    R_state = so3_to_matrix_batch_nb(state[np.newaxis, :])[0]  # (3,3)

    N = R_delta.shape[0]
    R_abs = np.empty((N, 3, 3), dtype=np.float64)

    # 预先复制 R_state 以避免视图问题
    S00 = R_state[0, 0]
    S01 = R_state[0, 1]
    S02 = R_state[0, 2]
    S10 = R_state[1, 0]
    S11 = R_state[1, 1]
    S12 = R_state[1, 2]
    S20 = R_state[2, 0]
    S21 = R_state[2, 1]
    S22 = R_state[2, 2]

    for i in prange(N):
        A00 = R_delta[i, 0, 0]
        A01 = R_delta[i, 0, 1]
        A02 = R_delta[i, 0, 2]
        A10 = R_delta[i, 1, 0]
        A11 = R_delta[i, 1, 1]
        A12 = R_delta[i, 1, 2]
        A20 = R_delta[i, 2, 0]
        A21 = R_delta[i, 2, 1]
        A22 = R_delta[i, 2, 2]

        R_abs[i, 0, 0] = A00 * S00 + A01 * S10 + A02 * S20
        R_abs[i, 0, 1] = A00 * S01 + A01 * S11 + A02 * S21
        R_abs[i, 0, 2] = A00 * S02 + A01 * S12 + A02 * S22

        R_abs[i, 1, 0] = A10 * S00 + A11 * S10 + A12 * S20
        R_abs[i, 1, 1] = A10 * S01 + A11 * S11 + A12 * S21
        R_abs[i, 1, 2] = A10 * S02 + A11 * S12 + A12 * S22

        R_abs[i, 2, 0] = A20 * S00 + A21 * S10 + A22 * S20
        R_abs[i, 2, 1] = A20 * S01 + A21 * S11 + A22 * S21
        R_abs[i, 2, 2] = A20 * S02 + A21 * S12 + A22 * S22

    # 将旋转矩阵转换为6D表示(提取前两行并展平)
    abs_6d = np.empty((N, 6), dtype=np.float64)
    for i in prange(N):
        abs_6d[i, 0] = R_abs[i, 0, 0]
        abs_6d[i, 1] = R_abs[i, 0, 1]
        abs_6d[i, 2] = R_abs[i, 0, 2]
        abs_6d[i, 3] = R_abs[i, 1, 0]
        abs_6d[i, 4] = R_abs[i, 1, 1]
        abs_6d[i, 5] = R_abs[i, 1, 2]

    return abs_6d


@jit(nopython=True, parallel=True)
def normalize_angle_rad_batch_nb2(angles):
    """
    angles: (N, 2) -> (N, 2)，每个分量规范到 (-pi, pi]
    """
    N = angles.shape[0]
    out = np.empty_like(angles)
    two_pi = 2.0 * np.pi
    for i in prange(N):
        a0 = angles[i, 0]
        a1 = angles[i, 1]
        a0 = (a0 + np.pi) % two_pi - np.pi
        a1 = (a1 + np.pi) % two_pi - np.pi
        out[i, 0] = a0
        out[i, 1] = a1
    return out


@jit(nopython=True, parallel=True)
def compute_head_delta_from_state_and_abs_nb(abs_py, state_py):
    """
    abs_py:   (N, 2)
    state_py: (2,)
    return:   (N, 2)  delta = wrap(abs - state)
    """
    N = abs_py.shape[0]
    out = np.empty((N, 2), dtype=np.float64)

    if state_py.ndim == 1:
        s0, s1 = state_py[0], state_py[1]
        for i in prange(N):
            d0 = abs_py[i, 0] - s0
            d1 = abs_py[i, 1] - s1
            out[i, 0] = d0
            out[i, 1] = d1
    elif state_py.ndim == 2:
        if state_py.shape[0] != N and state_py.shape[0] != 1:
            raise ValueError("state_py must be shape (2,) or (N,2) or (1,2)")
        if state_py.shape[0] == 1:
            s0, s1 = state_py[0, 0], state_py[0, 1]
            for i in prange(N):
                d0 = abs_py[i, 0] - s0
                d1 = abs_py[i, 1] - s1
                out[i, 0] = d0
                out[i, 1] = d1
        else:  # (N,2)
            for i in prange(N):
                d0 = abs_py[i, 0] - state_py[i, 0]
                d1 = abs_py[i, 1] - state_py[i, 1]
                out[i, 0] = d0
                out[i, 1] = d1
    else:
        raise ValueError("state_py.ndim must be 1 or 2")

    return normalize_angle_rad_batch_nb2(out)


@jit(nopython=True, parallel=True)
def compose_state_and_delta_to_abs_head_nb(delta_py, state_py):
    """
    delta_py: (N, 2)  相对 pitch/yaw（弧度）
    state_py: (2,) 或 (N, 2)
    return:   (N, 2)  abs = wrap(state + delta)
    """
    N = delta_py.shape[0]
    out = np.empty((N, 2), dtype=np.float64)

    if state_py.ndim == 1:
        s0, s1 = state_py[0], state_py[1]
        for i in prange(N):
            a0 = delta_py[i, 0] + s0
            a1 = delta_py[i, 1] + s1
            out[i, 0] = a0
            out[i, 1] = a1
    elif state_py.ndim == 2:
        if state_py.shape[0] != N and state_py.shape[0] != 1:
            raise ValueError("state_py must be shape (2,) or (N,2) or (1,2)")
        if state_py.shape[0] == 1:
            s0, s1 = state_py[0, 0], state_py[0, 1]
            for i in prange(N):
                a0 = delta_py[i, 0] + s0
                a1 = delta_py[i, 1] + s1
                out[i, 0] = a0
                out[i, 1] = a1
        else:  # (N,2)
            for i in prange(N):
                a0 = delta_py[i, 0] + state_py[i, 0]
                a1 = delta_py[i, 1] + state_py[i, 1]
                out[i, 0] = a0
                out[i, 1] = a1
    else:
        raise ValueError("state_py.ndim must be 1 or 2")

    return normalize_angle_rad_batch_nb2(out)


def convert_euler_to_Lang(euler_angle):
    """
    Convert Euler angles to Lang angle.

    Input:
        euler_angle: pytorch tensor of shape [batch, 3] (Euler angles in radians)
    Output:
        lang_angle: numpy array of shape [batch] (Lang angles)
    """
    # Convert the PyTorch tensor to a NumPy array
    if isinstance(euler_angle, torch.Tensor):
        euler_angle_numpy = euler_angle.cpu().numpy()
    else:
        euler_angle_numpy = np.array(euler_angle)

    if len(euler_angle_numpy.shape) == 3:
        euler_angle_numpy = euler_angle_numpy.reshape(-1, 3)

    # Convert Euler angles to rotation matrix using scipy
    rotation_matrix = Rotation.from_euler(
        "xyz", euler_angle_numpy
    ).as_matrix()  # Shape: [batch, 3, 3]

    # Extract the relevant elements M00, M11, M22 for each rotation matrix
    M00 = rotation_matrix[:, 0, 0]  # First column, first row
    M11 = rotation_matrix[:, 1, 1]  # Second column, second row
    M22 = rotation_matrix[:, 2, 2]  # Third column, third row

    # Calculate the Lang angle
    lang_angle = np.arccos((M00 + M11 + M22 - 1) / 2)  # Shape: [batch]

    return lang_angle


def convert_6D_to_Lang(rotation_6d):
    """
    Convert 6D rotation to Lang angle. (Don't ask me why it is called Lang angle, quick coding)
    """
    if isinstance(rotation_6d, torch.Tensor):
        rotation_6d_numpy = rotation_6d.cpu().numpy()
    else:
        rotation_6d_numpy = np.array(rotation_6d)
    euler_angle = convert_6D_to_euler(rotation_6d_numpy)
    lang_angle = convert_euler_to_Lang(euler_angle)
    return lang_angle


def convert_euler_to_6D(euler_angle):
    """
    Convert euler angle to 6D rotation
    Input:
        euler_angle: numpy array of shape [low_dim_obs_horizon+horizon, 3] or [3]
    Output:
        rotation_6d: numpy array of shape [low_dim_obs_horizon+horizon, 6] or [6]
    """
    # TODO: find more elegent way
    # Convert euler angle to rotation matrix
    if len(euler_angle.shape) == 1:
        euler_angle = euler_angle.reshape(1, 3)
    rotation_matrix = Rotation.from_euler(
        "xyz", euler_angle
    ).as_matrix()  # [horizon, 3, 3]
    # Convert rotation matrix to 6D rotation(first 2 columns of rotation matrix)
    rotation_6d = np.zeros((euler_angle.shape[0], 6))
    rotation_6d[:, :3] = rotation_matrix[:, :, 0]
    rotation_6d[:, 3:] = rotation_matrix[:, :, 1]
    assert rotation_6d.shape == (
        euler_angle.shape[0],
        6,
    ), f"rotation_6d shape is not correct, you get {rotation_6d.shape}"
    return rotation_6d.squeeze() if len(euler_angle.shape) == 1 else rotation_6d


def convert_6D_to_euler(rotation_6d):
    """
    Convert 6D rotation to euler angle
    Input:
        rotation_6d: numpy array of shape [low_dim_obs_horizon+horizon, 6] or [6]
    Output:
        euler_angle: numpy array of shape [low_dim_obs_horizon+horizon, 3]
    """
    if rotation_6d.shape[0] == 6:
        rotation_6d = rotation_6d.reshape(1, 6)
    if len(rotation_6d.shape) == 3:
        rotation_6d = rotation_6d.reshape(-1, 6)
    # Convert 6D rotation to rotation matrix
    rotation_matrix = np.zeros((rotation_6d.shape[0], 3, 3))
    rotation_matrix[:, :, 0] = rotation_6d[:, :3]
    rotation_matrix[:, :, 1] = rotation_6d[:, 3:6]
    # get the third column of rotation matrix
    rotation_matrix[:, :, 2] = np.cross(
        rotation_matrix[:, :, 0], rotation_matrix[:, :, 1]
    )
    assert rotation_matrix.shape == (
        rotation_6d.shape[0],
        3,
        3,
    ), "rotation_matrix shape is not correct"
    # Convert rotation matrix to euler angle
    euler_angle = Rotation.from_matrix(rotation_matrix).as_euler("xyz")
    assert euler_angle.shape == (
        rotation_6d.shape[0],
        3,
    ), "euler_angle shape is not correct"
    return euler_angle


def convert_xyzrpy_to_matrix(position, orientation, data_config):
    """
    Convert xyzrpy to matrix
    Input:
        postion: np.array [3]
        orientation: np.array [3] for euler angle or [6] for 6D representation
        data_config: X2...
    Output:
        configuration matrix: SE(3) np.array [4,4]
    """
    configuration_matrix = np.eye(4)
    configuration_matrix[0:3, 3] = position
    if data_config.use_6D_rotation is True:
        orientation = convert_6D_to_euler(orientation)
    configuration_matrix[0:3, 0:3] = Rotation.from_euler("xyz", orientation).as_matrix()
    return configuration_matrix


def pose_to_transformation_matrix(
    pose, xyz_start_end_idx, rotation_start_end_idx, data_config
):
    """
    Input:
        pose: [action_dim], np.array
        xyz_start_end_idx: (,) tuple
        rotation_start_end_idx: (,) tuple
    Output:
        transformation_matrix: [4*4]
    """
    position = pose[xyz_start_end_idx[0] : xyz_start_end_idx[1]]
    rotation = pose[rotation_start_end_idx[0] : rotation_start_end_idx[1]]
    transformation_matrix = convert_xyzrpy_to_matrix(position, rotation, data_config)
    return transformation_matrix


def absolute_pose_to_relative_pose(
    absolute_pose,
    pose_key,
    data_config,
    data_chunk_config,
    shape_mappings=ACTION_KEY_RANGES,
    drop_first_frame=False,
):
    """
    Definition of Pose: Position(xyz) + Orientation(rpy)
    Input:
        absolute_pose: numpy array of shape [low_dim_obs_horizon+horizon, action_dim]
        pose_key: list of keys(strings) [num of action key]
        data_config: X2RDataProcessingConfig
        data_chunk_config: X2RDataChunkConfig
        drop_first_frame: whether drop the first frame for the returned value
        one_by_one_relative: whether to take the relative frame 相对于上一帧算relative action/相对于第一帧算relative action
    Output:
        relative_pose: numpy array of shape [low_dim_obs_horizon+horizon, action_dim]
    """
    ### TODO: Not elegant enough
    # Construct action_dim idx according to data_config and data_chunk_config
    action_dim = absolute_pose.shape[-1]
    horizon = absolute_pose.shape[0]
    relative_pose = np.copy(absolute_pose)  # copy grippers or other irrelevant data
    start_idx, end_idx = 0, 0
    action_dim_idx = {}
    for key in pose_key:
        end_idx += shape_mappings[key]["shape"]
        action_dim_idx[key] = (start_idx, end_idx)
        start_idx = end_idx
    assert (
        end_idx == action_dim
    ), f"action_dim_idx is not correct, end_idx: {end_idx} != action_dim: {action_dim}"

    # Get the first frame of each arm
    left_position_key = [
        key for key in pose_key if "cartesian" in key and "left" in key
    ]
    right_position_key = [
        key for key in pose_key if "cartesian" in key and "right" in key
    ]
    left_orientation_key = [
        key for key in pose_key if "rotation" in key and "left" in key
    ]
    right_orientation_key = [
        key for key in pose_key if "rotation" in key and "right" in key
    ]

    assert (
        len(left_position_key) == 1
    ), f"there should be 1 left cartesian key, you get {len(left_position_key)}: {left_position_key}"
    assert (
        len(right_position_key) == 1
    ), f"there should be 1 right cartesian key, you get {len(right_position_key)}: {right_position_key}"
    assert (
        len(left_orientation_key) == 1
    ), f"there should be 1 left orientation key, you get {len(left_orientation_key)}: {left_orientation_key}"
    assert (
        len(right_orientation_key) == 1
    ), f"there should be 1 right orientation key, you get {len(right_orientation_key)}: {right_orientation_key}"

    left_position_key, right_position_key = left_position_key[0], right_position_key[0]
    left_orientation_key, right_orientation_key = (
        left_orientation_key[0],
        right_orientation_key[0],
    )

    left_position_idx = action_dim_idx[left_position_key]
    left_rotation_idx = action_dim_idx[left_orientation_key]
    right_position_idx = action_dim_idx[right_position_key]
    right_rotation_idx = action_dim_idx[right_orientation_key]

    ref_pose_left = pose_to_transformation_matrix(
        absolute_pose[0], left_position_idx, left_rotation_idx, data_config
    )
    ref_pose_left_inv = np.linalg.inv(ref_pose_left)
    ref_pose_right = pose_to_transformation_matrix(
        absolute_pose[0], right_position_idx, right_rotation_idx, data_config
    )
    ref_pose_right_inv = np.linalg.inv(ref_pose_right)

    relative_pose[0] = absolute_pose[0]
    for i in range(1, horizon):
        curr_pose_left = pose_to_transformation_matrix(
            absolute_pose[i], left_position_idx, left_rotation_idx, data_config
        )
        curr_pose_right = pose_to_transformation_matrix(
            absolute_pose[i], right_position_idx, right_rotation_idx, data_config
        )

        # compute relative matrix
        relative_pose_left = ref_pose_left_inv @ curr_pose_left
        relative_pose_right = ref_pose_right_inv @ curr_pose_right

        relative_pose[i, left_position_idx[0] : left_position_idx[1]] = (
            relative_pose_left[:3, 3]
        )
        relative_pose[i, right_position_idx[0] : right_position_idx[1]] = (
            relative_pose_right[:3, 3]
        )

        relative_orientation_left = Rotation.from_matrix(
            relative_pose_left[:3, :3]
        ).as_euler("xyz")
        relative_orientation_right = Rotation.from_matrix(
            relative_pose_right[:3, :3]
        ).as_euler("xyz")
        if data_config.use_6D_rotation is True:
            relative_orientation_left = convert_euler_to_6D(relative_orientation_left)
            relative_orientation_right = convert_euler_to_6D(relative_orientation_right)
        relative_pose[i, left_rotation_idx[0] : left_rotation_idx[1]] = (
            relative_orientation_left
        )
        relative_pose[i, right_rotation_idx[0] : right_rotation_idx[1]] = (
            relative_orientation_right
        )

        if data_config.one_by_one_relative is True:
            ref_pose_left = curr_pose_left
            ref_pose_right = curr_pose_right
            ref_pose_left_inv = np.linalg.inv(ref_pose_left)
            ref_pose_right_inv = np.linalg.inv(ref_pose_right)

    if drop_first_frame:
        return relative_pose[1:]
    else:
        return relative_pose


def relative_pose_to_absolute_pose(
    ref_pose,
    relative_pose,
    pose_key,
    data_config,
    data_chunk_config,
    shape_mappings=ACTION_KEY_RANGES,
    drop_first_frame=False,
):
    """
    Convert Delta EE to Real EE pose
    Input:
        ref_pose: np.array [1, action_dim]
        relative_pose: np.array [horizon, action_dim]
        pose_key: list for keys
        ... # TODO:以后再补，太困了orz zzZ
    Output:
        absolute_pose: np.array [horizon, action_dim]
    """
    if len(ref_pose.shape) == 1:
        ref_pose = ref_pose.reshape(1, ref_pose.shape[0])
    if len(relative_pose.shape) == 1:
        relative_pose = relative_pose.reshape(1, relative_pose.shape[0])

    ### TODO: Not elegant enough, too much redundent code
    horizon, action_dim = relative_pose.shape[0], relative_pose.shape[1]
    # assert ref_pose.shape == # TODO: Add shape check
    absolute_pose = np.zeros((horizon + 1, action_dim))
    absolute_pose[0] = ref_pose[0]
    absolute_pose[1:] = relative_pose  # copy grippers or other irrelevant data
    start_idx, end_idx = 0, 0
    action_dim_idx = {}
    for key in pose_key:
        end_idx += shape_mappings[key]["shape"]
        action_dim_idx[key] = (start_idx, end_idx)
        start_idx = end_idx
    assert (
        end_idx == action_dim
    ), f"action_dim_idx is not correct, end_idx: {end_idx} != action_dim: {action_dim}"

    # get reference matrix
    position_key = [key for key in pose_key if "cartesian" in key]
    orientation_key = [key for key in pose_key if "rotation" in key]
    assert (
        len(position_key) == 2
    ), f"there should be 2 cartesian keys, you get {len(position_key)}: {position_key}"
    assert (
        len(orientation_key) == 2
    ), f"there should be 2 orientation keys, you get {len(orientation_key)}: {orientation_key}"
    ref_position_left = ref_pose[
        0, action_dim_idx[position_key[0]][0] : action_dim_idx[position_key[0]][1]
    ]
    ref_rotation_left = ref_pose[
        0, action_dim_idx[orientation_key[0]][0] : action_dim_idx[orientation_key[0]][1]
    ]
    ref_pose_left = convert_xyzrpy_to_matrix(
        ref_position_left, ref_rotation_left, data_config
    )
    ref_position_right = ref_pose[
        0, action_dim_idx[position_key[1]][0] : action_dim_idx[position_key[1]][1]
    ]
    ref_rotation_right = ref_pose[
        0, action_dim_idx[orientation_key[1]][0] : action_dim_idx[orientation_key[1]][1]
    ]
    ref_pose_right = convert_xyzrpy_to_matrix(
        ref_position_right, ref_rotation_right, data_config
    )

    for i in range(0, horizon):
        relative_position_left = relative_pose[
            i, action_dim_idx[position_key[0]][0] : action_dim_idx[position_key[0]][1]
        ]
        relative_rotation_left = relative_pose[
            i,
            action_dim_idx[orientation_key[0]][0] : action_dim_idx[orientation_key[0]][
                1
            ],
        ]
        relative_pose_left = convert_xyzrpy_to_matrix(
            relative_position_left, relative_rotation_left, data_config
        )
        absolute_pose_matrix_left = ref_pose_left @ relative_pose_left
        absolute_pose[
            i + 1,
            action_dim_idx[position_key[0]][0] : action_dim_idx[position_key[0]][1],
        ] = absolute_pose_matrix_left[:3, 3]
        orientation_left = Rotation.from_matrix(
            absolute_pose_matrix_left[:3, :3]
        ).as_euler("xyz")
        if data_config.use_6D_rotation is True:
            orientation_left = convert_euler_to_6D(orientation_left)
        absolute_pose[
            i + 1,
            action_dim_idx[orientation_key[0]][0] : action_dim_idx[orientation_key[0]][
                1
            ],
        ] = orientation_left

        relative_position_right = relative_pose[
            i, action_dim_idx[position_key[1]][0] : action_dim_idx[position_key[1]][1]
        ]
        relative_rotation_right = relative_pose[
            i,
            action_dim_idx[orientation_key[1]][0] : action_dim_idx[orientation_key[1]][
                1
            ],
        ]
        relative_pose_right = convert_xyzrpy_to_matrix(
            relative_position_right, relative_rotation_right, data_config
        )
        absolute_pose_matrix_right = ref_pose_right @ relative_pose_right
        absolute_pose[
            i + 1,
            action_dim_idx[position_key[1]][0] : action_dim_idx[position_key[1]][1],
        ] = absolute_pose_matrix_right[:3, 3]
        orientation_right = Rotation.from_matrix(
            absolute_pose_matrix_right[:3, :3]
        ).as_euler("xyz")
        if data_config.use_6D_rotation is True:
            orientation_right = convert_euler_to_6D(orientation_right)
        absolute_pose[
            i + 1,
            action_dim_idx[orientation_key[1]][0] : action_dim_idx[orientation_key[1]][
                1
            ],
        ] = orientation_right

        if data_config.one_by_one_relative is True:
            ref_pose_left = absolute_pose_matrix_left
            ref_pose_right = absolute_pose_matrix_right

    if drop_first_frame:
        return absolute_pose[1:]
    else:
        return absolute_pose


def actions_to_relative(
    actions, add_noise=False, noise_scale=[0.05, 0.05, 0.05, 0.05, 0.05, 0.05]
):
    """Convert absolute actions to relative actions

    Args:
        actions: numpy array of shape [horizon, action_dim]
                action_dim=14, [left_arm(7), right_arm(7)]
                Each arm: [x,y,z,roll,pitch,yaw,gripper]
        add_noise: bool, whether to add noise to the first frame
        noise_scale: list of 6 numbers, scale of noise for [x,y,z,roll,pitch,yaw]

    Returns:
        relative_actions: numpy array of shape [horizon, action_dim]
    """
    horizon, _ = actions.shape
    relative_actions = np.zeros_like(actions)

    # 处理左臂和右臂
    for arm_idx in range(2):
        start_idx = arm_idx * 7

        # 获取第一帧的变换矩阵
        ref_pos = actions[0, start_idx : start_idx + 3]
        ref_rot = Rotation.from_euler("xyz", actions[0, start_idx + 3 : start_idx + 6])
        ref_matrix = np.eye(4)
        ref_matrix[:3, :3] = ref_rot.as_matrix()
        ref_matrix[:3, 3] = ref_pos

        # 如果需要加噪声，给第一帧加噪声
        if add_noise:
            noise = np.random.normal(scale=noise_scale, size=6)
            noisy_pos = ref_pos + noise[:3]
            noisy_rot = Rotation.from_euler(
                "xyz", actions[0, start_idx + 3 : start_idx + 6] + noise[3:]
            )
            noisy_matrix = np.eye(4)
            noisy_matrix[:3, :3] = noisy_rot.as_matrix()
            noisy_matrix[:3, 3] = noisy_pos
            ref_matrix_inv = np.linalg.inv(noisy_matrix)
        else:
            ref_matrix_inv = np.linalg.inv(ref_matrix)

        # 计算所有帧的相对变换（包括第一帧）
        for i in range(horizon):
            # 当前帧的变换矩阵
            curr_pos = actions[i, start_idx : start_idx + 3]
            curr_rot = Rotation.from_euler(
                "xyz", actions[i, start_idx + 3 : start_idx + 6]
            )
            curr_matrix = np.eye(4)
            curr_matrix[:3, :3] = curr_rot.as_matrix()
            curr_matrix[:3, 3] = curr_pos

            # 计算相对变换
            relative_matrix = ref_matrix_inv @ curr_matrix

            # 提取相对位置和旋转
            relative_actions[i, start_idx : start_idx + 3] = relative_matrix[:3, 3]
            relative_actions[i, start_idx + 3 : start_idx + 6] = Rotation.from_matrix(
                relative_matrix[:3, :3]
            ).as_euler("xyz")

            # Gripper保持不变
            relative_actions[i, start_idx + 6] = actions[i, start_idx + 6]

    return relative_actions


def relative_to_actions(relative_actions, start_pose, one_by_one_relative=False):
    """Convert relative actions back to absolute actions

    Args:
        relative_actions: numpy array of shape [horizon-1, action_dim]
                         从第二帧开始的相对位姿序列
                         action_dim=14, [left_arm(7), right_arm(7)]
                         Each arm: [x,y,z,roll,pitch,yaw,gripper]
        start_pose: numpy array of shape [action_dim]
                   第一帧的绝对位姿

    Returns:
        actions: numpy array of shape [horizon, action_dim]
                包含start_pose和转换后的绝对位姿序列
    """
    horizon = relative_actions.shape[0] + 1  # 加1是因为relative_actions不包含第一帧
    actions = np.zeros((horizon, relative_actions.shape[1]))

    # 设置第一帧为给定的start_pose
    actions[0] = start_pose

    # 处理左臂和右臂
    for arm_idx in range(2):
        start_idx = arm_idx * 7

        # 使用start_pose创建参考变换矩阵
        ref_pos = start_pose[start_idx : start_idx + 3]
        ref_rot = Rotation.from_euler("xyz", start_pose[start_idx + 3 : start_idx + 6])
        ref_matrix = np.eye(4)
        ref_matrix[:3, :3] = ref_rot.as_matrix()
        ref_matrix[:3, 3] = ref_pos

        # 从第二帧开始计算绝对位姿
        for i in range(horizon - 1):  # horizon-1是relative_actions的长度
            # 当前相对位姿的变换矩阵
            relative_pos = relative_actions[i, start_idx : start_idx + 3]
            relative_rot = Rotation.from_euler(
                "xyz", relative_actions[i, start_idx + 3 : start_idx + 6]
            )
            relative_matrix = np.eye(4)
            relative_matrix[:3, :3] = relative_rot.as_matrix()
            relative_matrix[:3, 3] = relative_pos

            # 计算绝对变换
            abs_matrix = ref_matrix @ relative_matrix

            # 提取绝对位置和旋转，存储在第i+1帧
            actions[i + 1, start_idx : start_idx + 3] = abs_matrix[:3, 3]
            actions[i + 1, start_idx + 3 : start_idx + 6] = Rotation.from_matrix(
                abs_matrix[:3, :3]
            ).as_euler("xyz")

            # Gripper保持不变
            actions[i + 1, start_idx + 6] = relative_actions[i, start_idx + 6]

            # 当前帧的相对动作是相对于上一帧
            if one_by_one_relative:
                ref_matrix = abs_matrix

    return actions[1:]  # 不包含start_pose


def remove_outliers(data, threshold=3):
    """
    使用Z-score方法移除异常值。

    参数:
        data: 输入数据数组
        threshold: Z-score阈值，默认为3

    返回:
        过滤后的数据
    """
    # 如果数据点太少或全为相同值，直接返回原始数据
    if len(data) < 3 or np.all(data == data[0]):
        return data.copy()

    # 计算标准差，避免灾难性抵消
    std = np.std(data)
    if std < 1e-10:  # 如果标准差接近0，直接返回原始数据
        return data.copy()

    # 计算Z-score
    try:
        z_scores = np.abs(stats.zscore(data))
    except (ValueError, FloatingPointError, TypeError):
        # 如果无法计算 Z-score（例如所有值相同），直接返回原始数据
        return data.copy()

    filtered_data = data.copy()

    # 找出异常值
    mask = z_scores > threshold

    # 如果异常值太多，可能是数据本身就有较大波动，保持原样
    if np.sum(mask) > len(data) * 0.4:  # 如果超过40%的数据被标记为异常
        return data.copy()

    # 标记异常值为NaN
    filtered_data[mask] = np.nan

    # 检查是否所有值都变成了NaN
    if np.all(np.isnan(filtered_data)):
        return data.copy()

    # 用插值填充NaN值
    nan_mask = np.isnan(filtered_data)

    # 确保有非NaN值可以用于插值
    if np.any(~nan_mask):
        filtered_data[nan_mask] = np.interp(
            np.flatnonzero(nan_mask),
            np.flatnonzero(~nan_mask),
            filtered_data[~nan_mask],
        )

    return filtered_data


def remove_jumps(data, threshold=1.0):
    """
    检测并修正数据中的突然跳变。

    参数:
        data: 输入数据数组
        threshold: 跳变检测阈值，默认为1.0

    返回:
        修正后的数据
    """
    # 如果数据点太少，无法有效检测跳变，直接返回原始数据
    if len(data) < 3:
        return data.copy()

    result = data.copy()

    # 计算相邻点之间的差值
    try:
        diffs = np.abs(np.diff(result))
    except (ValueError, FloatingPointError, TypeError):
        # 如果无法计算差值，直接返回原始数据
        return data.copy()

    # 找出大于阈值的跳变点
    jump_indices = np.where(diffs > threshold)[0]

    # 如果跳变点太多，说明数据本身波动较大，保持原样
    if len(jump_indices) > len(data) * 0.3:  # 如果超过30%的点被标记为跳变
        return data.copy()

    # 处理每个跳变点
    for idx in jump_indices:
        # 用前后点的平均值替换跳变
        if idx > 0 and idx < len(result) - 1:
            # 取跳变前后的平均值
            result[idx + 1] = (result[idx] + result[idx + 2]) / 2
        elif idx == len(result) - 2:  # 如果是倒数第二个点
            result[idx + 1] = result[idx]  # 用前一个点的值替换

    return result


def smooth_data(
    data, window_length=None, polyorder=3, iterations=1, strong_smooth=False
):
    """
    使用Savitzky-Golay滤波器平滑数据。

    参数:
        data: 输入数据数组
        window_length: 窗口长度，如果未指定则自动计算
        polyorder: 多项式阶数，默认为3
        iterations: 平滑迭代次数，默认为1
        strong_smooth: 是否使用强平滑模式，默认为False

    返回:
        平滑后的数据
    """
    # 确保数据至少有3个点
    if len(data) < 3:
        return data.copy()

    # 计算合适的窗口长度
    if window_length is None:
        if strong_smooth:
            # 强平滑模式下使用更大的窗口
            window_length = min(51, len(data) - 1)
        else:
            window_length = min(21, len(data) - 1)

    # 确保窗口长度为奇数且小于等于数据长度
    window_length = min(window_length, len(data) - 1)
    if window_length % 2 == 0:  # 确保窗口长度为奇数
        window_length -= 1

    # 确保窗口长度至少为3
    window_length = max(3, window_length)

    # 如果数据长度小于窗口长度，使用高斯滤波
    if window_length >= len(data):
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)

    # 在强平滑模式下使用更低的多项式阶数
    if strong_smooth:
        polyorder = min(2, polyorder)

    # 确保多项式阶数小于窗口长度
    polyorder = min(polyorder, window_length - 1)

    smooth_data_result = data.copy()

    try:
        # 应用多次平滑
        for _ in range(iterations):
            smooth_data_result = savgol_filter(
                smooth_data_result, window_length, polyorder
            )

        # 如果需要强平滑，再应用高斯滤波
        if strong_smooth:
            smooth_data_result = gaussian_filter1d(smooth_data_result, sigma=2.0)

        return smooth_data_result

    except Exception as e:
        # 如果savgol_filter失败，使用高斯滤波作为备选
        print(f"Savgol filter failed: {e}, using Gaussian filter instead")
        sigma = 3.0 if strong_smooth else 1.0
        return gaussian_filter1d(data, sigma=sigma)


def process_car_pose_to_base_velocity(
    car_pose,
    outlier_threshold=3,
    jump_threshold=1.0,
    smooth_iterations=3,
    strong_smooth=True,
):
    """
    🎯 处理car_pose数据，转换为本体坐标系base_velocity_decomposed，与batch_process_json_data.py完全一致。
    包含完整的异常值处理、角度unwrap、过滤和平滑处理，以及本体坐标系速度计算。

    参数:
        car_pose: 输入的car_pose数据，shape: (n, 3) [x, y, angle]
        outlier_threshold: 异常值检测阈值，默认为3
        jump_threshold: 跳变检测阈值，默认为1.0
        smooth_iterations: 平滑迭代次数，默认为3
        strong_smooth: 是否使用强平滑模式，默认为True

    返回:
        dict: 包含处理后的数据
            - 'base_velocity_decomposed': shape (n, 3) [vx_body, vy_body, vyaw] (本体坐标系)
            - 'valid': bool, 是否有效（通过速度范围检查）
    """
    # 定义速度限制（与data_analysis_filter.py中完全一致）
    velocity_limits = {
        "vx": {"min": -0.5, "max": 0.5},
        "vy": {"min": -0.5, "max": 0.5},
        "vyaw": {"min": -1.6, "max": 1.6},
    }

    # 处理空数据或单点数据
    if len(car_pose) == 0:
        return {"base_velocity_decomposed": np.zeros((0, 3)), "valid": False}

    if len(car_pose) == 1:
        return {
            "base_velocity_decomposed": np.zeros((1, 3)),
            "valid": True,  # 单点数据认为是有效的
        }

    # 🎯 步骤1: 提取位置和角度数据，并进行角度展开
    x_values = car_pose[:, 0].copy()
    y_values = car_pose[:, 1].copy()
    angle_values = car_pose[:, 2].copy()

    # 🎯 角度展开处理，避免跳变（移到函数内部）
    angle_values_unwrapped = np.unwrap(angle_values)

    # 🎯 步骤2-4: 异常值处理、跳变修正、平滑处理
    # 异常值处理
    x_filtered = remove_outliers(x_values, outlier_threshold)
    y_filtered = remove_outliers(y_values, outlier_threshold)
    angle_filtered = remove_outliers(angle_values_unwrapped, outlier_threshold)

    # 跳变修正
    x_filtered = remove_jumps(x_filtered, jump_threshold)
    y_filtered = remove_jumps(y_filtered, jump_threshold)
    angle_filtered = remove_jumps(angle_filtered, jump_threshold)

    # 平滑处理
    window_length = min(51 if strong_smooth else 21, len(x_filtered) - 1)
    if window_length % 2 == 0:
        window_length -= 1
    window_length = max(3, window_length)

    x_smooth = smooth_data(
        x_filtered,
        window_length,
        polyorder=2 if strong_smooth else 3,
        iterations=smooth_iterations,
        strong_smooth=strong_smooth,
    )
    y_smooth = smooth_data(
        y_filtered,
        window_length,
        polyorder=2 if strong_smooth else 3,
        iterations=smooth_iterations,
        strong_smooth=strong_smooth,
    )
    angle_smooth = smooth_data(
        angle_filtered,
        window_length,
        polyorder=2 if strong_smooth else 3,
        iterations=smooth_iterations,
        strong_smooth=strong_smooth,
    )

    # 🎯 步骤5: 使用与data_processor.py完全一致的本体坐标系速度计算方法
    dt = 1 / 20  # 20Hz采样频率

    # 计算全局位移
    x_diff = np.diff(x_smooth)
    y_diff = np.diff(y_smooth)
    angle_diff = np.diff(angle_smooth)

    # 获取当前帧的角度（用于坐标变换）
    current_theta = angle_smooth[:-1]  # shape: (n-1,)

    # 🎯 坐标变换：从全局坐标系到本体坐标系（与data_processor.py一致）
    cos_theta = np.cos(current_theta)
    sin_theta = np.sin(current_theta)

    # 🎯 计算本体坐标系下的速度
    vx_body = (x_diff * cos_theta + y_diff * sin_theta) / dt  # 前进速度（本体坐标系）
    vy_body = (-x_diff * sin_theta + y_diff * cos_theta) / dt  # 左侧速度（本体坐标系）
    vyaw = angle_diff / dt  # 角速度

    # 在开头添加零速度以匹配原始数据长度
    vx_array = np.concatenate([[0], vx_body])
    vy_array = np.concatenate([[0], vy_body])
    vyaw_array = np.concatenate([[0], vyaw])

    base_velocity_decomposed = np.stack([vx_array, vy_array, vyaw_array], axis=1)

    # 🎯 步骤6: 速度范围检查（与data_analysis_filter.py一致）
    valid = True

    if (
        abs(x_values).max() > 6
        or abs(y_values).max() > 6
        or abs(angle_values).max() > 6
    ):
        valid = False

    # 检查每个速度分量是否在允许范围内
    if valid:
        for vx_val in vx_body:
            if (
                vx_val < velocity_limits["vx"]["min"]
                or vx_val > velocity_limits["vx"]["max"]
            ):
                valid = False
                break

    if valid:  # 只有vx通过检查才继续检查vy
        for vy_val in vy_body:
            if (
                vy_val < velocity_limits["vy"]["min"]
                or vy_val > velocity_limits["vy"]["max"]
            ):
                valid = False
                break

    if valid:  # 只有vx和vy都通过检查才继续检查vyaw
        for vyaw_val in vyaw:
            if (
                vyaw_val < velocity_limits["vyaw"]["min"]
                or vyaw_val > velocity_limits["vyaw"]["max"]
            ):
                valid = False
                break

    return {"base_velocity_decomposed": base_velocity_decomposed, "valid": valid}
