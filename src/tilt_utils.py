import numpy as np
import pandas as pd


def smooth_1d(x: np.ndarray, win: int = 11) -> np.ndarray:
    if win <= 1 or len(x) < 3:
        return x
    if win % 2 == 0:
        win += 1
    win = min(win, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if win < 3:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def estimate_dt(num_samples: int, fps: float, num_frames: int):
    if num_samples <= 1 or fps <= 0:
        return 1.0 / 20.0
    total_time = (num_frames - 1) / fps
    return total_time / max(num_samples - 1, 1)


def compute_tilt(df: pd.DataFrame, dt: float, alpha=0.98, smooth_win=9):
    required = ["accl_x", "accl_y", "accl_z", "gyro_x", "gyro_y", "gyro_z"]

    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing column: {c}")

    ax = df["accl_x"].values
    ay = df["accl_y"].values
    az = df["accl_z"].values

    gx = df["gyro_x"].values
    gy = df["gyro_y"].values

    # Accelerometer-based tilt
    roll_acc = np.degrees(np.arctan2(ay, az))
    pitch_acc = np.degrees(np.arctan2(-ax, np.sqrt(ay**2 + az**2) + 1e-6))

    pitch = np.zeros(len(df))
    roll = np.zeros(len(df))

    for i in range(1, len(df)):
        roll_gyro = roll[i - 1] + np.degrees(gx[i] * dt)
        pitch_gyro = pitch[i - 1] + np.degrees(gy[i] * dt)

        roll[i] = alpha * roll_gyro + (1 - alpha) * roll_acc[i]
        pitch[i] = alpha * pitch_gyro + (1 - alpha) * pitch_acc[i]

    pitch = smooth_1d(pitch, smooth_win)
    roll = smooth_1d(roll, smooth_win)

    return pitch, roll