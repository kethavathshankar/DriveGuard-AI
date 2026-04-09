import os
import pandas as pd
import numpy as np


def load_master_timeline(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)

    required_cols = ["front_frame_id", "front_time_sec", "speed_kmph"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {csv_path}")

    df = df.sort_values("front_frame_id").reset_index(drop=True)
    return df


def load_sign_events(sign_csv_path: str | None = None) -> dict:
    """
    Expected stable sign event CSV columns:
        frame
        speed_limit

    Optional extra columns allowed:
        time_sec
        ocr_confidence

    Returns:
        sign_map = {frame: speed_limit}
    """
    if sign_csv_path is None or not os.path.exists(sign_csv_path):
        return {}

    df = pd.read_csv(sign_csv_path)

    required_cols = ["frame", "speed_limit"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Required column '{col}' not found in {sign_csv_path}")

    df = df.sort_values("frame").reset_index(drop=True)

    sign_map = {}
    for _, row in df.iterrows():
        frame_id = int(row["frame"])
        sign_val = row["speed_limit"]

        if pd.isna(sign_val):
            continue

        try:
            sign_val = float(sign_val)
        except Exception:
            continue

        sign_map[frame_id] = sign_val

    return sign_map


def compute_incremental_distance_m(df: pd.DataFrame) -> np.ndarray:
    """
    Compute cumulative traveled distance in meters from frame-wise speed_kmph and time.

    distance_step = speed_mps * dt
    cumulative_distance = sum(distance_step)
    """
    times = df["front_time_sec"].to_numpy(dtype=float)
    speeds_kmph = df["speed_kmph"].to_numpy(dtype=float)
    speeds_mps = speeds_kmph / 3.6

    n = len(df)
    cumulative_distance_m = np.zeros(n, dtype=float)

    for i in range(1, n):
        dt = max(0.0, times[i] - times[i - 1])
        dist_step = speeds_mps[i] * dt
        cumulative_distance_m[i] = cumulative_distance_m[i - 1] + dist_step

    return cumulative_distance_m


def build_speed_limit_timeline(
    master_timeline_csv: str,
    out_csv_path: str,
    sign_csv_path: str | None = None,
    sign_valid_distance_m: float = 300.0,
    default_speed_limit_kmph: float = 40.0,
):
    """
    Logic:
    - Use speed_kmph from master timeline.
    - If a speed sign event is detected, activate that speed limit.
    - Keep it active until vehicle travels sign_valid_distance_m.
    - If no new sign appears after that distance, fallback to default_speed_limit_kmph.
    - Overspeed = speed_kmph > active_speed_limit
    - No tolerance
    """
    df = load_master_timeline(master_timeline_csv)
    sign_map = load_sign_events(sign_csv_path)

    cumulative_distance_m = compute_incremental_distance_m(df)

    current_speed_limit = np.nan
    last_sign_frame = None
    last_sign_distance_m = None

    detected_speed_signs = []
    active_speed_limits = []
    limit_sources = []
    distance_since_last_signs = []
    overspeed_flags = []

    for idx, row in df.iterrows():
        frame_id = int(row["front_frame_id"])
        speed_kmph = float(row["speed_kmph"])
        current_distance_m = float(cumulative_distance_m[idx])

        detected_speed_sign = sign_map.get(frame_id, np.nan)

        # If new sign detected, update active sign
        if not pd.isna(detected_speed_sign):
            current_speed_limit = float(detected_speed_sign)
            last_sign_frame = frame_id
            last_sign_distance_m = current_distance_m

        # Check whether sign is still valid by distance
        if last_sign_distance_m is not None and not pd.isna(current_speed_limit):
            distance_since_last_sign = current_distance_m - last_sign_distance_m
        else:
            distance_since_last_sign = np.nan

        if (
            last_sign_distance_m is not None
            and not pd.isna(current_speed_limit)
            and distance_since_last_sign <= sign_valid_distance_m
        ):
            active_limit = float(current_speed_limit)
            limit_source = "SIGN"
        else:
            active_limit = float(default_speed_limit_kmph)
            limit_source = "DEFAULT"

        overspeed = 1 if speed_kmph > active_limit else 0

        detected_speed_signs.append(detected_speed_sign)
        active_speed_limits.append(active_limit)
        limit_sources.append(limit_source)
        distance_since_last_signs.append(distance_since_last_sign)
        overspeed_flags.append(overspeed)

    out_df = df.copy()
    out_df["cumulative_distance_m"] = cumulative_distance_m
    out_df["detected_speed_sign"] = detected_speed_signs
    out_df["distance_since_last_sign_m"] = distance_since_last_signs
    out_df["active_speed_limit"] = active_speed_limits
    out_df["limit_source"] = limit_sources
    out_df["overspeed_flag"] = overspeed_flags

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    out_df.to_csv(out_csv_path, index=False)
    return out_df


if __name__ == "__main__":
    master_timeline_csv = "outputs/master_timeline.csv"
    out_csv_path = "outputs/speed_limit_signals.csv"
    sign_csv_path = "outputs/speed_sign_events.csv"

    df = build_speed_limit_timeline(
        master_timeline_csv=master_timeline_csv,
        out_csv_path=out_csv_path,
        sign_csv_path=sign_csv_path,
        sign_valid_distance_m=300.0,
        default_speed_limit_kmph=40.0,
    )

    print(f"Saved speed-limit timeline to: {out_csv_path}")
    print(df.head())
    print(f"Total rows: {len(df)}")