import pandas as pd

def load_metadata(metadata_csv_path):
    df = pd.read_csv(metadata_csv_path)
    return df


def estimate_sensor_rate(metadata_df, video_duration_sec):
    if video_duration_sec <= 0:
        raise ValueError("Invalid video duration")
    return len(metadata_df) / video_duration_sec


def add_metadata_time_axis(metadata_df, sensor_rate_hz):
    df = metadata_df.copy()
    df["meta_row_id"] = range(len(df))
    df["meta_time_sec"] = df["meta_row_id"] / sensor_rate_hz
    return df


def attach_speed_kmph(metadata_df, speed_column="gps_spd_2d"):
    df = metadata_df.copy()
    if speed_column not in df.columns:
        raise ValueError(f"{speed_column} not found in metadata")
    df["speed_kmph"] = df[speed_column] * 3.6
    return df