import pandas as pd
from src.sync_utils import get_video_info, compute_dash_offset
from src.metadata_utils import (
    load_metadata,
    estimate_sensor_rate,
    add_metadata_time_axis,
    attach_speed_kmph,
)

def build_master_timeline(front_video_path, dash_video_path, metadata_csv_path):
    front_info = get_video_info(front_video_path)
    dash_info = get_video_info(dash_video_path)

    sync_info = compute_dash_offset(front_video_path, dash_video_path)
    offset_sec = sync_info["offset_sec"]

    metadata_df = load_metadata(metadata_csv_path)
    sensor_rate_hz = estimate_sensor_rate(metadata_df, front_info["duration_sec"])
    metadata_df = add_metadata_time_axis(metadata_df, sensor_rate_hz)
    metadata_df = attach_speed_kmph(metadata_df)

    rows = []

    for front_frame_id in range(front_info["frame_count"]):
        front_time_sec = front_frame_id / front_info["fps"]
        dash_time_sec = front_time_sec - offset_sec

        meta_row_id = int(round(front_time_sec * sensor_rate_hz))
        if meta_row_id < 0:
            meta_row_id = 0
        if meta_row_id >= len(metadata_df):
            meta_row_id = len(metadata_df) - 1

        speed_kmph = metadata_df.iloc[meta_row_id]["speed_kmph"]

        rows.append({
            "front_frame_id": front_frame_id,
            "front_time_sec": front_time_sec,
            "dash_time_sec": dash_time_sec,
            "meta_row_id": meta_row_id,
            "speed_kmph": speed_kmph
        })

    timeline_df = pd.DataFrame(rows)

    return timeline_df, {
        "front_info": front_info,
        "dash_info": dash_info,
        "sync_info": sync_info,
        "sensor_rate_hz": sensor_rate_hz
    }


if __name__ == "__main__":
    front_video = "data/front.mp4"
    dash_video = "data/dashboard.mp4"   # update this name to your actual dash file
    metadata_csv = "data/front_telemetry.csv"

    timeline_df, info = build_master_timeline(front_video, dash_video, metadata_csv)
    timeline_df.to_csv("outputs/master_timeline.csv", index=False)

    print("Master timeline saved to outputs/master_timeline.csv")
    print(info)