import os
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_read_image(path: str):
    if not path or not os.path.exists(path):
        return None
    return cv2.imread(path)


# =========================================================
# Geometry Metrics
# =========================================================
def compute_diameter(bbox_w, bbox_h):
    return float(max(bbox_w, bbox_h))


def compute_area(bbox_w, bbox_h):
    return float(bbox_w * bbox_h)


def compute_aspect_ratio(bbox_w, bbox_h):
    if bbox_h == 0:
        return 0.0
    return float(bbox_w / bbox_h)


def compute_quality_score(blur, brightness, contrast, visible_area):
    """
    Simple combined score (0–1)
    """
    blur_score = min(1.0, blur / 100.0)
    contrast_score = min(1.0, contrast / 50.0)

    # brightness ideal ~120–180
    brightness_score = 1.0 - abs(brightness - 140) / 140
    brightness_score = max(0.0, brightness_score)

    score = (
        0.3 * blur_score +
        0.25 * contrast_score +
        0.25 * brightness_score +
        0.2 * visible_area
    )

    return float(round(score, 4))


# =========================================================
# Main Phase 4
# =========================================================
def run_phase4_geometry_metrics(
    visibility_csv_path: str,
    output_dir="outputs/traffic_sign_audit_system",
    output_csv_name="traffic_sign_geometry.csv",
    progress_callback=None,
) -> Dict[str, Any]:

    if not os.path.exists(visibility_csv_path):
        raise FileNotFoundError(f"Visibility CSV not found: {visibility_csv_path}")

    ensure_dir(output_dir)
    out_csv = os.path.join(output_dir, output_csv_name)

    df = pd.read_csv(visibility_csv_path)

    rows_out = []
    total = len(df)

    print(f"[Phase4] Total rows: {total}")

    for idx, row in df.iterrows():
        crop_path = row["crop_path"]

        crop = safe_read_image(crop_path)

        if crop is None:
            continue

        h, w = crop.shape[:2]

        diameter = compute_diameter(w, h)
        area = compute_area(w, h)
        aspect_ratio = compute_aspect_ratio(w, h)

        blur = row.get("blur_score", 0)
        brightness = row.get("brightness", 0)
        contrast = row.get("contrast", 0)
        visible_area = row.get("visible_area_estimate", 0)

        quality_score = compute_quality_score(
            blur, brightness, contrast, visible_area
        )

        rows_out.append({
            "track_id": row["track_id"],
            "frame": row["frame"],
            "time_sec": row["time_sec"],
            "crop_path": crop_path,

            # geometry
            "bbox_width": w,
            "bbox_height": h,
            "bbox_area": area,
            "diameter_px": diameter,
            "aspect_ratio": aspect_ratio,

            # reuse from phase3
            "blur_score": blur,
            "brightness": brightness,
            "contrast": contrast,
            "visible_area_estimate": visible_area,
            "tilt_angle_estimate": row.get("tilt_angle_estimate", 0),

            # new
            "quality_score": quality_score,
        })

        if progress_callback:
            progress_callback(idx + 1, total)

        if (idx + 1) % 100 == 0:
            print(f"[Phase4] processed {idx+1}/{total}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    return {
        "geometry_csv": out_csv,
        "num_rows": len(out_df),
        "message": f"Phase 4 complete. Saved {len(out_df)} rows."
    }


# =========================================================
# Run
# =========================================================
def _progress(done, total):
    print(f"[Phase4] processed={done}/{total}")


if __name__ == "__main__":
    visibility_csv = "outputs/traffic_sign_audit_system/traffic_sign_visibility.csv"

    result = run_phase4_geometry_metrics(
        visibility_csv_path=visibility_csv,
        output_dir="outputs/traffic_sign_audit_system",
        output_csv_name="traffic_sign_geometry.csv",
        progress_callback=_progress,
    )

    print("\n[Phase4] Done.")
    print("[Phase4] Geometry CSV :", result["geometry_csv"])
    print("[Phase4] Num rows     :", result["num_rows"])
    print("[Phase4] Message      :", result["message"])