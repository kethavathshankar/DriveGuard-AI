import os
from typing import Dict, Any

import pandas as pd


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def assign_priority(severity, quality_score):
    severity = str(severity).lower()

    try:
        quality_score = float(quality_score)
    except Exception:
        quality_score = 1.0

    if severity == "high" or quality_score < 0.30:
        return "HIGH"
    elif severity == "medium" or quality_score < 0.60:
        return "MEDIUM"
    else:
        return "LOW"


def recommend_action(problem):
    problem = str(problem).lower()

    if "tree" in problem or "bush" in problem or "vegetation" in problem:
        return "Vegetation trimming required"
    elif "pole" in problem:
        return "Remove obstruction near sign"
    elif "vehicle" in problem:
        return "Clear temporary obstruction near sign"
    elif "blur" in problem:
        return "Inspect sign clarity and replace or clean sign board"
    elif "glare" in problem or "bright" in problem:
        return "Adjust sign angle or apply anti-glare solution"
    elif "faded" in problem or "low_contrast" in problem:
        return "Repaint or replace sign board"
    elif "tilted" in problem:
        return "Fix sign alignment"
    elif "small" in problem or "far" in problem:
        return "Install larger or closer sign if required"
    elif "edge" in problem:
        return "Reposition sign properly"
    elif "damaged" in problem:
        return "Repair or replace sign"
    elif "clear_visible" in problem:
        return "No action needed"
    else:
        return "Field inspection recommended"


def build_summary(row):
    sign_type = row.get("sign_type", "unknown_sign")
    frame = row.get("frame", "NA")
    problem = row.get("problem_type", "unknown")
    severity = row.get("severity", "unknown")
    lat = row.get("latitude", "NA")
    lon = row.get("longitude", "NA")

    return (
        f"Frame {frame}: {sign_type} sign has issue '{problem}' "
        f"(severity={severity}) at location ({lat}, {lon})."
    )


# =========================================================
# Main
# =========================================================
def run_phase6_final_audit(
    location_csv_path: str,
    recognition_csv_path: str,
    visibility_csv_path: str,
    output_dir="outputs/traffic_sign_audit_system",
    output_csv_name="traffic_sign_final_report.csv",
    progress_callback=None,
) -> Dict[str, Any]:

    ensure_dir(output_dir)
    out_csv = os.path.join(output_dir, output_csv_name)

    if not os.path.exists(location_csv_path):
        raise FileNotFoundError(f"Location CSV not found: {location_csv_path}")
    if not os.path.exists(recognition_csv_path):
        raise FileNotFoundError(f"Recognition CSV not found: {recognition_csv_path}")
    if not os.path.exists(visibility_csv_path):
        raise FileNotFoundError(f"Visibility CSV not found: {visibility_csv_path}")

    loc_df = pd.read_csv(location_csv_path)
    rec_df = pd.read_csv(recognition_csv_path)
    vis_df = pd.read_csv(visibility_csv_path)

    print("[Phase6] Merging data...")

    # Keep only needed columns from recognition
    rec_keep = [
        c for c in [
            "track_id", "frame",
            "sign_family", "sign_type", "speed_value",
            "ocr_text", "ocr_confidence", "recognition_confidence", "reason"
        ] if c in rec_df.columns
    ]
    rec_df = rec_df[rec_keep].copy()

    # Keep only needed columns from visibility
    vis_keep = [
        c for c in [
            "track_id", "frame",
            "problem_type", "severity", "is_clearly_visible", "explanation"
        ] if c in vis_df.columns
    ]
    vis_df = vis_df[vis_keep].copy()

    # Merge
    df = loc_df.merge(rec_df, on=["track_id", "frame"], how="left")
    df = df.merge(vis_df, on=["track_id", "frame"], how="left")

    rows_out = []
    total = len(df)

    for idx, row in df.iterrows():
        problem = row.get("problem_type", "unknown")
        severity = row.get("severity", "low")
        quality_score = row.get("quality_score", 1.0)

        priority = assign_priority(severity, quality_score)
        action = recommend_action(problem)
        summary = build_summary(row)

        rows_out.append({
            "track_id": row.get("track_id"),
            "frame": row.get("frame"),
            "time_sec": row.get("time_sec"),

            "sign_family": row.get("sign_family", "unknown"),
            "sign_type": row.get("sign_type", "unknown"),
            "speed_value": row.get("speed_value", None),

            "issue": problem,
            "severity": severity,
            "priority": priority,

            "latitude": row.get("latitude", None),
            "longitude": row.get("longitude", None),
            "altitude": row.get("altitude", None),
            "vehicle_speed": row.get("vehicle_speed", None),
            "maps_link": row.get("maps_link", ""),

            "quality_score": row.get("quality_score", None),
            "recommended_action": action,
            "visibility_explanation": row.get("explanation", ""),
            "summary": summary,
        })

        if progress_callback:
            progress_callback(idx + 1, total)

        if (idx + 1) % 100 == 0:
            print(f"[Phase6] processed {idx+1}/{total}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    return {
        "report_csv": out_csv,
        "num_rows": len(out_df),
        "message": f"Phase 6 complete. Saved {len(out_df)} rows."
    }


# =========================================================
# Run
# =========================================================
def _progress(done, total):
    print(f"[Phase6] processed={done}/{total}")


if __name__ == "__main__":
    result = run_phase6_final_audit(
        location_csv_path="outputs/traffic_sign_audit_system/traffic_sign_with_location.csv",
        recognition_csv_path="outputs/traffic_sign_audit_system/traffic_sign_recognition.csv",
        visibility_csv_path="outputs/traffic_sign_audit_system/traffic_sign_visibility.csv",
        output_dir="outputs/traffic_sign_audit_system",
        output_csv_name="traffic_sign_final_report.csv",
        progress_callback=_progress,
    )

    print("\n[Phase6] Done.")
    print("[Phase6] Report CSV :", result["report_csv"])
    print("[Phase6] Num rows   :", result["num_rows"])
    print("[Phase6] Message    :", result["message"])