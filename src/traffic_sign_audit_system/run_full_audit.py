import os
from typing import Dict, Any

try:
    from traffic_sign_audit_system.phase1_sign_detection_tracking import run_phase1_sign_detection_tracking
    from traffic_sign_audit_system.phase2_sign_recognition import run_phase2_sign_recognition
    from traffic_sign_audit_system.phase3_visibility_condition import run_phase3_visibility_condition
    from traffic_sign_audit_system.phase4_geometry_metrics import run_phase4_geometry_metrics
    from traffic_sign_audit_system.phase5_telemetry_location import run_phase5_telemetry_location
    from traffic_sign_audit_system.phase6_final_audit_report import run_phase6_final_audit
except ModuleNotFoundError:
    from phase1_sign_detection_tracking import run_phase1_sign_detection_tracking
    from phase2_sign_recognition import run_phase2_sign_recognition
    from phase3_visibility_condition import run_phase3_visibility_condition
    from phase4_geometry_metrics import run_phase4_geometry_metrics
    from phase5_telemetry_location import run_phase5_telemetry_location
    from phase6_final_audit_report import run_phase6_final_audit

# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def print_stage(title: str):
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# =========================================================
# Main Runner
# =========================================================
def run_full_audit(
    video_path: str,
    telemetry_csv_path: str,
    detector_model_path: str,
    output_dir: str = "outputs/traffic_sign_audit_system",
    start_frame: int = 0,
    end_frame: int = -1,
    conf_thres: float = 0.25,
    imgsz: int = 512,
    process_every: int = 1,
    tracker_cfg: str = "bytetrack.yaml",
    context_scale: float = 3.0,
) -> Dict[str, Any]:
    """
    Run full traffic sign audit pipeline:
    Phase 1 -> detection + tracking
    Phase 2 -> recognition
    Phase 3 -> visibility
    Phase 4 -> geometry
    Phase 5 -> telemetry/location
    Phase 6 -> final report
    """

    ensure_dir(output_dir)

    # -------------------------
    # Phase 1
    # -------------------------
    print_stage("[FULL AUDIT] Phase 1: Sign Detection + Tracking")

    p1 = run_phase1_sign_detection_tracking(
        video_path=video_path,
        detector_model_path=detector_model_path,
        output_dir=output_dir,
        start_frame=start_frame,
        end_frame=end_frame,
        conf_thres=conf_thres,
        imgsz=imgsz,
        process_every=process_every,
        tracker_cfg=tracker_cfg,
        context_scale=context_scale,
        progress_callback=lambda done, total, count:
            print(f"[Phase1] processed={done}/{total} best_tracks={count}")
    )

    tracks_csv = p1["tracks_csv"]

    # -------------------------
    # Phase 2
    # -------------------------
    print_stage("[FULL AUDIT] Phase 2: Sign Recognition")

    p2 = run_phase2_sign_recognition(
        tracks_csv_path=tracks_csv,
        output_dir=output_dir,
        output_csv_name="traffic_sign_recognition.csv",
        progress_callback=lambda done, total:
            print(f"[Phase2] processed={done}/{total}")
    )

    recognition_csv = p2["recognition_csv"]

    # -------------------------
    # Phase 3
    # -------------------------
    print_stage("[FULL AUDIT] Phase 3: Visibility Condition")

    p3 = run_phase3_visibility_condition(
        recognition_csv_path=recognition_csv,
        output_dir=output_dir,
        output_csv_name="traffic_sign_visibility.csv",
        progress_callback=lambda done, total:
            print(f"[Phase3] processed={done}/{total}")
    )

    visibility_csv = p3["visibility_csv"]

    # -------------------------
    # Phase 4
    # -------------------------
    print_stage("[FULL AUDIT] Phase 4: Geometry Metrics")

    p4 = run_phase4_geometry_metrics(
        visibility_csv_path=visibility_csv,
        output_dir=output_dir,
        output_csv_name="traffic_sign_geometry.csv",
        progress_callback=lambda done, total:
            print(f"[Phase4] processed={done}/{total}")
    )

    geometry_csv = p4["geometry_csv"]

    # -------------------------
    # Phase 5
    # -------------------------
    print_stage("[FULL AUDIT] Phase 5: Telemetry + Location")

    p5 = run_phase5_telemetry_location(
        geometry_csv_path=geometry_csv,
        telemetry_csv_path=telemetry_csv_path,
        output_dir=output_dir,
        output_csv_name="traffic_sign_with_location.csv",
        progress_callback=lambda done, total:
            print(f"[Phase5] processed={done}/{total}")
    )

    location_csv = p5["location_csv"]

    # -------------------------
    # Phase 6
    # -------------------------
    print_stage("[FULL AUDIT] Phase 6: Final Audit Report")

    p6 = run_phase6_final_audit(
        location_csv_path=location_csv,
        recognition_csv_path=recognition_csv,
        visibility_csv_path=visibility_csv,
        output_dir=output_dir,
        output_csv_name="traffic_sign_final_report.csv",
        progress_callback=lambda done, total:
            print(f"[Phase6] processed={done}/{total}")
    )

    report_csv = p6["report_csv"]

    # -------------------------
    # Final Summary
    # -------------------------
    print_stage("[FULL AUDIT] COMPLETE")

    print("[Summary] Tracks CSV       :", tracks_csv)
    print("[Summary] Recognition CSV  :", recognition_csv)
    print("[Summary] Visibility CSV   :", visibility_csv)
    print("[Summary] Geometry CSV     :", geometry_csv)
    print("[Summary] Location CSV     :", location_csv)
    print("[Summary] Final Report CSV :", report_csv)

    return {
        "tracks_csv": tracks_csv,
        "recognition_csv": recognition_csv,
        "visibility_csv": visibility_csv,
        "geometry_csv": geometry_csv,
        "location_csv": location_csv,
        "report_csv": report_csv,
        "phase1": p1,
        "phase2": p2,
        "phase3": p3,
        "phase4": p4,
        "phase5": p5,
        "phase6": p6,
        "message": "Full traffic sign audit pipeline completed successfully."
    }


# =========================================================
# Run
# =========================================================
if __name__ == "__main__":
    detector_path = "runs/detect/runs/traffic_sign_phase1/traffic_sign_detector_fast_v1/weights/best.pt"

    if not os.path.exists(detector_path):
        print(f"[FULL AUDIT] WARNING: detector not found at: {detector_path}")
        print("[FULL AUDIT] Update detector_path in run_full_audit.py before running.")

    result = run_full_audit(
        video_path="data/front.mp4",
        telemetry_csv_path="data/front_telemetry.csv",
        detector_model_path=detector_path,
        output_dir="outputs/traffic_sign_audit_system",
        start_frame=0,
        end_frame=-1,
        conf_thres=0.25,
        imgsz=512,
        process_every=1,
        tracker_cfg="bytetrack.yaml",
        context_scale=3.0,
    )

    print("\n[FULL AUDIT] Done.")
    print("[FULL AUDIT] Message:", result["message"])