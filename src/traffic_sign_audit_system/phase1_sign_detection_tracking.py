import os
from typing import Dict, Any

import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO


# =========================================================
# Helpers
# =========================================================

def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def get_best_device() -> str:
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))

    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)

    return x1, y1, x2, y2


def save_crop_and_context(
    frame_bgr: np.ndarray,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    crop_path: str,
    context_path: str,
    context_scale: float = 3.0,
    min_context_pad: int = 80,
):
    """
    Save:
    - exact sign crop
    - larger surrounding context crop
    """
    h, w = frame_bgr.shape[:2]

    crop = frame_bgr[y1:y2, x1:x2].copy()
    cv2.imwrite(crop_path, crop)

    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    cx = (x1 + x2) // 2
    cy = (y1 + y2) // 2

    ctx_w = max(int(bw * context_scale), bw + 2 * min_context_pad)
    ctx_h = max(int(bh * context_scale), bh + 2 * min_context_pad)

    cx1 = max(0, cx - ctx_w // 2)
    cy1 = max(0, cy - ctx_h // 2)
    cx2 = min(w, cx + ctx_w // 2)
    cy2 = min(h, cy + ctx_h // 2)

    if cx2 <= cx1:
        cx2 = min(w, cx1 + 1)
    if cy2 <= cy1:
        cy2 = min(h, cy1 + 1)

    context = frame_bgr[cy1:cy2, cx1:cx2].copy()
    cv2.imwrite(context_path, context)

    return crop, context


# =========================================================
# Main Phase 1
# =========================================================

def run_phase1_sign_detection_tracking(
    video_path: str,
    detector_model_path: str,
    output_dir: str = "outputs/traffic_sign_audit_system",
    start_frame: int = 0,
    end_frame: int = -1,
    conf_thres: float = 0.25,
    imgsz: int = 512,
    process_every: int = 1,
    tracker_cfg: str = "bytetrack.yaml",
    context_scale: float = 3.0,
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Phase 1:
    - detect traffic signs
    - track across frames
    - keep best observation per track
    - save sign crop
    - save context crop
    - save traffic_sign_tracks.csv
    """

    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")

    if not os.path.exists(detector_model_path):
        raise FileNotFoundError(f"Detector model not found: {detector_model_path}")

    ensure_dir(output_dir)
    crops_dir = os.path.join(output_dir, "sign_crops")
    context_dir = os.path.join(output_dir, "context_crops")
    ensure_dir(crops_dir)
    ensure_dir(context_dir)

    tracks_csv = os.path.join(output_dir, "traffic_sign_tracks.csv")

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if fps <= 0:
        cap.release()
        raise RuntimeError("Invalid video FPS")

    if end_frame < 0 or end_frame > num_frames:
        end_frame = num_frames

    start_frame = max(0, min(start_frame, max(0, num_frames - 1)))
    end_frame = max(start_frame + 1, min(end_frame, num_frames))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    model = YOLO(detector_model_path)
    device = get_best_device()

    # keep only best sample per track_id
    track_best: Dict[int, Dict[str, Any]] = {}

    total_to_process = max(1, end_frame - start_frame)
    processed = 0

    print(f"[Phase1] Video : {video_path}")
    print(f"[Phase1] Model : {detector_model_path}")
    print(f"[Phase1] Device: {device}")
    print(f"[Phase1] Frames: {start_frame} -> {end_frame}")

    for global_frame in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break

        if (global_frame - start_frame) % int(process_every) != 0:
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total_to_process, len(track_best))
            continue

        # try chosen device, fallback cpu
        try:
            results = model.track(
                frame,
                conf=conf_thres,
                imgsz=imgsz,
                device=device,
                tracker=tracker_cfg,
                persist=True,
                verbose=False,
            )
        except Exception:
            results = model.track(
                frame,
                conf=conf_thres,
                imgsz=imgsz,
                device="cpu",
                tracker=tracker_cfg,
                persist=True,
                verbose=False,
            )

        if not results or results[0].boxes is None or results[0].boxes.xyxy is None:
            processed += 1
            if progress_callback is not None:
                progress_callback(processed, total_to_process, len(track_best))
            continue

        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy().astype(float)
        cls_ids = boxes.cls.cpu().numpy().astype(int)

        # if tracker ids missing, create fallback ids
        if boxes.id is not None:
            track_ids = boxes.id.cpu().numpy().astype(int)
        else:
            track_ids = np.arange(len(xyxy)) + (global_frame * 10000)

        names = model.names if hasattr(model, "names") else {0: "traffic_sign"}

        for box, conf, cls_id, track_id in zip(xyxy, confs, cls_ids, track_ids):
            x1, y1, x2, y2 = box.tolist()
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, frame_w, frame_h)

            bbox_w = max(1, x2 - x1)
            bbox_h = max(1, y2 - y1)
            bbox_area = float(bbox_w * bbox_h)
            aspect_ratio = bbox_w / max(bbox_h, 1)

            # -------------------------------------------------
            # Basic filtering to remove junk detections
            # -------------------------------------------------
            if bbox_w < 12 or bbox_h < 12:
                continue

            if bbox_w > frame_w * 0.45 or bbox_h > frame_h * 0.45:
                continue

            if aspect_ratio < 0.3 or aspect_ratio > 3.5:
                continue

            # single-class detector should mostly be class 0
            class_name = str(names[int(cls_id)]) if int(cls_id) in names else "traffic_sign"

            time_sec = float(global_frame / fps)

            prev = track_best.get(int(track_id))

            # keep largest / best view of the track
            if prev is not None and bbox_area <= float(prev["best_area"]):
                continue

            crop_path = os.path.join(
                crops_dir,
                f"track_{int(track_id):05d}_frame_{int(global_frame):06d}.png"
            )
            context_path = os.path.join(
                context_dir,
                f"track_{int(track_id):05d}_frame_{int(global_frame):06d}.png"
            )

            save_crop_and_context(
                frame_bgr=frame,
                x1=x1,
                y1=y1,
                x2=x2,
                y2=y2,
                crop_path=crop_path,
                context_path=context_path,
                context_scale=context_scale,
                min_context_pad=80,
            )

            track_best[int(track_id)] = {
                "track_id": int(track_id),
                "frame": int(global_frame),
                "time_sec": time_sec,
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "bbox_w": int(bbox_w),
                "bbox_h": int(bbox_h),
                "best_area": float(bbox_area),
                "conf": float(conf),
                "detector_class_id": int(cls_id),
                "detector_class_name": class_name,
                "crop_path": crop_path,
                "context_path": context_path,
                "timestamp_sec": time_sec,
            }

        processed += 1
        if progress_callback is not None:
            progress_callback(processed, total_to_process, len(track_best))

    cap.release()

    if not track_best:
        empty_cols = [
            "track_id", "frame", "time_sec",
            "x1", "y1", "x2", "y2",
            "bbox_w", "bbox_h", "best_area",
            "conf", "detector_class_id", "detector_class_name",
            "crop_path", "context_path", "timestamp_sec"
        ]
        tracks_df = pd.DataFrame(columns=empty_cols)
        tracks_df.to_csv(tracks_csv, index=False)

        return {
            "tracks_csv": tracks_csv,
            "tracks_df": tracks_df,
            "num_tracks": 0,
            "message": "No traffic sign tracks found."
        }

    rows = sorted(track_best.values(), key=lambda r: (r["frame"], r["track_id"]))
    tracks_df = pd.DataFrame(rows)
    tracks_df.to_csv(tracks_csv, index=False)

    return {
        "tracks_csv": tracks_csv,
        "tracks_df": tracks_df,
        "num_tracks": len(tracks_df),
        "message": f"Phase 1 complete. Saved {len(tracks_df)} best traffic sign tracks."
    }


# =========================================================
# Standalone runner
# =========================================================
def _progress(done, total, count):
    print(f"[Phase1] processed={done}/{total} best_tracks={count}")


if __name__ == "__main__":
    detector_path = "runs/detect/runs/traffic_sign_phase1/traffic_sign_detector_fast_v1/weights/best.pt"

    if not os.path.exists(detector_path):
        print(f"[Phase1] WARNING: trained detector not found at: {detector_path}")
        print("[Phase1] Update detector_path at bottom of this file after training finishes.")

    result = run_phase1_sign_detection_tracking(
        video_path="data/front.mp4",
        detector_model_path=detector_path,
        output_dir="outputs/traffic_sign_audit_system",
        start_frame=0,
        end_frame=-1,
        conf_thres=0.25,
        imgsz=512,
        process_every=1,
        tracker_cfg="bytetrack.yaml",
        context_scale=3.0,
        progress_callback=_progress,
    )

    print("\n[Phase1] Done.")
    print("[Phase1] Tracks CSV :", result["tracks_csv"])
    print("[Phase1] Num tracks :", result["num_tracks"])
    print("[Phase1] Message    :", result["message"])