import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


def kmh_from_mps(mps: float) -> float:
    return float(mps) * 3.6


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


def pick_speed_column(df: pd.DataFrame) -> np.ndarray:
    if "gps_spd_2d" in df.columns:
        return df["gps_spd_2d"].to_numpy(dtype=float)
    if "gps_spd_3d" in df.columns:
        return df["gps_spd_3d"].to_numpy(dtype=float)
    raise ValueError("Telemetry CSV missing gps_spd_2d or gps_spd_3d columns.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input mp4")
    ap.add_argument("--telemetry", required=True, help="Path to telemetry CSV")

    ap.add_argument("--out", default="outputs/tracked_speed.mp4", help="Output mp4 path")
    ap.add_argument("--tracks_csv", default="outputs/tracks.csv", help="Tracking CSV path")
    ap.add_argument("--events", default="outputs/events.csv", help="Overspeed events CSV path")

    ap.add_argument("--model", default="yolov8s.pt", help="YOLO model")
    ap.add_argument("--conf", type=float, default=0.35, help="YOLO confidence")
    ap.add_argument("--imgsz", type=int, default=640, help="Inference image size")

    ap.add_argument("--speed_limit_kmh", type=float, default=30.0, help="Speed limit")
    ap.add_argument("--tolerance_kmh", type=float, default=5.0, help="Tolerance")

    # Run only part of video (fast dev)
    ap.add_argument("--start_frame", type=int, default=0, help="Start from this frame")
    ap.add_argument("--max_frames", type=int, default=5400,
                    help="Process only N frames (-1 = full). 5400=3 minutes at 30fps")

    args = ap.parse_args()

    # --- Telemetry ---
    df = pd.read_csv(args.telemetry)
    spd_mps = pick_speed_column(df)
    spd_kmh = np.array([kmh_from_mps(v) for v in spd_mps], dtype=float)
    spd_kmh = smooth_1d(spd_kmh, win=11)

    # --- Video ---
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # --- Output writer ---
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # --- Model ---
    model = YOLO(args.model)

    # --- Device (MPS preferred) ---
    device = "mps"
    try:
        import torch
        if not torch.backends.mps.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    print(f"✅ Using device: {device}")

    # --- Tracker config ---
    tracker_cfg = "bytetrack.yaml"  # built-in in Ultralytics

    # --- Overspeed event segmentation (clean report) ---
    limit_thr = args.speed_limit_kmh + args.tolerance_kmh
    overspeed_on = False
    current = None
    max_speed = 0.0
    overspeed_events = []

    # --- Tracking rows (for future violations like tailgating) ---
    track_rows = []

    total_to_process = nframes if args.max_frames < 0 else min(args.max_frames, max(0, nframes - args.start_frame))
    pbar = tqdm(total=total_to_process, desc="Track+Speed")

    frame_i = 0
    while True:
        if args.max_frames > 0 and frame_i >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        global_frame = args.start_frame + frame_i

        # Telemetry index mapping (global frame aligned)
        ti = int(global_frame * (len(spd_kmh) - 1) / max(nframes - 1, 1))
        ti = max(0, min(ti, len(spd_kmh) - 1))
        speed_now = float(spd_kmh[ti])
        t_sec = global_frame / fps if fps > 0 else None

        # --- Tracking inference ---
        results = model.track(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            tracker=tracker_cfg,
            persist=True,
            verbose=False
        )

        # Draw boxes + IDs
        annotated = results[0].plot()

        # --- Speed overlay ---
        cv2.putText(
            annotated,
            f"Speed: {speed_now:.1f} km/h | Limit: {args.speed_limit_kmh:.0f}+{args.tolerance_kmh:.0f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.9,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # --- Overspeed logic (instant red alert + clean events) ---
        is_overspeed = speed_now > limit_thr

        if is_overspeed:
            cv2.putText(
                annotated,
                "OVERSPEEDING!",
                (20, 85),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 0, 255),
                3,
                cv2.LINE_AA,
            )

            if not overspeed_on:
                overspeed_on = True
                current = {
                    "start_frame": int(global_frame),
                    "start_time_sec": float(t_sec) if t_sec is not None else None,
                }
                max_speed = speed_now
            else:
                max_speed = max(max_speed, speed_now)

        else:
            if overspeed_on:
                overspeed_on = False
                current["end_frame"] = int(global_frame)
                current["end_time_sec"] = float(t_sec) if t_sec is not None else None
                if current["start_time_sec"] is not None and current["end_time_sec"] is not None:
                    current["duration_sec"] = current["end_time_sec"] - current["start_time_sec"]
                else:
                    current["duration_sec"] = None
                current["max_speed_kmh"] = float(max_speed)
                current["limit_thr_kmh"] = float(limit_thr)
                overspeed_events.append(current)
                current = None
                max_speed = 0.0

        # --- Save tracking rows ---
        boxes = results[0].boxes
        if boxes is not None and boxes.id is not None:
            ids = boxes.id.cpu().numpy().astype(int)
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)
            xyxy = boxes.xyxy.cpu().numpy()

            for tid, c, cf, b in zip(ids, cls, confs, xyxy):
                x1, y1, x2, y2 = b.tolist()
                track_rows.append({
                    "frame": int(global_frame),
                    "time_sec": float(t_sec) if t_sec is not None else None,
                    "track_id": int(tid),
                    "cls": int(c),
                    "conf": float(cf),
                    "x1": float(x1), "y1": float(y1), "x2": float(x2), "y2": float(y2),
                    "speed_kmh": float(speed_now),
                })

        out.write(annotated)
        frame_i += 1
        pbar.update(1)

    # Close overspeed event if video ends while ON
    if overspeed_on and current is not None:
        global_frame = args.start_frame + (frame_i - 1)
        t_sec = global_frame / fps if fps > 0 else None
        current["end_frame"] = int(global_frame)
        current["end_time_sec"] = float(t_sec) if t_sec is not None else None
        if current["start_time_sec"] is not None and current["end_time_sec"] is not None:
            current["duration_sec"] = current["end_time_sec"] - current["start_time_sec"]
        else:
            current["duration_sec"] = None
        current["max_speed_kmh"] = float(max_speed)
        current["limit_thr_kmh"] = float(limit_thr)
        overspeed_events.append(current)

    pbar.close()
    cap.release()
    out.release()

    # Save CSVs
    if len(track_rows) > 0:
        pd.DataFrame(track_rows).to_csv(args.tracks_csv, index=False)
    else:
        pd.DataFrame(columns=["frame", "time_sec", "track_id", "cls", "conf", "x1", "y1", "x2", "y2", "speed_kmh"]).to_csv(
            args.tracks_csv, index=False
        )

    if len(overspeed_events) > 0:
        pd.DataFrame(overspeed_events).to_csv(args.events, index=False)
    else:
        pd.DataFrame(columns=["start_frame", "start_time_sec", "end_frame", "end_time_sec",
                              "duration_sec", "max_speed_kmh", "limit_thr_kmh"]).to_csv(args.events, index=False)

    print(f"✅ Saved video: {args.out}")
    print(f"✅ Saved tracks: {args.tracks_csv}")
    print(f"✅ Saved overspeed events: {args.events}")


if __name__ == "__main__":
    main()