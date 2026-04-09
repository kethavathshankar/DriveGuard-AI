import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


# ---------------- Utilities ----------------
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


# ---------------- Traffic light state classifier ----------------
def classify_signal_color(roi_bgr: np.ndarray) -> tuple[str, float]:
    """
    Returns (state, confidence_proxy).
    Very practical baseline: counts red/yellow/green pixels in HSV.
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN", 0.0

    # Resize to stabilize
    roi = cv2.resize(roi_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # Masks (HSV)
    # Red wraps around: [0..10] and [160..180]
    red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    # Yellow
    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))

    # Green
    green = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

    r = int(np.sum(red > 0))
    y = int(np.sum(yellow > 0))
    g = int(np.sum(green > 0))

    total = r + y + g
    if total < 30:  # too few colored pixels
        return "UNKNOWN", 0.0

    # pick max
    mx = max(r, y, g)
    conf = mx / (total + 1e-6)

    if mx == r:
        return "RED", float(conf)
    if mx == y:
        return "YELLOW", float(conf)
    return "GREEN", float(conf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--telemetry", required=True)

    ap.add_argument("--out", default="outputs/first_3min_tracked_signal.mp4")
    ap.add_argument("--tracks_csv", default="outputs/first_3min_tracks.csv")
    ap.add_argument("--overspeed_csv", default="outputs/first_3min_overspeed.csv")
    ap.add_argument("--signal_csv", default="outputs/first_3min_signal.csv")

    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument("--speed_limit_kmh", type=float, default=30.0)
    ap.add_argument("--tolerance_kmh", type=float, default=5.0)

    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=5400)

    # Traffic light detection filter
    ap.add_argument("--min_tl_area", type=float, default=150.0, help="Ignore tiny TL boxes")
    args = ap.parse_args()

    # Telemetry
    df = pd.read_csv(args.telemetry)
    spd_mps = pick_speed_column(df)
    spd_kmh = np.array([kmh_from_mps(v) for v in spd_mps], dtype=float)
    spd_kmh = smooth_1d(spd_kmh, win=11)

    # Video
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES, args.start_frame)

    # Writer
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    # Model + device
    model = YOLO(args.model)
    device = "mps"
    try:
        import torch
        if not torch.backends.mps.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    print(f"✅ Using device: {device}")

    tracker_cfg = "bytetrack.yaml"

    # Overspeed events
    limit_thr = args.speed_limit_kmh + args.tolerance_kmh
    overspeed_on = False
    current = None
    max_speed = 0.0
    overspeed_events = []

    # Logs
    track_rows = []
    signal_rows = []

    total_to_process = nframes if args.max_frames < 0 else min(args.max_frames, max(0, nframes - args.start_frame))
    pbar = tqdm(total=total_to_process, desc="Track+Speed+Signal")

    frame_i = 0
    while True:
        if args.max_frames > 0 and frame_i >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        global_frame = args.start_frame + frame_i
        t_sec = global_frame / fps if fps > 0 else None

        # Telemetry mapping
        ti = int(global_frame * (len(spd_kmh) - 1) / max(nframes - 1, 1))
        ti = max(0, min(ti, len(spd_kmh) - 1))
        speed_now = float(spd_kmh[ti])

        # Track
        results = model.track(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            tracker=tracker_cfg,
            persist=True,
            verbose=False
        )

        annotated = results[0].plot()

        # Speed overlay
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

        # Overspeed
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
                current = {"start_frame": int(global_frame), "start_time_sec": float(t_sec) if t_sec is not None else None}
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
                current, max_speed = None, 0.0

        # ---- Traffic light state (baseline) ----
        # COCO: traffic light class id is 9 in COCO for many YOLO models,
        # but we should NOT hardcode. We use model.names to find "traffic light".
        tl_cls_id = None
        for k, v in model.names.items():
            if str(v).lower() == "traffic light":
                tl_cls_id = int(k)
                break

        tl_state = "UNKNOWN"
        tl_conf = 0.0
        tl_box = None

        boxes = results[0].boxes
        if boxes is not None and boxes.xyxy is not None and tl_cls_id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)

            # pick the best traffic light box (highest conf, not tiny)
            best = None
            for b, c, cf in zip(xyxy, cls, confs):
                if c != tl_cls_id:
                    continue
                x1, y1, x2, y2 = b.tolist()
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                if area < args.min_tl_area:
                    continue
                if best is None or cf > best[0]:
                    best = (cf, (x1, y1, x2, y2))

            if best is not None:
                _, (x1, y1, x2, y2) = best
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, w, h)
                tl_box = (x1, y1, x2, y2)
                roi = frame[y1:y2, x1:x2]
                tl_state, tl_conf = classify_signal_color(roi)

                # overlay state
                cv2.putText(
                    annotated,
                    f"SIGNAL: {tl_state} ({tl_conf:.2f})",
                    (20, 130),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255) if tl_state == "RED" else (255, 255, 255),
                    3 if tl_state == "RED" else 2,
                    cv2.LINE_AA,
                )

                # Draw TL box stronger
                cv2.rectangle(annotated, (x1, y1), (x2, y2), (0, 0, 255) if tl_state == "RED" else (255, 255, 255), 2)

        signal_rows.append({
            "frame": int(global_frame),
            "time_sec": float(t_sec) if t_sec is not None else None,
            "signal_state": tl_state,
            "signal_conf": float(tl_conf),
            "speed_kmh": float(speed_now),
            "tl_box": str(tl_box) if tl_box is not None else ""
        })

        # Save tracking rows
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

    # Close overspeed event if still ON
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
    pd.DataFrame(track_rows).to_csv(args.tracks_csv, index=False)
    pd.DataFrame(overspeed_events).to_csv(args.overspeed_csv, index=False)
    pd.DataFrame(signal_rows).to_csv(args.signal_csv, index=False)

    print(f"✅ Saved video: {args.out}")
    print(f"✅ Saved tracks: {args.tracks_csv}")
    print(f"✅ Saved overspeed events: {args.overspeed_csv}")
    print(f"✅ Saved signal states: {args.signal_csv}")


if __name__ == "__main__":
    main()