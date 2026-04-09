import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from collections import deque


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
    if x2 <= x1: x2 = min(w - 1, x1 + 1)
    if y2 <= y1: y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def classify_signal_color(roi_bgr: np.ndarray):
    """HSV pixel-count baseline. Returns (state, conf_proxy)."""
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN", 0.0
    roi = cv2.resize(roi_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
    green  = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

    r = int(np.sum(red > 0))
    y = int(np.sum(yellow > 0))
    g = int(np.sum(green > 0))
    total = r + y + g
    if total < 30:
        return "UNKNOWN", 0.0

    mx = max(r, y, g)
    conf = mx / (total + 1e-6)
    if mx == r: return "RED", float(conf)
    if mx == y: return "YELLOW", float(conf)
    return "GREEN", float(conf)


def find_class_id(names: dict, target: str):
    target = target.lower()
    for k, v in names.items():
        if str(v).lower() == target:
            return int(k)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True)
    ap.add_argument("--telemetry", required=True)

    ap.add_argument("--out", default="outputs/first_3min_redlight.mp4")
    ap.add_argument("--violations", default="outputs/red_light_violations.csv")
    ap.add_argument("--signal_csv", default="outputs/signal_state.csv")

    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--imgsz", type=int, default=640)

    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=5400)  # 3 min @30fps

    # Traffic light detection filtering
    ap.add_argument("--min_tl_area", type=float, default=80.0)

    # RED stability + movement thresholds
    ap.add_argument("--red_hold_frames", type=int, default=12, help="RED must be stable for N frames")
    ap.add_argument("--min_speed_kmh", type=float, default=3.0, help="consider vehicle moving above this")

    # Intersection proximity proxy:
    # When traffic light bbox height ratio exceeds this, assume we're close to intersection
    ap.add_argument("--near_tl_h_ratio", type=float, default=0.06)

    # Cooldown to avoid repeated triggers
    ap.add_argument("--cooldown_sec", type=float, default=8.0)

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

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    model = YOLO(args.model)

    device = "mps"
    try:
        import torch
        if not torch.backends.mps.is_available():
            device = "cpu"
    except Exception:
        device = "cpu"
    print(f"✅ Using device: {device}")

    tl_cls_id = find_class_id(model.names, "traffic light")
    if tl_cls_id is None:
        raise RuntimeError("Could not find 'traffic light' class in model.names")

    # State smoothing (stability)
    state_window = deque(maxlen=args.red_hold_frames)
    conf_window = deque(maxlen=args.red_hold_frames)

    # Red-light violation finite state machine
    # 0 = idle
    # 1 = red_stable_and_near (watching for passing)
    fsm = 0
    red_start_frame = None
    last_trigger_time = -1e9

    violations = []
    signal_rows = []

    total_to_process = nframes if args.max_frames < 0 else min(args.max_frames, max(0, nframes - args.start_frame))
    pbar = tqdm(total=total_to_process, desc="RedLight")

    frame_i = 0
    while True:
        if args.max_frames > 0 and frame_i >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        global_frame = args.start_frame + frame_i
        t_sec = global_frame / fps if fps > 0 else None

        # speed now
        ti = int(global_frame * (len(spd_kmh) - 1) / max(nframes - 1, 1))
        ti = max(0, min(ti, len(spd_kmh) - 1))
        speed_now = float(spd_kmh[ti])

        # Detect traffic lights (no tracking needed here)
        res = model.predict(frame, conf=args.conf, imgsz=args.imgsz, device=device, verbose=False)[0]

        tl_state = "UNKNOWN"
        tl_conf = 0.0
        tl_box = None
        near_intersection = False

        if res.boxes is not None and res.boxes.xyxy is not None:
            xyxy = res.boxes.xyxy.cpu().numpy()
            cls = res.boxes.cls.cpu().numpy().astype(int)
            confs = res.boxes.conf.cpu().numpy().astype(float)

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

                # proximity proxy: if TL bbox becomes “large enough”
                tl_h_ratio = (y2 - y1) / float(h)
                near_intersection = tl_h_ratio >= args.near_tl_h_ratio

                # draw box + label
                color = (0, 0, 255) if tl_state == "RED" else (255, 255, 255)
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, f"TL {tl_state} {tl_conf:.2f}",
                            (x1, max(0, y1 - 10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2, cv2.LINE_AA)

        # smooth state
        state_window.append(tl_state)
        conf_window.append(tl_conf)
        red_count = sum(1 for s in state_window if s == "RED")
        red_stable = (len(state_window) == state_window.maxlen) and (red_count == state_window.maxlen)
        avg_red_conf = float(np.mean(conf_window)) if len(conf_window) else 0.0

        # overlay speed + signal
        cv2.putText(frame, f"Speed: {speed_now:.1f} km/h",
                    (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(frame, f"SIGNAL: {tl_state} ({tl_conf:.2f})",
                    (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255) if tl_state == "RED" else (255, 255, 255),
                    3 if tl_state == "RED" else 2, cv2.LINE_AA)

        # FSM logic
        cooldown_ok = (t_sec is None) or (t_sec - last_trigger_time >= args.cooldown_sec)

        if fsm == 0:
            # Arm when RED is stable + near intersection
            if red_stable and near_intersection:
                fsm = 1
                red_start_frame = global_frame

        elif fsm == 1:
            # If red is no longer stable, disarm
            if not red_stable:
                fsm = 0
                red_start_frame = None
            else:
                # If moving while RED stable & near intersection -> violation trigger
                if speed_now >= args.min_speed_kmh and cooldown_ok:
                    # Trigger
                    last_trigger_time = float(t_sec) if t_sec is not None else last_trigger_time
                    violations.append({
                        "event": "red_light_violation",
                        "frame": int(global_frame),
                        "time_sec": float(t_sec) if t_sec is not None else None,
                        "speed_kmh": float(speed_now),
                        "red_stable_frames": int(args.red_hold_frames),
                        "avg_red_conf_proxy": float(avg_red_conf),
                        "near_intersection": bool(near_intersection),
                        "red_start_frame": int(red_start_frame) if red_start_frame is not None else None,
                        "tl_box": str(tl_box) if tl_box is not None else ""
                    })

                    cv2.putText(frame, "RED LIGHT VIOLATION!",
                                (20, 135), cv2.FONT_HERSHEY_SIMPLEX, 1.1,
                                (0, 0, 255), 4, cv2.LINE_AA)

                    # After trigger, disarm (will re-arm after red stable again)
                    fsm = 0
                    red_start_frame = None

        # always log signal state
        signal_rows.append({
            "frame": int(global_frame),
            "time_sec": float(t_sec) if t_sec is not None else None,
            "signal_state": tl_state,
            "signal_conf_proxy": float(tl_conf),
            "near_intersection": bool(near_intersection),
            "speed_kmh": float(speed_now),
            "tl_box": str(tl_box) if tl_box is not None else ""
        })

        out.write(frame)
        frame_i += 1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    pd.DataFrame(signal_rows).to_csv(args.signal_csv, index=False)
    pd.DataFrame(violations).to_csv(args.violations, index=False)

    print(f"✅ Saved: {args.out}")
    print(f"✅ Saved signal log: {args.signal_csv}")
    print(f"✅ Saved violations: {args.violations}")


if __name__ == "__main__":
    main()