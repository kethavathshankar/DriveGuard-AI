import argparse
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from collections import deque
import os


# ---------------- Utils ----------------
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


def find_class_id(names: dict, target: str):
    target = target.lower()
    for k, v in names.items():
        if str(v).lower() == target:
            return int(k)
    return None


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


# ------------- Traffic light color classifier -------------
def classify_signal_color(roi_bgr: np.ndarray):
    """
    HSV pixel-count baseline.
    Returns (state, conf_proxy).
    """
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN", 0.0

    roi = cv2.resize(roi_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    # RED (wraps)
    red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    # YELLOW
    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))

    # GREEN
    green = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

    r = int(np.sum(red > 0))
    y = int(np.sum(yellow > 0))
    g = int(np.sum(green > 0))

    total = r + y + g
    if total < 30:
        return "UNKNOWN", 0.0

    mx = max(r, y, g)
    conf = mx / (total + 1e-6)

    if mx == r:
        return "RED", float(conf)
    if mx == y:
        return "YELLOW", float(conf)
    return "GREEN", float(conf)


# ---------------- Main ----------------
def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--video", required=True)
    ap.add_argument("--telemetry", required=True)

    ap.add_argument("--model", default="yolov8s.pt")
    ap.add_argument("--conf", type=float, default=0.30)
    ap.add_argument("--imgsz", type=int, default=960)

    ap.add_argument("--start_frame", type=int, default=0)
    ap.add_argument("--max_frames", type=int, default=5400)

    ap.add_argument("--out", default="outputs/live_overlay.mp4")
    ap.add_argument("--tracks_csv", default="outputs/tracks.csv")
    ap.add_argument("--signal_csv", default="outputs/signal.csv")
    ap.add_argument("--overspeed_csv", default="outputs/overspeed_events.csv")
    ap.add_argument("--red_csv", default="outputs/red_violations.csv")

    # speed rule
    ap.add_argument("--speed_limit_kmh", type=float, default=30.0)
    ap.add_argument("--tolerance_kmh", type=float, default=5.0)

    # traffic light detection filtering
    ap.add_argument("--min_tl_area", type=float, default=40.0)

    # RED stability + red-light logic
    ap.add_argument("--red_hold_frames", type=int, default=12)
    ap.add_argument("--min_speed_kmh", type=float, default=3.0)
    ap.add_argument("--near_tl_h_ratio", type=float, default=0.03)
    ap.add_argument("--cooldown_sec", type=float, default=8.0)

    # UI
    ap.add_argument("--show_safe_text", action="store_true")
    ap.add_argument("--violation_hold_sec", type=float, default=3.0)

    args = ap.parse_args()

    os.makedirs("outputs", exist_ok=True)

    # ----- Telemetry -----
    df = pd.read_csv(args.telemetry)
    spd_mps = pick_speed_column(df)
    spd_kmh = np.array([kmh_from_mps(v) for v in spd_mps], dtype=float)
    spd_kmh = smooth_1d(spd_kmh, win=11)

    # ----- Video -----
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

    # ----- Model -----
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

    tl_cls_id = find_class_id(model.names, "traffic light")
    if tl_cls_id is None:
        print("⚠️ 'traffic light' class not found in model.names. Signal will stay UNKNOWN.")
    limit_thr = args.speed_limit_kmh + args.tolerance_kmh

    # ----- RED stability -----
    state_window = deque(maxlen=args.red_hold_frames)
    conf_window = deque(maxlen=args.red_hold_frames)

    # FSM for red-light
    fsm = 0
    red_start_frame = None
    last_trigger_time = -1e9
    last_red_violation_time = -1e9

    # overspeed event segmentation
    overspeed_on = False
    overspeed_cur = None
    overspeed_max = 0.0
    overspeed_events = []

    # logs
    track_rows = []
    signal_rows = []
    red_violations = []

    total_to_process = nframes if args.max_frames < 0 else min(args.max_frames, max(0, nframes - args.start_frame))
    pbar = tqdm(total=total_to_process, desc="LiveOverlay")

    frame_i = 0
    while True:
        if args.max_frames > 0 and frame_i >= args.max_frames:
            break

        ret, frame = cap.read()
        if not ret:
            break

        global_frame = args.start_frame + frame_i
        t_sec = global_frame / fps if fps > 0 else None

        # ---- speed from telemetry mapped to frame ----
        ti = int(global_frame * (len(spd_kmh) - 1) / max(nframes - 1, 1))
        ti = max(0, min(ti, len(spd_kmh) - 1))
        speed_now = float(spd_kmh[ti])

        # ---- tracking ----
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

        # ---- speed overlay ----
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

        is_overspeed = speed_now > limit_thr

        # ---- overspeed event segmentation ----
        if is_overspeed:
            if not overspeed_on:
                overspeed_on = True
                overspeed_cur = {
                    "event": "overspeed",
                    "start_frame": int(global_frame),
                    "start_time_sec": float(t_sec) if t_sec is not None else None,
                    "limit_thr_kmh": float(limit_thr),
                }
                overspeed_max = speed_now
            else:
                overspeed_max = max(overspeed_max, speed_now)
        else:
            if overspeed_on and overspeed_cur is not None:
                overspeed_on = False
                overspeed_cur["end_frame"] = int(global_frame)
                overspeed_cur["end_time_sec"] = float(t_sec) if t_sec is not None else None
                if overspeed_cur["start_time_sec"] is not None and overspeed_cur["end_time_sec"] is not None:
                    overspeed_cur["duration_sec"] = overspeed_cur["end_time_sec"] - overspeed_cur["start_time_sec"]
                else:
                    overspeed_cur["duration_sec"] = None
                overspeed_cur["max_speed_kmh"] = float(overspeed_max)
                overspeed_events.append(overspeed_cur)
                overspeed_cur = None
                overspeed_max = 0.0

        # ---- traffic light state ----
        tl_state, tl_conf, tl_box = "UNKNOWN", 0.0, None
        near_intersection = False

        boxes = results[0].boxes
        if boxes is not None and boxes.xyxy is not None and tl_cls_id is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)

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

                tl_h_ratio = (y2 - y1) / float(h)
                near_intersection = tl_h_ratio >= args.near_tl_h_ratio

                color = (0, 0, 255) if tl_state == "RED" else (255, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

        # signal overlay
        cv2.putText(
            annotated,
            f"SIGNAL: {tl_state} ({tl_conf:.2f})",
            (20, 85),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255) if tl_state == "RED" else (255, 255, 255),
            3 if tl_state == "RED" else 2,
            cv2.LINE_AA,
        )

        # ---- RED stability ----
        state_window.append(tl_state)
        conf_window.append(tl_conf)
        red_count = sum(1 for s in state_window if s == "RED")
        red_stable = (len(state_window) == state_window.maxlen) and (red_count == state_window.maxlen)
        avg_red_conf = float(np.mean(conf_window)) if len(conf_window) else 0.0
        cooldown_ok = (t_sec is None) or (t_sec - last_trigger_time >= args.cooldown_sec)

        # ---- Red light FSM ----
        if fsm == 0:
            if red_stable and near_intersection:
                fsm = 1
                red_start_frame = global_frame

        elif fsm == 1:
            if not red_stable:
                fsm = 0
                red_start_frame = None
            else:
                if speed_now >= args.min_speed_kmh and cooldown_ok:
                    last_trigger_time = float(t_sec) if t_sec is not None else last_trigger_time
                    if t_sec is not None:
                        last_red_violation_time = float(t_sec)

                    red_violations.append({
                        "event": "red_light_violation",
                        "frame": int(global_frame),
                        "time_sec": float(t_sec) if t_sec is not None else None,
                        "speed_kmh": float(speed_now),
                        "red_hold_frames": int(args.red_hold_frames),
                        "avg_red_conf_proxy": float(avg_red_conf),
                        "near_intersection": bool(near_intersection),
                        "red_start_frame": int(red_start_frame) if red_start_frame is not None else None,
                        "tl_box": str(tl_box) if tl_box is not None else ""
                    })

                    # reset
                    fsm = 0
                    red_start_frame = None

        # ---- Live violation overlay list ----
        active = []
        if is_overspeed:
            active.append(f"OVERSPEEDING: {speed_now:.1f} > {limit_thr:.0f} km/h")

        if t_sec is not None and (t_sec - last_red_violation_time) <= args.violation_hold_sec:
            active.append("RED LIGHT VIOLATION")

        # warning (not a violation) if moving while signal looks red
        if tl_state == "RED" and near_intersection and speed_now >= args.min_speed_kmh:
            active.append("WARNING: MOVING ON RED")

        # draw
        y0 = 130
        if len(active) > 0:
            for i, msg in enumerate(active):
                cv2.putText(
                    annotated,
                    msg,
                    (20, y0 + 40*i),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (0, 0, 255),
                    3,
                    cv2.LINE_AA,
                )
        else:
            if args.show_safe_text:
                cv2.putText(
                    annotated,
                    "DRIVING SAFE",
                    (20, y0),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.0,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

        # ---- logging ----
        signal_rows.append({
            "frame": int(global_frame),
            "time_sec": float(t_sec) if t_sec is not None else None,
            "signal_state": tl_state,
            "signal_conf_proxy": float(tl_conf),
            "near_intersection": bool(near_intersection),
            "speed_kmh": float(speed_now),
            "tl_box": str(tl_box) if tl_box is not None else ""
        })

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

    # close overspeed event if still on
    if overspeed_on and overspeed_cur is not None:
        global_frame = args.start_frame + (frame_i - 1)
        t_sec = global_frame / fps if fps > 0 else None
        overspeed_cur["end_frame"] = int(global_frame)
        overspeed_cur["end_time_sec"] = float(t_sec) if t_sec is not None else None
        if overspeed_cur["start_time_sec"] is not None and overspeed_cur["end_time_sec"] is not None:
            overspeed_cur["duration_sec"] = overspeed_cur["end_time_sec"] - overspeed_cur["start_time_sec"]
        else:
            overspeed_cur["duration_sec"] = None
        overspeed_cur["max_speed_kmh"] = float(overspeed_max)
        overspeed_events.append(overspeed_cur)

    pbar.close()
    cap.release()
    out.release()

    pd.DataFrame(track_rows).to_csv(args.tracks_csv, index=False)
    pd.DataFrame(signal_rows).to_csv(args.signal_csv, index=False)
    pd.DataFrame(overspeed_events).to_csv(args.overspeed_csv, index=False)
    pd.DataFrame(red_violations).to_csv(args.red_csv, index=False)

    print(f"✅ Saved video: {args.out}")
    print(f"✅ Saved tracks: {args.tracks_csv}")
    print(f"✅ Saved signal: {args.signal_csv}")
    print(f"✅ Saved overspeed events: {args.overspeed_csv}")
    print(f"✅ Saved red violations: {args.red_csv}")


if __name__ == "__main__":
    main()