import cv2
import pandas as pd
import numpy as np
import os


def green_ratio_and_peak(bgr_roi):
    hsv = cv2.cvtColor(bgr_roi, cv2.COLOR_BGR2HSV)

    lower = np.array([35, 35, 35], dtype=np.uint8)
    upper = np.array([95, 255, 255], dtype=np.uint8)

    mask = cv2.inRange(hsv, lower, upper)
    green_ratio = float(np.count_nonzero(mask)) / float(mask.size)

    g_channel = bgr_roi[:, :, 1].astype(np.float32)
    if np.count_nonzero(mask) > 0:
        green_peak = float(np.max(g_channel[mask > 0]))
    else:
        green_peak = 0.0

    return green_ratio, green_peak, mask


def choose_rois(video_path, frame_id=500):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_id)
    ok, frame = cap.read()
    cap.release()

    if not ok:
        raise RuntimeError(f"Could not read frame {frame_id} for ROI selection")

    print("\nSelect LEFT indicator ROI (ONLY tiny top-left blinking lamp), then press ENTER")
    left = cv2.selectROI("LEFT indicator ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("LEFT indicator ROI")

    print("Select RIGHT indicator ROI (ONLY tiny top-right blinking lamp), then press ENTER")
    right = cv2.selectROI("RIGHT indicator ROI", frame, fromCenter=False, showCrosshair=True)
    cv2.destroyWindow("RIGHT indicator ROI")

    lx, ly, lw, lh = left
    rx, ry, rw, rh = right

    left_roi = (int(lx), int(ly), int(lx + lw), int(ly + lh))
    right_roi = (int(rx), int(ry), int(rx + rw), int(ry + rh))

    print("\nSelected ROIs:")
    print("left_roi =", left_roi)
    print("right_roi =", right_roi)

    return left_roi, right_roi


def rolling_active(raw_series, window_size=15, min_on_count=2):
    raw_arr = np.array(raw_series, dtype=np.int32)
    active = []

    running_sum = 0
    q = []

    for v in raw_arr:
        q.append(v)
        running_sum += v

        if len(q) > window_size:
            running_sum -= q.pop(0)

        active.append(1 if running_sum >= min_on_count else 0)

    return active


def extract_indicators(
    dash_video_path,
    out_csv_path,
    left_roi,
    right_roi,
    process_every=2,
    save_debug_video=True,
    debug_video_path="outputs/dashboard_indicator_debug.mp4",
    max_frames=3000,
    baseline_frames=80,
    left_ratio_margin=0.0012,
    right_ratio_margin=0.0012,
    left_peak_margin=25.0,
    right_peak_margin=25.0,
    blink_window_frames=15,
    blink_min_on_count=2,
):
    cap = cv2.VideoCapture(dash_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open dashboard video: {dash_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    rows = []

    writer = None
    if save_debug_video:
        os.makedirs(os.path.dirname(debug_video_path), exist_ok=True)
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        debug_fps = max(1.0, fps / process_every)
        writer = cv2.VideoWriter(debug_video_path, fourcc, debug_fps, (width, height))

    frame_id = 0

    # collect initial baseline stats
    left_ratio_hist = []
    right_ratio_hist = []
    left_peak_hist = []
    right_peak_hist = []

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        if max_frames is not None and frame_id >= max_frames:
            break

        if frame_id % 300 == 0:
            print(f"Processing frame {frame_id}/{frame_count}")

        if frame_id % process_every != 0:
            frame_id += 1
            continue

        t_sec = frame_id / fps

        lx1, ly1, lx2, ly2 = left_roi
        rx1, ry1, rx2, ry2 = right_roi

        left_crop = frame[ly1:ly2, lx1:lx2]
        right_crop = frame[ry1:ry2, rx1:rx2]

        left_ratio, left_peak, _ = green_ratio_and_peak(left_crop)
        right_ratio, right_peak, _ = green_ratio_and_peak(right_crop)

        # build baseline from early samples
        if len(left_ratio_hist) < baseline_frames:
            left_ratio_hist.append(left_ratio)
            right_ratio_hist.append(right_ratio)
            left_peak_hist.append(left_peak)
            right_peak_hist.append(right_peak)

        left_ratio_base = float(np.median(left_ratio_hist)) if left_ratio_hist else 0.0
        right_ratio_base = float(np.median(right_ratio_hist)) if right_ratio_hist else 0.0
        left_peak_base = float(np.median(left_peak_hist)) if left_peak_hist else 0.0
        right_peak_base = float(np.median(right_peak_hist)) if right_peak_hist else 0.0

        left_raw_on = 1 if (
            (left_ratio - left_ratio_base) >= left_ratio_margin or
            (left_peak - left_peak_base) >= left_peak_margin
        ) else 0

        right_raw_on = 1 if (
            (right_ratio - right_ratio_base) >= right_ratio_margin or
            (right_peak - right_peak_base) >= right_peak_margin
        ) else 0

        rows.append({
            "dash_frame_id": frame_id,
            "dash_time_sec": t_sec,
            "left_ratio": left_ratio,
            "right_ratio": right_ratio,
            "left_peak": left_peak,
            "right_peak": right_peak,
            "left_ratio_base": left_ratio_base,
            "right_ratio_base": right_ratio_base,
            "left_peak_base": left_peak_base,
            "right_peak_base": right_peak_base,
            "left_raw_on": left_raw_on,
            "right_raw_on": right_raw_on,
        })

        if writer is not None:
            dbg = frame.copy()

            cv2.rectangle(dbg, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
            cv2.rectangle(dbg, (rx1, ry1), (rx2, ry2), (255, 255, 0), 2)

            cv2.putText(
                dbg,
                f"L r:{left_ratio:.4f} b:{left_ratio_base:.4f} p:{left_peak:.1f} raw:{left_raw_on}",
                (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                dbg,
                f"R r:{right_ratio:.4f} b:{right_ratio_base:.4f} p:{right_peak:.1f} raw:{right_raw_on}",
                (20, 75),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (255, 255, 0),
                2,
            )

            writer.write(dbg)

        frame_id += 1

    cap.release()
    if writer is not None:
        writer.release()

    df = pd.DataFrame(rows)

    df["left_indicator_active"] = rolling_active(
        df["left_raw_on"].tolist(),
        window_size=blink_window_frames,
        min_on_count=blink_min_on_count,
    )
    df["right_indicator_active"] = rolling_active(
        df["right_raw_on"].tolist(),
        window_size=blink_window_frames,
        min_on_count=blink_min_on_count,
    )

    os.makedirs(os.path.dirname(out_csv_path), exist_ok=True)
    df.to_csv(out_csv_path, index=False)
    return df


if __name__ == "__main__":
    dash_video = "data/dashboard.mp4"
    out_csv = "outputs/dashboard_indicators.csv"

    left_roi, right_roi = choose_rois(dash_video, frame_id=500)

    df = extract_indicators(
        dash_video_path=dash_video,
        out_csv_path=out_csv,
        left_roi=left_roi,
        right_roi=right_roi,
        process_every=2,
        save_debug_video=True,
        debug_video_path="outputs/dashboard_indicator_debug.mp4",
        max_frames=3000,
        baseline_frames=80,
        left_ratio_margin=0.0012,
        right_ratio_margin=0.0012,
        left_peak_margin=25.0,
        right_peak_margin=25.0,
        blink_window_frames=15,
        blink_min_on_count=2,
    )

    print(f"\nSaved indicator CSV to: {out_csv}")
    print(df.head())
    print(f"\nTotal processed rows: {len(df)}")