import os
import re
import cv2
import numpy as np
import pandas as pd
import easyocr
from collections import defaultdict

VALID_SPEEDS = {10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 120}


# -----------------------------
# HELPERS
# -----------------------------
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


def normalize_text(text):
    s = str(text).strip().upper()
    s = s.replace("O", "0").replace("I", "1").replace("L", "1").replace("S", "5").replace("B", "8")
    s = re.sub(r"[^0-9]", "", s)
    return s


def score_crop_quality(crop_bgr):
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    contrast = float(gray.std())
    h, w = gray.shape[:2]
    size_score = float(h * w)
    return 0.60 * sharpness + 0.35 * contrast + 0.0003 * size_score


# -----------------------------
# ROI FILTER
# -----------------------------
def get_roi(frame):
    h, w = frame.shape[:2]

    # right side, upper/middle part where roadside speed signs usually appear
    y1 = int(0.08 * h)
    y2 = int(0.78 * h)
    x1 = int(0.45 * w)
    x2 = int(0.98 * w)

    roi = frame[y1:y2, x1:x2]
    return roi, (x1, y1, x2, y2)


# -----------------------------
# RED + WHITE CANDIDATE DETECTOR
# -----------------------------
def detect_candidates(roi_bgr):
    hsv = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
    red2 = cv2.inRange(hsv, (160, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    white = cv2.inRange(hsv, (0, 0, 150), (180, 90, 255))

    mask = cv2.bitwise_or(red, white)
    mask = cv2.medianBlur(mask, 5)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 20:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 15 or h < 15:
            continue
        if w > 150 or h > 150:
            continue

        ar = w / float(max(h, 1))
        if ar < 0.60 or ar > 1.40:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue
        circularity = 4.0 * np.pi * area / (peri * peri)
        if circularity < 0.25:
            continue

        pad = int(0.18 * max(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(roi_bgr.shape[1] - 1, x + w + pad)
        y2 = min(roi_bgr.shape[0] - 1, y + h + pad)

        cand = roi_bgr[y1:y2, x1:x2]
        if cand.size == 0:
            continue

        cand_hsv = cv2.cvtColor(cand, cv2.COLOR_BGR2HSV)
        cand_red1 = cv2.inRange(cand_hsv, (0, 60, 60), (12, 255, 255))
        cand_red2 = cv2.inRange(cand_hsv, (160, 60, 60), (180, 255, 255))
        cand_red = cv2.bitwise_or(cand_red1, cand_red2)
        cand_white = cv2.inRange(cand_hsv, (0, 0, 150), (180, 90, 255))

        red_ratio = float(np.mean(cand_red > 0))
        white_ratio = float(np.mean(cand_white > 0))

        if red_ratio < 0.03 or white_ratio < 0.10:
            continue

        score = 0.55 * circularity + 0.25 * red_ratio + 0.20 * white_ratio
        boxes.append((x1, y1, x2, y2, score))

    boxes = sorted(boxes, key=lambda z: z[4], reverse=True)
    return boxes[:8]


# -----------------------------
# OCR
# -----------------------------
def read_speed(crop_bgr, reader):
    if crop_bgr is None or crop_bgr.size == 0:
        return None, 0.0, ""

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.resize(gray, None, fx=2.5, fy=2.5, interpolation=cv2.INTER_CUBIC)

    variants = []

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    sharp = cv2.addWeighted(clahe, 1.8, blur, -0.8, 0)

    _, otsu = cv2.threshold(sharp, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu_inv = cv2.bitwise_not(otsu)

    variants.append(("gray", gray))
    variants.append(("clahe", clahe))
    variants.append(("sharp", sharp))
    variants.append(("otsu", otsu))
    variants.append(("otsu_inv", otsu_inv))

    hits = []
    for variant_name, img in variants:
        try:
            results = reader.readtext(img, allowlist="0123456789", detail=1, paragraph=False)
        except Exception:
            results = []

        for item in results:
            if len(item) < 3:
                continue
            text = normalize_text(item[1])
            conf = float(item[2])

            if not text:
                continue

            try:
                val = int(text)
            except Exception:
                continue

            if val in VALID_SPEEDS and conf >= 0.35:
                bonus = 0.0
                if variant_name in {"sharp", "otsu", "otsu_inv"}:
                    bonus += 0.05
                hits.append((val, min(0.99, conf + bonus), text, variant_name))

    if not hits:
        return None, 0.0, ""

    vote_map = defaultdict(float)
    best_hit = {}

    for val, conf, text, variant_name in hits:
        vote_map[val] += conf
        if val not in best_hit or conf > best_hit[val][0]:
            best_hit[val] = (conf, text, variant_name)

    ranked = sorted(vote_map.items(), key=lambda z: z[1], reverse=True)
    best_speed, best_vote = ranked[0]
    second_vote = ranked[1][1] if len(ranked) > 1 else 0.0

    # reject confusing OCR
    if second_vote > 0 and best_vote < second_vote * 1.10:
        return None, 0.0, ""

    best_conf, best_text, best_variant = best_hit[best_speed]
    return best_speed, float(best_conf), f"{best_text}|{best_variant}"


# -----------------------------
# SIMPLE TRACK / CONFIRMATION
# -----------------------------
class CandidateTrack:
    def __init__(self, track_id, frame_id, box):
        self.track_id = track_id
        self.start_frame = frame_id
        self.last_frame = frame_id
        self.last_box = box
        self.frames = [frame_id]
        self.boxes = [box]
        self.speed_votes = defaultdict(float)
        self.speed_counts = defaultdict(int)
        self.best_conf = defaultdict(float)
        self.first_positive_frame = {}

    def update(self, frame_id, box, speed, conf):
        self.last_frame = frame_id
        self.last_box = box
        self.frames.append(frame_id)
        self.boxes.append(box)

        if speed is not None:
            self.speed_votes[speed] += conf
            self.speed_counts[speed] += 1
            self.best_conf[speed] = max(self.best_conf[speed], conf)
            if speed not in self.first_positive_frame:
                self.first_positive_frame[speed] = frame_id


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b

    ix1 = max(ax1, bx1)
    iy1 = max(ay1, by1)
    ix2 = min(ax2, bx2)
    iy2 = min(ay2, by2)

    iw = max(0, ix2 - ix1)
    ih = max(0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0

    area_a = max(1, ax2 - ax1) * max(1, ay2 - ay1)
    area_b = max(1, bx2 - bx1) * max(1, by2 - by1)
    union = area_a + area_b - inter
    return float(inter) / float(max(union, 1))


def assign_to_track(box, tracks, frame_id, max_gap_frames=12, min_iou=0.12):
    best_idx = None
    best_iou = 0.0
    for idx, tr in enumerate(tracks):
        if frame_id - tr.last_frame > max_gap_frames:
            continue
        ov = iou_xyxy(box, tr.last_box)
        if ov >= min_iou and ov > best_iou:
            best_iou = ov
            best_idx = idx
    return best_idx


# -----------------------------
# MAIN
# -----------------------------
def run_speed_sign_detector(
    front_video_path,
    model_path=None,                    # kept for dashboard compatibility
    out_raw_csv_path="outputs/speed_sign_raw_detections.csv",
    out_events_csv_path="outputs/speed_sign_events.csv",
    out_debug_video_path="outputs/speed_sign_debug.mp4",
    crops_dir="outputs/speed_sign_crops",
    process_every=2,
    max_frames=None,
    conf_thres=0.20,                    # unused, kept for compatibility
    save_crops=True,
    min_repeat=2,
    max_gap_frames=25,
    min_box_width=12,                   # unused, kept for compatibility
    min_box_height=12,                  # unused, kept for compatibility
    progress_callback=None,
    start_frame=0,
    use_tesseract=False,                # unused, kept for compatibility
    **kwargs
):
    if not os.path.exists(front_video_path):
        raise FileNotFoundError(f"Video not found: {front_video_path}")

    os.makedirs(os.path.dirname(out_raw_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_events_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_debug_video_path) or ".", exist_ok=True)
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    cap = cv2.VideoCapture(front_video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {front_video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if fps <= 0 or W <= 0 or H <= 0:
        cap.release()
        raise ValueError("Invalid video metadata.")

    process_every = max(1, int(process_every))
    start_frame = max(0, int(start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    end_frame = total_video_frames if max_frames is None else min(total_video_frames, int(max_frames))
    total_frames = max(0, end_frame - start_frame)

    reader = easyocr.Reader(["en"], gpu=False)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    debug_fps = max(1.0, fps / process_every)
    writer = cv2.VideoWriter(out_debug_video_path, fourcc, debug_fps, (W, H))

    raw_rows = []
    tracks = []
    next_track_id = 1

    frame_id = start_frame
    processed_counter = 0

    if progress_callback is not None:
        progress_callback(0, total_frames, 0, "Stage 1: Fast ROI sign detection + OCR")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_id >= end_frame:
            break

        if (frame_id - start_frame) % process_every != 0:
            frame_id += 1
            continue

        time_sec = frame_id / fps
        debug = frame.copy()

        roi, (roi_x1, roi_y1, roi_x2, roi_y2) = get_roi(frame)
        candidates = detect_candidates(roi)

        for cand_idx, (x1r, y1r, x2r, y2r, cand_score) in enumerate(candidates):
            x1 = roi_x1 + x1r
            y1 = roi_y1 + y1r
            x2 = roi_x1 + x2r
            y2 = roi_y1 + y2r
            x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

            crop = frame[y1:y2, x1:x2]
            if crop.size == 0:
                continue

            quality = score_crop_quality(crop)

            # reject poor crops early
            if quality < 8:
                speed, conf, ocr_text = None, 0.0, ""
            else:
                speed, conf, ocr_text = read_speed(crop, reader)

            crop_path = ""
            if save_crops:
                crop_path = os.path.join(crops_dir, f"frame_{frame_id:06d}_{cand_idx:02d}.png")
                cv2.imwrite(crop_path, crop)

            box = (x1, y1, x2, y2)
            tr_idx = assign_to_track(box, tracks, frame_id, max_gap_frames=max_gap_frames, min_iou=0.12)

            if tr_idx is None:
                tr = CandidateTrack(next_track_id, frame_id, box)
                tr.update(frame_id, box, speed, conf)
                tracks.append(tr)
                track_id = next_track_id
                next_track_id += 1
            else:
                tracks[tr_idx].update(frame_id, box, speed, conf)
                track_id = tracks[tr_idx].track_id

            raw_rows.append({
                "frame": int(frame_id),
                "time_sec": float(time_sec),
                "det_confidence": float(cand_score),
                "x1": int(x1),
                "y1": int(y1),
                "x2": int(x2),
                "y2": int(y2),
                "class_name": "tracked_speed_sign_candidate",
                "crop_path": crop_path,
                "ocr_speed_limit": speed,
                "ocr_confidence": float(conf),
                "ocr_text": ocr_text,
                "ocr_variant": "fast_roi_ocr",
                "crop_quality": float(quality),
                "track_id": int(track_id),
            })

            color = (0, 255, 0) if speed is not None else (0, 255, 255)
            cv2.rectangle(debug, (x1, y1), (x2, y2), color, 2)
            label = f"{speed if speed is not None else 'cand'} {conf:.2f}"
            cv2.putText(
                debug,
                label,
                (x1, max(20, y1 - 5)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                color,
                2,
                cv2.LINE_AA
            )

        writer.write(debug)
        processed_counter += 1

        if progress_callback is not None and processed_counter % 5 == 0:
            progress_callback(
                frame_id - start_frame,
                total_frames,
                len(raw_rows),
                "Stage 1: Fast ROI sign detection + OCR"
            )

        frame_id += 1

    cap.release()
    writer.release()

    raw_df = pd.DataFrame(raw_rows)
    if len(raw_df) == 0:
        raw_df = pd.DataFrame(columns=[
            "frame", "time_sec", "det_confidence",
            "x1", "y1", "x2", "y2",
            "class_name", "crop_path",
            "ocr_speed_limit", "ocr_confidence",
            "ocr_text", "ocr_variant",
            "crop_quality", "track_id"
        ])
    raw_df.to_csv(out_raw_csv_path, index=False)

    # -----------------------------
    # Stable event generation
    # -----------------------------
    event_rows = []
    for tr in tracks:
        if len(tr.speed_votes) == 0:
            continue

        ranked = sorted(tr.speed_votes.items(), key=lambda z: z[1], reverse=True)
        final_speed, final_vote = ranked[0]
        second_vote = ranked[1][1] if len(ranked) > 1 else 0.0
        final_count = tr.speed_counts.get(final_speed, 0)
        final_conf = tr.best_conf.get(final_speed, 0.0)
        first_positive = tr.first_positive_frame.get(final_speed, tr.start_frame)

        # need repeated confirmation
        if final_count < max(2, int(min_repeat)):
            continue

        # reject confusing tracks
        if second_vote > 0 and final_vote < second_vote * 1.10 and final_conf < 0.85:
            continue

        # back-activate to track start
        event_rows.append({
            "frame": int(tr.start_frame),
            "time_sec": float(tr.start_frame / fps),
            "speed_limit": int(final_speed),
            "ocr_confidence": float(final_conf),
            "track_start_frame": int(tr.start_frame),
            "track_end_frame": int(tr.last_frame),
            "first_positive_frame": int(first_positive),
            "track_id": int(tr.track_id),
        })

    events_df = pd.DataFrame(event_rows)
    if len(events_df) > 0:
        events_df = (
            events_df.sort_values(["frame", "ocr_confidence"], ascending=[True, False])
            .drop_duplicates(subset=["frame", "speed_limit"], keep="first")
            .reset_index(drop=True)
        )

        filtered = []
        last_frame = -10**9
        last_speed = None
        min_sep = max(20, process_every * 10)

        for _, row in events_df.iterrows():
            fr = int(row["frame"])
            sp = int(row["speed_limit"])
            if last_speed == sp and (fr - last_frame) < min_sep:
                continue
            filtered.append(row)
            last_frame = fr
            last_speed = sp

        events_df = pd.DataFrame(filtered).reset_index(drop=True)
    else:
        events_df = pd.DataFrame(columns=[
            "frame", "time_sec", "speed_limit", "ocr_confidence",
            "track_start_frame", "track_end_frame", "first_positive_frame", "track_id"
        ])

    events_df.to_csv(out_events_csv_path, index=False)

    if progress_callback is not None:
        progress_callback(total_frames, total_frames, len(events_df), f"Finished. Stable sign events: {len(events_df)}")

    print("Detected raw rows:", len(raw_df))
    print("Detected stable events:", len(events_df))
    return raw_df, events_df


# -----------------------------
# RUN
# -----------------------------
if __name__ == "__main__":
    run_speed_sign_detector(
        front_video_path="data/front.mp4",
        out_raw_csv_path="outputs/speed_sign_raw_detections.csv",
        out_events_csv_path="outputs/speed_sign_events.csv",
        out_debug_video_path="outputs/speed_sign_debug.mp4",
        crops_dir="outputs/speed_sign_crops",
        process_every=2,
    )