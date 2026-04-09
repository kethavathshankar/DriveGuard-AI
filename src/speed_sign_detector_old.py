import os
import json
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.models import resnet18
from ultralytics import YOLO


DEFAULT_DETECTOR_MODEL_PATH = "models/speed_sign_detector.pt"
DEFAULT_CLASSIFIER_MODEL_PATH = "models/best_classifier.pth"
DEFAULT_CLASS_TO_IDX_PATH = "models/class_to_idx.json"

TARGET_SPEED = 30
NEGATIVE_CLASS_NAME = "not_speed_sign"


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


def box_area(box):
    x1, y1, x2, y2 = box
    return max(1, x2 - x1) * max(1, y2 - y1)


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

    union = box_area(a) + box_area(b) - inter
    return float(inter) / float(max(union, 1))


def expand_box(box, w, h, margin_ratio=0.18):
    x1, y1, x2, y2 = box
    bw = x2 - x1
    bh = y2 - y1
    mx = int(bw * margin_ratio)
    my = int(bh * margin_ratio)
    return clamp_box(x1 - mx, y1 - my, x2 + mx, y2 + my, w, h)


def detect_fallback_sign_candidates(frame):
    """
    Fallback candidate search for circular red/white roadside speed signs.
    Returns list of (box_xyxy, pseudo_conf).
    """
    H, W = frame.shape[:2]

    x_start = int(W * 0.55)
    x_end = W
    y_start = int(H * 0.10)
    y_end = int(H * 0.75)

    roi = frame[y_start:y_end, x_start:x_end]
    if roi is None or roi.size == 0:
        return []

    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 60, 60), (12, 255, 255))
    red2 = cv2.inRange(hsv, (160, 60, 60), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    white = cv2.inRange(hsv, (0, 0, 150), (180, 80, 255))
    mask = cv2.bitwise_or(red, white)

    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    out = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 60:
            continue

        x, y, w, h = cv2.boundingRect(cnt)

        if w < 12 or h < 12:
            continue
        if w > 180 or h > 180:
            continue

        ar = w / float(h + 1e-6)
        if ar < 0.65 or ar > 1.35:
            continue

        peri = cv2.arcLength(cnt, True)
        if peri <= 0:
            continue

        circularity = 4.0 * np.pi * area / (peri * peri + 1e-6)
        if circularity < 0.35:
            continue

        x1 = x_start + x
        y1 = y_start + y
        x2 = x1 + w
        y2 = y1 + h

        x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

        pseudo_conf = min(0.99, 0.40 + circularity * 0.5)
        out.append(((x1, y1, x2, y2), float(pseudo_conf)))

    return out


def load_class_mapping(json_path):
    with open(json_path, "r") as f:
        class_to_idx = json.load(f)
    idx_to_class = {int(v): str(k) for k, v in class_to_idx.items()}
    return class_to_idx, idx_to_class


def build_classifier_model(num_classes):
    model = resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


class SpeedClassifier:
    def __init__(self, model_path, class_to_idx_path, device="cpu"):
        self.device = torch.device(device)
        self.class_to_idx, self.idx_to_class = load_class_mapping(class_to_idx_path)

        self.model = build_classifier_model(len(self.class_to_idx))
        state = torch.load(model_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state:
            state = state["state_dict"]

        cleaned = {k.replace("module.", ""): v for k, v in state.items()}
        self.model.load_state_dict(cleaned, strict=True)
        self.model.to(self.device)
        self.model.eval()

        self.tf = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    @torch.no_grad()
    def predict(self, crop_bgr):
        if crop_bgr is None or crop_bgr.size == 0:
            return None, 0.0, "invalid"

        rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
        x = self.tf(rgb).unsqueeze(0).to(self.device)

        logits = self.model(x)
        probs = torch.softmax(logits, dim=1)
        conf, idx = torch.max(probs, dim=1)

        idx = int(idx.item())
        conf = float(conf.item())
        label = self.idx_to_class[idx]

        if label == NEGATIVE_CLASS_NAME:
            return None, conf, label

        if label == "30":
            return 30, conf, label

        return None, conf, label


@dataclass
class DetectionItem:
    frame: int
    time_sec: float
    box: Tuple[int, int, int, int]
    det_conf: float
    cls_speed: Optional[int]
    cls_conf: float
    cls_label: str
    crop_path: str = ""


@dataclass
class SignTrack:
    track_id: int
    start_frame: int
    last_frame: int
    boxes: List[Tuple[int, int, int, int]] = field(default_factory=list)
    items: List[DetectionItem] = field(default_factory=list)

    def add(self, item):
        self.last_frame = item.frame
        self.boxes.append(item.box)
        self.items.append(item)


def assign_detection_to_track(det_box, tracks, frame_id, max_gap_frames=12, min_iou=0.12):
    best_idx = None
    best_iou = 0.0

    for idx, tr in enumerate(tracks):
        if frame_id - tr.last_frame > max_gap_frames:
            continue
        ov = iou_xyxy(det_box, tr.boxes[-1])
        if ov >= min_iou and ov > best_iou:
            best_iou = ov
            best_idx = idx

    return best_idx


def stable_track_decision(track, min_repeat=2, min_cls_conf=0.55):
    positives = [it for it in track.items if it.cls_speed == 30 and it.cls_conf >= min_cls_conf]
    if len(positives) < min_repeat:
        return None

    first_positive = min(it.frame for it in positives)
    best_conf = max(it.cls_conf for it in positives)

    return {
        "frame": int(track.start_frame),
        "speed_limit": 30,
        "ocr_confidence": float(best_conf),
        "track_start_frame": int(track.start_frame),
        "track_end_frame": int(track.last_frame),
        "first_positive_frame": int(first_positive),
        "track_id": int(track.track_id),
    }


def run_speed_sign_detector(
    front_video_path,
    model_path="",
    out_raw_csv_path="outputs/speed_sign_raw_detections.csv",
    out_events_csv_path="outputs/speed_sign_events.csv",
    out_debug_video_path="outputs/speed_sign_debug.mp4",
    crops_dir="outputs/speed_sign_crops",
    process_every=1,
    max_frames=None,
    conf_thres=0.10,
    save_crops=True,
    min_repeat=2,
    max_gap_frames=12,
    min_box_width=10,
    min_box_height=10,
    progress_callback=None,
    start_frame=0,
    use_tesseract=False,
    detector_model_path=DEFAULT_DETECTOR_MODEL_PATH,
    classifier_model_path=DEFAULT_CLASSIFIER_MODEL_PATH,
    class_to_idx_path=DEFAULT_CLASS_TO_IDX_PATH,
    detector_imgsz=960,
    classifier_min_conf=0.55,
    debug_draw=True,
    device="cpu",
    **kwargs
):
    os.makedirs(os.path.dirname(out_raw_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_events_csv_path) or ".", exist_ok=True)
    os.makedirs(os.path.dirname(out_debug_video_path) or ".", exist_ok=True)
    if save_crops:
        os.makedirs(crops_dir, exist_ok=True)

    detector = YOLO(detector_model_path)
    classifier = SpeedClassifier(classifier_model_path, class_to_idx_path, device=device)

    cap = cv2.VideoCapture(front_video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {front_video_path}")

    fps = float(cap.get(cv2.CAP_PROP_FPS))
    total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    start_frame = max(0, int(start_frame))
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    end_frame = total_video_frames if max_frames is None else min(total_video_frames, int(max_frames))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    debug_fps = max(1.0, fps / max(1, process_every))
    writer = cv2.VideoWriter(out_debug_video_path, fourcc, debug_fps, (W, H))

    tracks = []
    next_track_id = 1
    raw_rows = []

    frame_id = start_frame
    processed_counter = 0
    total_frames = max(0, end_frame - start_frame)

    while True:
        ok, frame = cap.read()
        if not ok or frame_id >= end_frame:
            break

        if (frame_id - start_frame) % process_every != 0:
            frame_id += 1
            continue

        time_sec = frame_id / fps
        debug = frame.copy()

        results = detector.predict(
            source=frame,
            conf=conf_thres,
            imgsz=detector_imgsz,
            device=device,
            verbose=False
        )

        boxes = []
        r = results[0]
        if r.boxes is not None and len(r.boxes) > 0:
            xyxy = r.boxes.xyxy.cpu().numpy()
            confs = r.boxes.conf.cpu().numpy()
            for box, dconf in zip(xyxy, confs):
                x1, y1, x2, y2 = map(int, box.tolist())
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)
                if (x2 - x1) < min_box_width or (y2 - y1) < min_box_height:
                    continue
                boxes.append(((x1, y1, x2, y2), float(dconf)))

        if len(boxes) == 0:
            boxes = detect_fallback_sign_candidates(frame)

        for box, dconf in boxes:
            crop_box = expand_box(box, W, H, margin_ratio=0.18)
            cx1, cy1, cx2, cy2 = crop_box
            crop = frame[cy1:cy2, cx1:cx2]
            if crop is None or crop.size == 0:
                continue

            cls_speed, cls_conf, cls_label = classifier.predict(crop)

            tr_idx = assign_detection_to_track(
                box,
                tracks,
                frame_id,
                max_gap_frames=max_gap_frames,
                min_iou=0.12
            )

            item = DetectionItem(
                frame=frame_id,
                time_sec=time_sec,
                box=box,
                det_conf=dconf,
                cls_speed=cls_speed,
                cls_conf=cls_conf,
                cls_label=cls_label,
            )

            if tr_idx is None:
                tr = SignTrack(
                    track_id=next_track_id,
                    start_frame=frame_id,
                    last_frame=frame_id
                )
                tr.add(item)
                tracks.append(tr)
                track_id = next_track_id
                next_track_id += 1
            else:
                tracks[tr_idx].add(item)
                track_id = tracks[tr_idx].track_id

            crop_path = ""
            if save_crops:
                crop_path = os.path.join(
                    crops_dir,
                    f"track_{track_id:04d}_frame_{frame_id:06d}.png"
                )
                cv2.imwrite(crop_path, crop)

            raw_rows.append({
                "frame": int(frame_id),
                "time_sec": float(time_sec),
                "det_confidence": float(dconf),
                "x1": int(box[0]),
                "y1": int(box[1]),
                "x2": int(box[2]),
                "y2": int(box[3]),
                "class_name": "speed_limit_sign_candidate",
                "crop_path": crop_path,
                "ocr_speed_limit": cls_speed,
                "ocr_confidence": float(cls_conf),
                "ocr_text": str(cls_label),
                "ocr_variant": "detector+binary_classifier",
                "track_id": int(track_id),
            })

            if debug_draw:
                color = (0, 255, 0) if cls_speed == 30 else (0, 255, 255)
                cv2.rectangle(debug, (box[0], box[1]), (box[2], box[3]), color, 2)
                txt = f"{cls_label} {cls_conf:.2f}"
                cv2.putText(
                    debug,
                    txt,
                    (box[0], max(20, box[1] - 5)),
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
                "Detector + binary classifier"
            )

        frame_id += 1

    cap.release()
    writer.release()

    raw_df = pd.DataFrame(raw_rows)
    if len(raw_df) == 0:
        raw_df = pd.DataFrame(columns=[
            "frame", "time_sec", "det_confidence", "x1", "y1", "x2", "y2",
            "class_name", "crop_path", "ocr_speed_limit", "ocr_confidence",
            "ocr_text", "ocr_variant", "track_id"
        ])
    raw_df.to_csv(out_raw_csv_path, index=False)

    event_rows = []
    for tr in tracks:
        ev = stable_track_decision(tr, min_repeat=min_repeat, min_cls_conf=classifier_min_conf)
        if ev is not None:
            event_rows.append(ev)

    events_df = pd.DataFrame(event_rows)
    if len(events_df) > 0:
        events_df = (
            events_df.sort_values(["frame", "ocr_confidence"], ascending=[True, False])
            .drop_duplicates(subset=["frame", "speed_limit"], keep="first")
            .reset_index(drop=True)
        )
        events_df["time_sec"] = events_df["frame"].astype(float) / float(fps)
    else:
        events_df = pd.DataFrame(columns=[
            "frame", "time_sec", "speed_limit", "ocr_confidence",
            "track_start_frame", "track_end_frame", "first_positive_frame", "track_id"
        ])

    events_df.to_csv(out_events_csv_path, index=False)

    return raw_df, events_df


if __name__ == "__main__":
    raw_df, events_df = run_speed_sign_detector(
        front_video_path="data/front.mp4",
        detector_model_path="models/speed_sign_detector.pt",
        classifier_model_path="models/best_classifier.pth",
        class_to_idx_path="models/class_to_idx.json",
        process_every=1,
        conf_thres=0.10,
        min_repeat=2,
        max_gap_frames=12,
        classifier_min_conf=0.55,
        device="cpu",
    )

    print("Raw detections:", len(raw_df))
    print("Stable events:", len(events_df))