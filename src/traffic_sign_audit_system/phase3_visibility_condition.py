import os
from typing import Dict, Any, Optional, List, Tuple

import cv2
import numpy as np
import pandas as pd


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def safe_read_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    return cv2.imread(path)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


# =========================================================
# Basic measurable signals
# =========================================================
def compute_blur_score(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def compute_brightness(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.mean(gray))


def compute_contrast(img_bgr: np.ndarray) -> float:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    return float(np.std(gray))


def compute_bbox_area_ratio(
    bbox_w: float,
    bbox_h: float,
    context_img: Optional[np.ndarray]
) -> float:
    if context_img is None:
        return 0.0
    ch, cw = context_img.shape[:2]
    ctx_area = max(1.0, float(cw * ch))
    return float((bbox_w * bbox_h) / ctx_area)


def touches_crop_border(crop_bgr: np.ndarray, pad: int = 3, dark_th: int = 20) -> float:
    """
    Approx edge-cut hint:
    if meaningful pixels are touching crop border strongly, sign may be cut.
    """
    if crop_bgr is None or crop_bgr.size == 0:
        return 0.0

    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape[:2]
    pad = max(1, min(pad, h // 4 if h > 4 else 1, w // 4 if w > 4 else 1))

    top = gray[:pad, :]
    bottom = gray[h - pad:h, :]
    left = gray[:, :pad]
    right = gray[:, w - pad:w]

    # non-background approximation
    top_score = np.mean(top > dark_th)
    bottom_score = np.mean(bottom > dark_th)
    left_score = np.mean(left > dark_th)
    right_score = np.mean(right > dark_th)

    return float(max(top_score, bottom_score, left_score, right_score))


def compute_visible_area_estimate(crop_bgr: np.ndarray) -> float:
    """
    Rough estimate:
    larger structured foreground -> higher visible area.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    _, th = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # choose smaller foreground side
    fg_ratio = float(np.mean(th > 0))
    fg_ratio = min(fg_ratio, 1.0 - fg_ratio)
    return float(clamp(fg_ratio * 2.0, 0.0, 1.0))


def estimate_tilt_angle(crop_bgr: np.ndarray) -> float:
    """
    Rough sign tilt estimate from dominant contour minAreaRect angle.
    """
    gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return 0.0

    c = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(c) < 10:
        return 0.0

    rect = cv2.minAreaRect(c)
    angle = float(rect[-1])

    # normalize angle
    if angle < -45:
        angle = 90 + angle
    if angle > 45:
        angle = angle - 90

    return float(abs(angle))


# =========================================================
# Visibility logic
# =========================================================
def classify_visibility_condition(
    crop_bgr: np.ndarray,
    context_bgr: Optional[np.ndarray],
    bbox_w: float,
    bbox_h: float,
    sign_family: str,
    sign_type: str,
    recognition_confidence: float,
) -> Dict[str, Any]:
    blur_score = compute_blur_score(crop_bgr)
    brightness = compute_brightness(crop_bgr)
    contrast = compute_contrast(crop_bgr)
    area_ratio = compute_bbox_area_ratio(bbox_w, bbox_h, context_bgr)
    edge_cut_score = touches_crop_border(crop_bgr)
    visible_area_est = compute_visible_area_estimate(crop_bgr)
    tilt_angle = estimate_tilt_angle(crop_bgr)

    issues: List[Tuple[str, str, str]] = []

    # -------------------------------------------------
    # Rules
    # -------------------------------------------------
    # very small / far
    if bbox_w < 24 or bbox_h < 24 or area_ratio < 0.015:
        issues.append((
            "too_small_far",
            "high",
            "Sign appears too small/far in the image."
        ))
    elif bbox_w < 40 or bbox_h < 40 or area_ratio < 0.03:
        issues.append((
            "too_small_far",
            "medium",
            "Sign is relatively small and may be difficult to inspect clearly."
        ))

    # blur
    if blur_score < 20:
        issues.append((
            "blurred",
            "high",
            "Sign is strongly blurred and text/symbol visibility is poor."
        ))
    elif blur_score < 45:
        issues.append((
            "blurred",
            "medium",
            "Sign is moderately blurred."
        ))

    # brightness / glare / darkness
    if brightness > 220:
        issues.append((
            "glare_overexposed",
            "high",
            "Sign region is over-bright; glare/overexposure may hide content."
        ))
    elif brightness > 190:
        issues.append((
            "bright_glare",
            "medium",
            "Sign is quite bright and may have partial glare."
        ))

    if brightness < 40:
        issues.append((
            "too_dark",
            "high",
            "Sign region is too dark to inspect properly."
        ))
    elif brightness < 65:
        issues.append((
            "low_light",
            "medium",
            "Sign is under low-light conditions."
        ))

    # low contrast
    if contrast < 18:
        issues.append((
            "low_contrast_or_faded",
            "medium",
            "Contrast is low; sign may look faded or visually weak."
        ))

    # edge cut
    if edge_cut_score > 0.85:
        issues.append((
            "edge_cut",
            "high",
            "Sign touches the crop border strongly and may be partially cut."
        ))
    elif edge_cut_score > 0.65:
        issues.append((
            "edge_cut",
            "medium",
            "Sign is close to crop border and may be partially cut."
        ))

    # visibility estimate
    if visible_area_est < 0.18:
        issues.append((
            "partially_visible",
            "high",
            "Only a limited visible portion of the sign is clearly separable."
        ))
    elif visible_area_est < 0.28:
        issues.append((
            "partially_visible",
            "medium",
            "Visible sign area appears limited."
        ))

    # tilt
    if tilt_angle > 20:
        issues.append((
            "tilted",
            "medium",
            f"Sign appears tilted by about {tilt_angle:.1f} degrees."
        ))
    elif tilt_angle > 10:
        issues.append((
            "slightly_tilted",
            "low",
            f"Sign appears slightly tilted by about {tilt_angle:.1f} degrees."
        ))

    # weak recognition can support unreadable
    if recognition_confidence < 0.20:
        issues.append((
            "unreadable",
            "high",
            "Recognition confidence is very low; sign content may be unreadable."
        ))
    elif recognition_confidence < 0.35:
        issues.append((
            "weak_readability",
            "medium",
            "Recognition confidence is low; readability may be limited."
        ))

    # -------------------------------------------------
    # Final decision
    # -------------------------------------------------
    if not issues:
        problem_type = "clear_visible"
        severity = "none"
        explanation = "Sign appears clearly visible."
        is_clearly_visible = True
    else:
        severity_rank = {"none": 0, "low": 1, "medium": 2, "high": 3}

        # choose highest severity issue as primary
        issues_sorted = sorted(issues, key=lambda x: severity_rank.get(x[1], 0), reverse=True)
        problem_type, severity, explanation = issues_sorted[0]
        is_clearly_visible = False

        # merge short explanation if multiple
        if len(issues_sorted) > 1:
            extra = [x[0] for x in issues_sorted[1:3]]
            explanation = explanation + " Additional issues: " + ", ".join(extra) + "."

    return {
        "problem_type": problem_type,
        "severity": severity,
        "is_clearly_visible": bool(is_clearly_visible),
        "explanation": explanation,
        "blur_score": round(float(blur_score), 4),
        "brightness": round(float(brightness), 4),
        "contrast": round(float(contrast), 4),
        "bbox_area_ratio": round(float(area_ratio), 6),
        "edge_cut_score": round(float(edge_cut_score), 4),
        "visible_area_estimate": round(float(visible_area_est), 4),
        "tilt_angle_estimate": round(float(tilt_angle), 4),
    }


# =========================================================
# Main Phase 3
# =========================================================
def run_phase3_visibility_condition(
    recognition_csv_path: str,
    output_dir: str = "outputs/traffic_sign_audit_system",
    output_csv_name: str = "traffic_sign_visibility.csv",
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Phase 3:
    - read recognition CSV
    - analyze crop + context visibility
    - write traffic_sign_visibility.csv
    """
    if not os.path.exists(recognition_csv_path):
        raise FileNotFoundError(f"Recognition CSV not found: {recognition_csv_path}")

    ensure_dir(output_dir)
    out_csv = os.path.join(output_dir, output_csv_name)

    df = pd.read_csv(recognition_csv_path)

    if len(df) == 0:
        empty_cols = [
            "track_id", "frame", "time_sec", "timestamp_sec",
            "crop_path", "context_path",
            "sign_family", "sign_type", "speed_value",
            "problem_type", "severity", "is_clearly_visible", "explanation",
            "blur_score", "brightness", "contrast",
            "bbox_area_ratio", "edge_cut_score", "visible_area_estimate", "tilt_angle_estimate"
        ]
        out_df = pd.DataFrame(columns=empty_cols)
        out_df.to_csv(out_csv, index=False)
        return {
            "visibility_csv": out_csv,
            "visibility_df": out_df,
            "num_rows": 0,
            "message": "No rows in recognition CSV."
        }

    rows_out = []
    total = len(df)

    print(f"[Phase3] Reading recognition CSV: {recognition_csv_path}")
    print(f"[Phase3] Total rows: {total}")

    for idx, row in df.iterrows():
        track_id = int(row["track_id"])
        frame = int(row["frame"])
        time_sec = float(row["time_sec"]) if "time_sec" in row else float(row.get("timestamp_sec", 0.0))
        timestamp_sec = float(row["timestamp_sec"]) if "timestamp_sec" in row else time_sec

        crop_path = str(row["crop_path"]) if "crop_path" in row else ""
        context_path = str(row["context_path"]) if "context_path" in row else ""

        crop_bgr = safe_read_image(crop_path)
        context_bgr = safe_read_image(context_path)

        sign_family = str(row["sign_family"]) if "sign_family" in row else "unknown_sign"
        sign_type = str(row["sign_type"]) if "sign_type" in row else "unknown_sign"
        speed_value = row["speed_value"] if "speed_value" in row else None
        recognition_confidence = float(row["recognition_confidence"]) if "recognition_confidence" in row else 0.0

        # bbox_w / bbox_h may not exist in recognition CSV, so infer from crop
        if crop_bgr is not None:
            bbox_h, bbox_w = crop_bgr.shape[:2]
        else:
            bbox_w, bbox_h = 0, 0

        if crop_bgr is None:
            rows_out.append({
                "track_id": track_id,
                "frame": frame,
                "time_sec": time_sec,
                "timestamp_sec": timestamp_sec,
                "crop_path": crop_path,
                "context_path": context_path,
                "sign_family": sign_family,
                "sign_type": sign_type,
                "speed_value": speed_value,
                "problem_type": "invalid_crop",
                "severity": "high",
                "is_clearly_visible": False,
                "explanation": "Crop image not found.",
                "blur_score": 0.0,
                "brightness": 0.0,
                "contrast": 0.0,
                "bbox_area_ratio": 0.0,
                "edge_cut_score": 0.0,
                "visible_area_estimate": 0.0,
                "tilt_angle_estimate": 0.0,
            })
            if progress_callback is not None:
                progress_callback(idx + 1, total)
            continue

        vis = classify_visibility_condition(
            crop_bgr=crop_bgr,
            context_bgr=context_bgr,
            bbox_w=bbox_w,
            bbox_h=bbox_h,
            sign_family=sign_family,
            sign_type=sign_type,
            recognition_confidence=recognition_confidence,
        )

        rows_out.append({
            "track_id": track_id,
            "frame": frame,
            "time_sec": time_sec,
            "timestamp_sec": timestamp_sec,
            "crop_path": crop_path,
            "context_path": context_path,
            "sign_family": sign_family,
            "sign_type": sign_type,
            "speed_value": speed_value,
            "problem_type": vis["problem_type"],
            "severity": vis["severity"],
            "is_clearly_visible": vis["is_clearly_visible"],
            "explanation": vis["explanation"],
            "blur_score": vis["blur_score"],
            "brightness": vis["brightness"],
            "contrast": vis["contrast"],
            "bbox_area_ratio": vis["bbox_area_ratio"],
            "edge_cut_score": vis["edge_cut_score"],
            "visible_area_estimate": vis["visible_area_estimate"],
            "tilt_angle_estimate": vis["tilt_angle_estimate"],
        })

        if progress_callback is not None:
            progress_callback(idx + 1, total)

        if (idx + 1) % 100 == 0:
            print(f"[Phase3] processed {idx + 1}/{total}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    return {
        "visibility_csv": out_csv,
        "visibility_df": out_df,
        "num_rows": len(out_df),
        "message": f"Phase 3 complete. Saved {len(out_df)} visibility rows."
    }


# =========================================================
# Standalone runner
# =========================================================
def _progress(done, total):
    print(f"[Phase3] processed={done}/{total}")


if __name__ == "__main__":
    recognition_csv = "outputs/traffic_sign_audit_system/traffic_sign_recognition.csv"

    result = run_phase3_visibility_condition(
        recognition_csv_path=recognition_csv,
        output_dir="outputs/traffic_sign_audit_system",
        output_csv_name="traffic_sign_visibility.csv",
        progress_callback=_progress,
    )

    print("\n[Phase3] Done.")
    print("[Phase3] Visibility CSV :", result["visibility_csv"])
    print("[Phase3] Num rows       :", result["num_rows"])
    print("[Phase3] Message        :", result["message"])