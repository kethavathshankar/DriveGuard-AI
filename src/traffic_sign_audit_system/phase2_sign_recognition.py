import os
import re
from typing import Dict, Any, Optional, Tuple, List

import cv2
import numpy as np
import pandas as pd


# =========================================================
# Optional OCR backends
# =========================================================
_EASYOCR_READER = None
_TESSERACT_AVAILABLE = False


def init_ocr():
    global _EASYOCR_READER, _TESSERACT_AVAILABLE

    try:
        import easyocr
        _EASYOCR_READER = easyocr.Reader(["en"], gpu=False)
        print("[Phase2] EasyOCR loaded.")
    except Exception as e:
        _EASYOCR_READER = None
        print(f"[Phase2] EasyOCR not available: {e}")

    try:
        import pytesseract  # noqa: F401
        _TESSERACT_AVAILABLE = True
        print("[Phase2] Tesseract available.")
    except Exception as e:
        _TESSERACT_AVAILABLE = False
        print(f"[Phase2] Tesseract not available: {e}")


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def safe_read_image(path: str) -> Optional[np.ndarray]:
    if not path or not os.path.exists(path):
        return None
    img = cv2.imread(path)
    return img


def normalize_text(s: str) -> str:
    return re.sub(r"[^A-Z0-9]", "", str(s).upper())


def extract_speed_value(text: str, allowed: Optional[List[int]] = None) -> Optional[int]:
    """
    Extract speed value from OCR text.
    Default allowed list can be edited as needed.
    """
    if allowed is None:
        allowed = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 110, 120]

    if not text:
        return None

    txt = normalize_text(text)

    # direct numeric groups
    nums = re.findall(r"\d{1,3}", txt)
    for n in nums:
        try:
            val = int(n)
            if val in allowed:
                return val
        except Exception:
            pass

    # common OCR confusions
    replacements = {
        "3O": "30",
        "SO": "50",
        "8O": "80",
        "IO": "10",
        "1O": "10",
        "4O": "40",
        "6O": "60",
        "7O": "70",
        "9O": "90",
        "12O": "120",
        "1OO": "100",
        "11O": "110",
    }

    for k, v in replacements.items():
        if k in txt:
            try:
                val = int(v)
                if val in allowed:
                    return val
            except Exception:
                pass

    return None


def preprocess_for_ocr(img_bgr: np.ndarray) -> List[np.ndarray]:
    """
    Return multiple OCR variants.
    """
    out = []

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    out.append(gray)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(gray)
    out.append(clahe)

    blur = cv2.GaussianBlur(clahe, (3, 3), 0)
    out.append(blur)

    _, th1 = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    out.append(th1)

    _, th2 = cv2.threshold(clahe, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    out.append(th2)

    adap = cv2.adaptiveThreshold(
        clahe, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        31, 7
    )
    out.append(adap)

    sharp = cv2.filter2D(
        clahe,
        -1,
        np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]], dtype=np.float32)
    )
    out.append(sharp)

    return out


def run_easyocr_text(img) -> List[Tuple[str, float]]:
    """
    Returns list of (text, conf)
    """
    if _EASYOCR_READER is None:
        return []

    try:
        results = _EASYOCR_READER.readtext(img, detail=1, paragraph=False)
        out = []
        for item in results:
            if len(item) >= 3:
                _, text, conf = item
                out.append((str(text), float(conf)))
        return out
    except Exception:
        return []


def run_tesseract_text(img) -> List[Tuple[str, float]]:
    if not _TESSERACT_AVAILABLE:
        return []

    try:
        import pytesseract
        txt = pytesseract.image_to_string(img, config="--psm 8 -c tessedit_char_whitelist=0123456789STOPNOPARKING")
        txt = txt.strip()
        if txt:
            return [(txt, 0.50)]
    except Exception:
        pass
    return []


def run_ocr_multi(img_bgr: np.ndarray) -> Tuple[str, float, Optional[int]]:
    """
    Multi-variant OCR with best-score selection.
    """
    variants = preprocess_for_ocr(img_bgr)
    candidates: List[Tuple[str, float, Optional[int]]] = []

    for idx, var in enumerate(variants):
        # EasyOCR
        for text, conf in run_easyocr_text(var):
            speed = extract_speed_value(text)
            bonus = 0.10 if speed is not None else 0.0
            bonus += 0.02 * max(0, 3 - idx)
            candidates.append((text, conf + bonus, speed))

        # Tesseract fallback
        for text, conf in run_tesseract_text(var):
            speed = extract_speed_value(text)
            bonus = 0.08 if speed is not None else 0.0
            bonus += 0.01 * max(0, 3 - idx)
            candidates.append((text, conf + bonus, speed))

    if not candidates:
        return "", 0.0, None

    candidates.sort(key=lambda x: x[1], reverse=True)
    best_text, best_conf, best_speed = candidates[0]
    return best_text, float(best_conf), best_speed


def compute_color_features(img_bgr: np.ndarray) -> Dict[str, float]:
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 60, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 60, 50), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    blue = cv2.inRange(hsv, (90, 50, 40), (135, 255, 255))
    white = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))
    yellow = cv2.inRange(hsv, (15, 60, 60), (40, 255, 255))

    total = float(img_bgr.shape[0] * img_bgr.shape[1]) + 1e-6

    return {
        "red_ratio": float(np.sum(red > 0) / total),
        "blue_ratio": float(np.sum(blue > 0) / total),
        "white_ratio": float(np.sum(white > 0) / total),
        "yellow_ratio": float(np.sum(yellow > 0) / total),
    }


def largest_contour_shape_features(img_bgr: np.ndarray) -> Dict[str, float]:
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 150)

    cnts, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "approx_vertices": 0.0,
            "aspect_ratio": 1.0,
        }

    c = max(cnts, key=cv2.contourArea)
    area = float(cv2.contourArea(c))
    peri = float(cv2.arcLength(c, True))
    circularity = 0.0
    if peri > 1e-6:
        circularity = float(4.0 * np.pi * area / (peri * peri))

    approx = cv2.approxPolyDP(c, 0.04 * peri if peri > 0 else 1.0, True)
    x, y, w, h = cv2.boundingRect(c)
    aspect = float(w / max(h, 1))

    return {
        "area": area,
        "perimeter": peri,
        "circularity": circularity,
        "approx_vertices": float(len(approx)),
        "aspect_ratio": aspect,
    }


def detect_red_ring_score(img_bgr: np.ndarray) -> float:
    """
    Heuristic for speed sign look:
    - outer red ring
    - inner white region
    """
    h, w = img_bgr.shape[:2]
    cx, cy = w // 2, h // 2
    radius = min(h, w) // 2

    if radius < 8:
        return 0.0

    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 60, 50), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 60, 50), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    white = cv2.inRange(hsv, (0, 0, 160), (180, 60, 255))

    yy, xx = np.indices((h, w))
    dist = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2) / max(radius, 1)

    ring_mask = (dist >= 0.60) & (dist <= 0.95)
    inner_mask = dist <= 0.55

    ring_red = float(np.mean((red > 0)[ring_mask])) if np.any(ring_mask) else 0.0
    inner_white = float(np.mean((white > 0)[inner_mask])) if np.any(inner_mask) else 0.0

    return 0.6 * ring_red + 0.4 * inner_white


def classify_sign_family(
    crop_bgr: np.ndarray,
    context_bgr: Optional[np.ndarray],
    ocr_text: str,
    ocr_conf: float,
    speed_value: Optional[int],
) -> Dict[str, Any]:
    """
    Simple rule-based classifier for pipeline completion.
    Later you can replace with trained classifier.
    """
    color = compute_color_features(crop_bgr)
    shape = largest_contour_shape_features(crop_bgr)
    ring_score = detect_red_ring_score(crop_bgr)

    red_ratio = color["red_ratio"]
    blue_ratio = color["blue_ratio"]
    white_ratio = color["white_ratio"]
    yellow_ratio = color["yellow_ratio"]
    circularity = shape["circularity"]
    vertices = shape["approx_vertices"]
    aspect = shape["aspect_ratio"]

    txt_norm = normalize_text(ocr_text)

    # 1) Speed sign
    if speed_value is not None and ring_score >= 0.35:
        return {
            "sign_family": "speed_limit",
            "sign_type": f"speed_limit_{speed_value}",
            "speed_value": speed_value,
            "recognition_confidence": float(clamp(0.55 + 0.25 * ocr_conf + 0.20 * ring_score, 0.0, 0.99)),
            "reason": "ocr_number_and_red_ring",
        }

    # 2) Stop sign heuristic
    if "STOP" in txt_norm or (red_ratio > 0.20 and vertices >= 7 and vertices <= 10):
        return {
            "sign_family": "stop",
            "sign_type": "stop",
            "speed_value": None,
            "recognition_confidence": float(clamp(0.55 + 0.20 * red_ratio + 0.10 * (1.0 if "STOP" in txt_norm else 0.0), 0.0, 0.95)),
            "reason": "stop_heuristic",
        }

    # 3) Warning sign: triangular + red/yellow
    if (vertices >= 3 and vertices <= 4 and red_ratio > 0.05) or (yellow_ratio > 0.12 and red_ratio > 0.03):
        return {
            "sign_family": "warning",
            "sign_type": "warning",
            "speed_value": None,
            "recognition_confidence": float(clamp(0.45 + 0.20 * red_ratio + 0.10 * yellow_ratio, 0.0, 0.90)),
            "reason": "warning_triangle_like",
        }

    # 4) Mandatory: blue circular
    if blue_ratio > 0.12 and circularity > 0.45:
        return {
            "sign_family": "mandatory",
            "sign_type": "mandatory",
            "speed_value": None,
            "recognition_confidence": float(clamp(0.45 + 0.25 * blue_ratio + 0.10 * circularity, 0.0, 0.90)),
            "reason": "blue_circular_like",
        }

    # 5) Direction: blue rectangle-ish
    if blue_ratio > 0.10 and (aspect > 1.15 or aspect < 0.85):
        return {
            "sign_family": "direction",
            "sign_type": "direction",
            "speed_value": None,
            "recognition_confidence": float(clamp(0.45 + 0.25 * blue_ratio, 0.0, 0.90)),
            "reason": "blue_rectangular_like",
        }

    # 6) No parking heuristic
    if ("NOPARKING" in txt_norm) or (red_ratio > 0.10 and blue_ratio > 0.05 and circularity > 0.35):
        return {
            "sign_family": "no_parking",
            "sign_type": "no_parking",
            "speed_value": None,
            "recognition_confidence": float(clamp(0.40 + 0.20 * red_ratio + 0.15 * blue_ratio, 0.0, 0.90)),
            "reason": "no_parking_heuristic",
        }

    # 7) Unknown sign but likely valid sign
    likely_sign_score = 0.0
    likely_sign_score += 0.25 if red_ratio > 0.05 else 0.0
    likely_sign_score += 0.25 if blue_ratio > 0.05 else 0.0
    likely_sign_score += 0.20 if white_ratio > 0.10 else 0.0
    likely_sign_score += 0.20 if circularity > 0.35 else 0.0
    likely_sign_score += 0.10 if ocr_conf > 0.30 else 0.0

    if likely_sign_score >= 0.35:
        return {
            "sign_family": "unknown_sign",
            "sign_type": "unknown_sign",
            "speed_value": speed_value,
            "recognition_confidence": float(clamp(0.35 + 0.30 * likely_sign_score, 0.0, 0.85)),
            "reason": "generic_sign_like",
        }

    return {
        "sign_family": "not_sign",
        "sign_type": "not_sign",
        "speed_value": None,
        "recognition_confidence": float(clamp(0.20 + 0.20 * likely_sign_score, 0.0, 0.60)),
        "reason": "weak_sign_evidence",
    }


# =========================================================
# Main Phase 2
# =========================================================
def run_phase2_sign_recognition(
    tracks_csv_path: str,
    output_dir: str = "outputs/traffic_sign_audit_system",
    output_csv_name: str = "traffic_sign_recognition.csv",
    progress_callback=None,
) -> Dict[str, Any]:
    """
    Phase 2:
    - read traffic_sign_tracks.csv
    - load crop + context images
    - OCR
    - simple rule-based sign family recognition
    - write traffic_sign_recognition.csv
    """
    if not os.path.exists(tracks_csv_path):
        raise FileNotFoundError(f"Tracks CSV not found: {tracks_csv_path}")

    ensure_dir(output_dir)
    out_csv = os.path.join(output_dir, output_csv_name)

    df = pd.read_csv(tracks_csv_path)
    if len(df) == 0:
        empty_cols = [
            "track_id", "frame", "time_sec", "timestamp_sec",
            "crop_path", "context_path",
            "sign_family", "sign_type", "speed_value",
            "ocr_text", "ocr_confidence", "recognition_confidence",
            "reason"
        ]
        out_df = pd.DataFrame(columns=empty_cols)
        out_df.to_csv(out_csv, index=False)
        return {
            "recognition_csv": out_csv,
            "recognition_df": out_df,
            "num_rows": 0,
            "message": "No rows in tracks CSV."
        }

    init_ocr()

    rows_out = []
    total = len(df)

    print(f"[Phase2] Reading tracks from: {tracks_csv_path}")
    print(f"[Phase2] Total track rows: {total}")

    for idx, row in df.iterrows():
        track_id = int(row["track_id"])
        frame = int(row["frame"])
        time_sec = float(row["time_sec"]) if "time_sec" in row else float(row.get("timestamp_sec", 0.0))
        timestamp_sec = float(row["timestamp_sec"]) if "timestamp_sec" in row else time_sec
        crop_path = str(row["crop_path"])
        context_path = str(row["context_path"]) if "context_path" in row else ""

        crop_bgr = safe_read_image(crop_path)
        context_bgr = safe_read_image(context_path)

        if crop_bgr is None:
            rows_out.append({
                "track_id": track_id,
                "frame": frame,
                "time_sec": time_sec,
                "timestamp_sec": timestamp_sec,
                "crop_path": crop_path,
                "context_path": context_path,
                "sign_family": "invalid_crop",
                "sign_type": "invalid_crop",
                "speed_value": None,
                "ocr_text": "",
                "ocr_confidence": 0.0,
                "recognition_confidence": 0.0,
                "reason": "crop_not_found",
            })
            if progress_callback is not None:
                progress_callback(idx + 1, total)
            continue

        ocr_text, ocr_conf, speed_value = run_ocr_multi(crop_bgr)

        pred = classify_sign_family(
            crop_bgr=crop_bgr,
            context_bgr=context_bgr,
            ocr_text=ocr_text,
            ocr_conf=ocr_conf,
            speed_value=speed_value,
        )

        rows_out.append({
            "track_id": track_id,
            "frame": frame,
            "time_sec": time_sec,
            "timestamp_sec": timestamp_sec,
            "crop_path": crop_path,
            "context_path": context_path,
            "sign_family": pred["sign_family"],
            "sign_type": pred["sign_type"],
            "speed_value": pred["speed_value"],
            "ocr_text": ocr_text,
            "ocr_confidence": round(float(ocr_conf), 4),
            "recognition_confidence": round(float(pred["recognition_confidence"]), 4),
            "reason": pred["reason"],
        })

        if progress_callback is not None:
            progress_callback(idx + 1, total)

        if (idx + 1) % 100 == 0:
            print(f"[Phase2] processed {idx + 1}/{total}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    return {
        "recognition_csv": out_csv,
        "recognition_df": out_df,
        "num_rows": len(out_df),
        "message": f"Phase 2 complete. Saved {len(out_df)} recognition rows."
    }


# =========================================================
# Standalone runner
# =========================================================
def _progress(done, total):
    print(f"[Phase2] processed={done}/{total}")


if __name__ == "__main__":
    tracks_csv = "outputs/traffic_sign_audit_system/traffic_sign_tracks.csv"

    result = run_phase2_sign_recognition(
        tracks_csv_path=tracks_csv,
        output_dir="outputs/traffic_sign_audit_system",
        output_csv_name="traffic_sign_recognition.csv",
        progress_callback=_progress,
    )

    print("\n[Phase2] Done.")
    print("[Phase2] Recognition CSV :", result["recognition_csv"])
    print("[Phase2] Num rows        :", result["num_rows"])
    print("[Phase2] Message         :", result["message"])