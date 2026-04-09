import cv2

def get_video_info(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0.0
    cap.release()

    return {
        "fps": fps,
        "frame_count": frame_count,
        "duration_sec": duration,
    }


def find_first_qr_time(video_path, max_seconds=20, step_frames=2):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    max_frames = int(max_seconds * fps)

    qr_detector = cv2.QRCodeDetector()
    frame_idx = 0

    while frame_idx < max_frames:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ok, frame = cap.read()
        if not ok:
            break

        h, w = frame.shape[:2]
        crop = frame[int(0.05*h):int(0.95*h), int(0.05*w):int(0.95*w)]

        data, points, _ = qr_detector.detectAndDecode(crop)

        if points is not None and len(points) > 0:
            cap.release()
            return {
                "qr_frame_id": frame_idx,
                "qr_time_sec": frame_idx / fps,
                "qr_data": data
            }

        frame_idx += step_frames

    cap.release()
    raise RuntimeError(f"QR code not found in first {max_seconds}s of {video_path}")


def compute_dash_offset(front_video_path, dash_video_path):
    front_qr = find_first_qr_time(front_video_path)
    dash_qr = find_first_qr_time(dash_video_path)

    offset_sec = front_qr["qr_time_sec"] - dash_qr["qr_time_sec"]

    return {
        "front_qr": front_qr,
        "dash_qr": dash_qr,
        "offset_sec": offset_sec
    }