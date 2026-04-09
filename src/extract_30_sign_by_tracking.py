import os
import cv2
from pathlib import Path

VIDEO_PATH = "data/front.mp4"
OUT_ROOT = Path("data/speed_classification")
TRAIN_DIR = OUT_ROOT / "train" / "30"
VAL_DIR = OUT_ROOT / "val" / "30"

VAL_EVERY_N = 5
PAD_RATIO = 0.12


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


def expand_box(x, y, w, h, W, H, pad_ratio=0.12):
    px = int(w * pad_ratio)
    py = int(h * pad_ratio)
    return clamp_box(x - px, y - py, x + w + px, y + h + py, W, H)


def save_crop(frame, bbox, frame_id, save_idx):
    x, y, w, h = map(int, bbox)
    H, W = frame.shape[:2]
    x1, y1, x2, y2 = expand_box(x, y, w, h, W, H, PAD_RATIO)
    crop = frame[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return False

    out_dir = VAL_DIR if (save_idx % VAL_EVERY_N == 0) else TRAIN_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"30_frame_{frame_id:06d}_{save_idx:04d}.png"
    cv2.imwrite(str(out_path), crop)
    return True


def main():
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {VIDEO_PATH}")

    frame_id = 0
    save_idx = 0

    print("\nControls:")
    print("n = next frame")
    print("k = skip 5 frames")
    print("b = draw box and save crop")
    print("q = quit\n")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        disp = frame.copy()
        cv2.putText(
            disp,
            f"frame: {frame_id} | n=next, k=skip5, b=box+save, q=quit",
            (20, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.75,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

        cv2.imshow("Manual 30 crop tool", disp)
        key = cv2.waitKey(0) & 0xFF

        if key in [ord("q"), ord("Q"), 27]:
            break

        elif key in [ord("n"), ord("N")]:
            frame_id += 1
            continue

        elif key in [ord("k"), ord("K")]:
            for _ in range(5):
                ok2, _ = cap.read()
                if not ok2:
                    ok = False
                    break
                frame_id += 1
            if not ok:
                break
            frame_id += 1
            continue

        elif key in [ord("b"), ord("B")]:
            bbox = cv2.selectROI("Draw 30 sign box", frame, fromCenter=False, showCrosshair=True)
            x, y, w, h = map(int, bbox)
            cv2.destroyWindow("Draw 30 sign box")

            if w > 0 and h > 0:
                save_idx += 1
                saved = save_crop(frame, bbox, frame_id, save_idx)
                if saved:
                    print(f"Saved crop {save_idx} from frame {frame_id}")
                else:
                    print(f"Failed to save crop from frame {frame_id}")

            frame_id += 1
            continue

        else:
            frame_id += 1
            continue

    cap.release()
    cv2.destroyAllWindows()

    print(f"\nDone. Saved {save_idx} crops.")
    print(f"Train dir: {TRAIN_DIR}")
    print(f"Val dir:   {VAL_DIR}")


if __name__ == "__main__":
    main()