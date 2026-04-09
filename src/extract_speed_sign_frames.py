import os
import cv2

VIDEO_PATH = "data/front.mp4"
OUT_DIR = "data/speed_detector_finetune/raw_frames"

# Change these after checking where sign appears
START_FRAME = 0
END_FRAME = 1200
STEP = 3   # save every 3rd frame


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("Total frames:", total)

    START = max(0, START_FRAME)
    END = min(END_FRAME, total)

    cap.set(cv2.CAP_PROP_POS_FRAMES, START)

    frame_id = START
    saved = 0

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if frame_id >= END:
            break

        if (frame_id - START) % STEP == 0:
            out_path = os.path.join(OUT_DIR, f"frame_{frame_id:06d}.jpg")
            cv2.imwrite(out_path, frame)
            saved += 1

        frame_id += 1

    cap.release()
    print(f"Saved {saved} frames to {OUT_DIR}")


if __name__ == "__main__":
    main()