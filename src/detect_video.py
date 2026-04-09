import argparse
from ultralytics import YOLO
import cv2
from tqdm import tqdm

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--video", required=True, help="Path to input mp4")
    ap.add_argument("--out", default="outputs/annotated.mp4", help="Path to output mp4")
    ap.add_argument("--model", default="yolov8n.pt", help="YOLO model: yolov8n.pt / yolov8s.pt ...")
    ap.add_argument("--conf", type=float, default=0.35)
    args = ap.parse_args()

    model = YOLO(args.model)

    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out, fourcc, fps, (w, h))

    pbar = tqdm(total=nframes, desc="Detecting")
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # YOLO inference
        results = model.predict(
    frame,
    conf=args.conf,
    verbose=False,
    device="mps",
    imgsz=640
)
        annotated = results[0].plot()

        out.write(annotated)
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print(f"✅ Saved: {args.out}")

if __name__ == "__main__":
    main()