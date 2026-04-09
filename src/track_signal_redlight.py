import argparse
import os
import cv2
import numpy as np
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm
from collections import deque


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


def classify_signal_color(roi_bgr: np.ndarray):

    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN", 0.0

    roi = cv2.resize(roi_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0,70,70),(10,255,255))
    red2 = cv2.inRange(hsv,(160,70,70),(180,255,255))
    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv,(15,70,70),(35,255,255))
    green = cv2.inRange(hsv,(40,70,70),(90,255,255))

    r = int(np.sum(red>0))
    y = int(np.sum(yellow>0))
    g = int(np.sum(green>0))

    total = r+y+g

    if total < 30:
        return "UNKNOWN",0.0

    mx = max(r,y,g)
    conf = mx/(total+1e-6)

    if mx == r:
        return "RED",float(conf)
    if mx == y:
        return "YELLOW",float(conf)

    return "GREEN",float(conf)


def main():

    ap = argparse.ArgumentParser()

    ap.add_argument("--video",required=True)
    ap.add_argument("--telemetry",required=True)

    ap.add_argument("--out",default="outputs/tracked_signal_redlight.mp4")
    ap.add_argument("--tracks_csv",default="outputs/tracks.csv")
    ap.add_argument("--signal_csv",default="outputs/signal.csv")
    ap.add_argument("--violations_csv",default="outputs/violations_redlight.csv")

    ap.add_argument("--model",default="yolo26x.pt")
    ap.add_argument("--conf",type=float,default=0.35)
    ap.add_argument("--imgsz",type=int,default=960)

    ap.add_argument("--start_frame",type=int,default=0)
    ap.add_argument("--max_frames",type=int,default=5400)

    ap.add_argument("--min_tl_area",type=float,default=60.0)

    ap.add_argument("--red_hold_frames",type=int,default=12)
    ap.add_argument("--min_speed_kmh",type=float,default=3.0)
    ap.add_argument("--near_tl_h_ratio",type=float,default=0.035)
    ap.add_argument("--cooldown_sec",type=float,default=8.0)

    args = ap.parse_args()

    # CREATE DATASET FOLDER
    dataset_dir = "data/tl_dataset"
    os.makedirs(f"{dataset_dir}/red",exist_ok=True)
    os.makedirs(f"{dataset_dir}/yellow",exist_ok=True)
    os.makedirs(f"{dataset_dir}/green",exist_ok=True)
    os.makedirs(f"{dataset_dir}/unknown",exist_ok=True)

    print("Dataset folder ready")

    df = pd.read_csv(args.telemetry)
    spd_mps = pick_speed_column(df)
    spd_kmh = np.array([kmh_from_mps(v) for v in spd_mps],dtype=float)
    spd_kmh = smooth_1d(spd_kmh,win=11)

    cap = cv2.VideoCapture(args.video)

    if not cap.isOpened():
        raise RuntimeError("Cannot open video")

    fps = cap.get(cv2.CAP_PROP_FPS)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    nframes = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if args.start_frame > 0:
        cap.set(cv2.CAP_PROP_POS_FRAMES,args.start_frame)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(args.out,fourcc,fps,(w,h))

    model = YOLO(args.model)

    device="mps"
    try:
        import torch
        if not torch.backends.mps.is_available():
            device="cpu"
    except:
        device="cpu"

    tracker_cfg="bytetrack.yaml"

    tl_cls_id=find_class_id(model.names,"traffic light")

    state_window=deque(maxlen=args.red_hold_frames)
    conf_window=deque(maxlen=args.red_hold_frames)

    fsm=0
    red_start_frame=None
    last_trigger_time=-1e9

    track_rows=[]
    signal_rows=[]
    violations=[]

    total_to_process=nframes if args.max_frames<0 else min(args.max_frames,max(0,nframes-args.start_frame))
    pbar=tqdm(total=total_to_process)

    frame_i=0

    while True:

        if args.max_frames>0 and frame_i>=args.max_frames:
            break

        ret,frame=cap.read()
        if not ret:
            break

        global_frame=args.start_frame+frame_i
        t_sec=global_frame/fps if fps>0 else None

        ti=int(global_frame*(len(spd_kmh)-1)/max(nframes-1,1))
        ti=max(0,min(ti,len(spd_kmh)-1))
        speed_now=float(spd_kmh[ti])

        results=model.track(
            frame,
            conf=args.conf,
            imgsz=args.imgsz,
            device=device,
            tracker=tracker_cfg,
            persist=True,
            verbose=False
        )

        annotated=results[0].plot()

        boxes=results[0].boxes

        tl_state="UNKNOWN"
        tl_conf=0
        tl_box=None
        near_intersection=False

        if boxes is not None and boxes.xyxy is not None:

            xyxy=boxes.xyxy.cpu().numpy()
            cls=boxes.cls.cpu().numpy().astype(int)
            confs=boxes.conf.cpu().numpy().astype(float)

            best=None

            for b,c,cf in zip(xyxy,cls,confs):

                if c!=tl_cls_id:
                    continue

                x1,y1,x2,y2=b.tolist()

                area=max(0,(x2-x1))*max(0,(y2-y1))

                if area<args.min_tl_area:
                    continue

                if best is None or cf>best[0]:
                    best=(cf,(x1,y1,x2,y2))

            if best is not None:

                _,(x1,y1,x2,y2)=best

                x1,y1,x2,y2=clamp_box(x1,y1,x2,y2,w,h)

                roi=frame[y1:y2,x1:x2]

                tl_state,tl_conf=classify_signal_color(roi)

                # SAVE DATASET IMAGE
                if roi is not None and roi.size>0:

                    label=tl_state.lower()

                    if label not in ["red","yellow","green"]:
                        label="unknown"

                    filename=f"{global_frame}.jpg"

                    save_path=os.path.join(dataset_dir,label,filename)

                    cv2.imwrite(save_path,roi)

        frame_i+=1
        pbar.update(1)

    pbar.close()
    cap.release()
    out.release()

    print("Processing finished")
    print("Dataset saved in data/tl_dataset")


if __name__=="__main__":
    main()