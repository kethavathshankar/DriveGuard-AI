import os
import shutil
import random
import pandas as pd
from pathlib import Path
from PIL import Image

RAW_ROOT = Path("data/raw_gtsdb/TrainIJCNN2013/TrainIJCNN2013")
ANNOT_CSV = RAW_ROOT / "gt.txt"
OUT_ROOT = Path("data/speed_sign_dataset")

# GTSDB speed-limit classes
SPEED_CLASS_MAP = {
    0: (0, "speed_limit_20"),
    1: (1, "speed_limit_30"),
    2: (2, "speed_limit_50"),
    3: (3, "speed_limit_60"),
    4: (4, "speed_limit_70"),
    5: (5, "speed_limit_80"),
    7: (6, "speed_limit_100"),
    8: (7, "speed_limit_120"),
}

def ensure_dirs():
    for split in ["train", "val"]:
        (OUT_ROOT / "images" / split).mkdir(parents=True, exist_ok=True)
        (OUT_ROOT / "labels" / split).mkdir(parents=True, exist_ok=True)

def to_yolo_box(x1, y1, x2, y2, w, h):
    xc = ((x1 + x2) / 2) / w
    yc = ((y1 + y2) / 2) / h
    bw = (x2 - x1) / w
    bh = (y2 - y1) / h
    return xc, yc, bw, bh

def main():
    ensure_dirs()

    df = pd.read_csv(ANNOT_CSV, sep=";", header=None)
    df.columns = ["filename", "x1", "y1", "x2", "y2", "class_id"]

    df = df[df["class_id"].isin(SPEED_CLASS_MAP.keys())].copy()

    filenames = sorted(df["filename"].unique().tolist())
    random.seed(42)
    random.shuffle(filenames)

    split_idx = int(0.8 * len(filenames))
    train_files = set(filenames[:split_idx])
    val_files = set(filenames[split_idx:])

    grouped = df.groupby("filename")

    total_images = 0
    total_labels = 0

    for filename, rows in grouped:
        split = "train" if filename in train_files else "val"
        img_src = RAW_ROOT / filename

        if not img_src.exists():
            print(f"Missing image: {img_src}")
            continue

        stem = Path(filename).stem
        img_dst = OUT_ROOT / "images" / split / f"{stem}.jpg"
        label_dst = OUT_ROOT / "labels" / split / f"{stem}.txt"

        # Convert PPM -> JPG
        with Image.open(img_src) as im:
            w, h = im.size
            im = im.convert("RGB")
            im.save(img_dst, format="JPEG", quality=95)

        yolo_lines = []
        for _, r in rows.iterrows():
            orig_cls = int(r["class_id"])
            new_cls, _ = SPEED_CLASS_MAP[orig_cls]

            xc, yc, bw, bh = to_yolo_box(
                float(r["x1"]), float(r["y1"]),
                float(r["x2"]), float(r["y2"]),
                w, h
            )
            yolo_lines.append(f"{new_cls} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")

        with open(label_dst, "w") as f:
            f.write("\n".join(yolo_lines))

        total_images += 1
        total_labels += 1

    print(f"Done. Created {total_images} images and {total_labels} label files in {OUT_ROOT}")

if __name__ == "__main__":
    main()