import pandas as pd

CLASS_NAMES = {
    0: "person",
    1: "bicycle",
    2: "car",
    3: "motorcycle",
    5: "bus",
    6: "train",
    7: "truck",
    9: "traffic light"
}

RELEVANT_CLASSES = [0, 2, 3, 5, 7, 9]

def compute_metrics(csv_path, model_name):
    df = pd.read_csv(csv_path)

    df["class_name"] = df["cls"].map(CLASS_NAMES).fillna("other")

    total_detections = len(df)
    unique_frames = df["frame"].nunique()
    detections_per_frame = total_detections / unique_frames
    unique_tracks = df["track_id"].nunique()
    track_lengths = df.groupby("track_id")["frame"].count()
    avg_track_length = track_lengths.mean()
    avg_conf = df["conf"].mean()
    short_tracks = (track_lengths < 10).sum()

    relevant_df = df[df["cls"].isin(RELEVANT_CLASSES)]
    irrelevant_df = df[~df["cls"].isin(RELEVANT_CLASSES)]

    print("\n==============================")
    print(f"Model: {model_name}")
    print("==============================")
    print(f"Total detections: {total_detections}")
    print(f"Frames processed: {unique_frames}")
    print(f"Detections per frame: {detections_per_frame:.2f}")
    print(f"Unique track IDs: {unique_tracks}")
    print(f"Average track length: {avg_track_length:.2f}")
    print(f"Average confidence: {avg_conf:.3f}")
    print(f"Tracks shorter than 10 frames: {short_tracks}")

    print("\nRelevant road-object detections:", len(relevant_df))
    print("Irrelevant detections:", len(irrelevant_df))

    print("\nRelevant detections per class:")
    print(relevant_df["class_name"].value_counts())

    print("\nIrrelevant detections per class:")
    print(irrelevant_df["class_name"].value_counts())

yolo11_path = "outputs/experiments/yolo_compare/yolo11/tracks.csv"
yolo26_path = "outputs/experiments/yolo_compare/yolo26/tracks.csv"

compute_metrics(yolo11_path, "YOLO11x")
compute_metrics(yolo26_path, "YOLO26x")