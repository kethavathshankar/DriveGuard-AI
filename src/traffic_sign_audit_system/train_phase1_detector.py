from ultralytics import YOLO

def train_phase1():
    print("[Phase1 Training] Starting...")

    model = YOLO("models/yolo11n.pt")

    model.train(
        data="data/traffic_sign_dataset/data.yaml",
        epochs=1,
        imgsz=640,
        batch=2,
        device="mps",
        project="runs/traffic_sign_phase1",
        name="traffic_sign_detector_fast_v1",
        single_cls=True,
        pretrained=True,
        workers=2,
        cache=False,
        verbose=True
    )

    print("\n[Phase1 Training] Done!")

if __name__ == "__main__":
    train_phase1()