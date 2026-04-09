import json
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "speed_classification"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
MODELS_DIR = PROJECT_ROOT / "models"
MODEL_OUT = MODELS_DIR / "best_classifier.pth"
CLASSMAP_OUT = MODELS_DIR / "class_to_idx.json"

REQUIRED_CLASSES = [
    "30",
    "not_speed_sign",
]

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def get_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def list_class_dirs(root_dir: Path):
    return sorted([p.name for p in root_dir.iterdir() if p.is_dir()])


def count_images_in_class_dir(class_dir: Path) -> int:
    return len([
        x for x in class_dir.iterdir()
        if x.is_file() and x.suffix.lower() in VALID_EXTS
    ])


def check_dataset_structure() -> None:
    if not TRAIN_DIR.exists():
        raise FileNotFoundError(f"Train folder not found: {TRAIN_DIR}")
    if not VAL_DIR.exists():
        raise FileNotFoundError(f"Val folder not found: {VAL_DIR}")

    train_classes = list_class_dirs(TRAIN_DIR)
    val_classes = list_class_dirs(VAL_DIR)

    if train_classes != REQUIRED_CLASSES:
        raise ValueError(
            f"Train class folders mismatch.\nExpected: {REQUIRED_CLASSES}\nFound: {train_classes}"
        )
    if val_classes != REQUIRED_CLASSES:
        raise ValueError(
            f"Val class folders mismatch.\nExpected: {REQUIRED_CLASSES}\nFound: {val_classes}"
        )

    for cls in REQUIRED_CLASSES:
        n_train = count_images_in_class_dir(TRAIN_DIR / cls)
        n_val = count_images_in_class_dir(VAL_DIR / cls)

        if n_train == 0:
            raise ValueError(f"Train folder is empty: {TRAIN_DIR / cls}")
        if n_val == 0:
            raise ValueError(f"Val folder is empty: {VAL_DIR / cls}")

    print("✅ Dataset folders found")
    print("✅ Required classes found:", REQUIRED_CLASSES)


def count_images(root_dir: Path) -> None:
    print(f"\nImage count under: {root_dir}")
    total = 0
    for class_name in REQUIRED_CLASSES:
        class_dir = root_dir / class_name
        n = count_images_in_class_dir(class_dir)
        total += n
        print(f"  {class_name:15s} : {n}")
    print(f"  TOTAL{'':11s} : {total}")


def build_loaders(batch_size: int = 32, image_size: int = 224):
    train_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.RandomHorizontalFlip(p=0.2),
        transforms.RandomRotation(8),
        transforms.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(str(TRAIN_DIR), transform=train_tf)
    val_ds = datasets.ImageFolder(str(VAL_DIR), transform=val_tf)

    if train_ds.classes != REQUIRED_CLASSES:
        raise ValueError(f"Train ImageFolder classes mismatch: {train_ds.classes}")
    if val_ds.classes != REQUIRED_CLASSES:
        raise ValueError(f"Val ImageFolder classes mismatch: {val_ds.classes}")

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_ds, val_ds, train_loader, val_loader


def build_model(num_classes: int) -> nn.Module:
    try:
        weights = models.ResNet18_Weights.DEFAULT
        model = models.resnet18(weights=weights)
    except Exception:
        model = models.resnet18(pretrained=True)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            out = model(imgs)
            loss = criterion(out, labels)

            total_loss += loss.item() * labels.size(0)
            preds = out.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / max(total, 1), correct / max(total, 1)


def train(
    epochs: int = 10,
    batch_size: int = 32,
    image_size: int = 224,
    lr: float = 1e-4,
):
    check_dataset_structure()
    count_images(TRAIN_DIR)
    count_images(VAL_DIR)

    device = get_device()
    print(f"\n✅ Using device: {device}")

    train_ds, val_ds, train_loader, val_loader = build_loaders(
        batch_size=batch_size,
        image_size=image_size,
    )

    num_classes = len(train_ds.classes)
    print("\nTrain classes:", train_ds.classes)
    print("Val classes:", val_ds.classes)
    print("Num classes:", num_classes)
    print("Train images:", len(train_ds))
    print("Val images:", len(val_ds))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(CLASSMAP_OUT, "w") as f:
        json.dump(train_ds.class_to_idx, f, indent=2)
    print(f"✅ Saved class map to: {CLASSMAP_OUT}")

    model = build_model(num_classes).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_acc = 0.0
    best_epoch = 0

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_total = 0

        for imgs, labels in train_loader:
            imgs = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            running_total += labels.size(0)

        train_loss = running_loss / max(running_total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"Epoch {epoch + 1}/{epochs} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

        if val_acc > best_acc:
            best_acc = val_acc
            best_epoch = epoch + 1
            torch.save(model.state_dict(), MODEL_OUT)
            print(f"✅ Saved best model to: {MODEL_OUT}")

    print("\nTraining complete")
    print(f"Best val acc: {best_acc:.4f} at epoch {best_epoch}")
    print(f"Best model path: {MODEL_OUT}")
    print(f"Class map path: {CLASSMAP_OUT}")


if __name__ == "__main__":
    train(
        epochs=50,
        batch_size=32,
        image_size=224,
        lr=1e-4,
    )