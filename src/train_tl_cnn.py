import os
from pathlib import Path
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

DATA_DIR = "data/tl_dataset/hsv"
MODEL_OUT = "models/tl_cnn.pt"
IMG_SIZE = 64
BATCH_SIZE = 32
EPOCHS = 5
LR = 1e-4

def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"

def main():
    os.makedirs("models", exist_ok=True)

    transform = transforms.Compose([
        transforms.Resize((IMG_SIZE, IMG_SIZE)),
        transforms.ToTensor(),
    ])

    dataset = datasets.ImageFolder(DATA_DIR, transform=transform)

    n = len(dataset)
    n_train = int(0.8 * n)
    n_val = n - n_train
    train_ds, val_ds = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH_SIZE, shuffle=False)

    device = get_device()
    print("Using device:", device)
    print("Classes:", dataset.classes)

    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(dataset.classes))
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    best_val_acc = 0.0

    for epoch in range(EPOCHS):
        model.train()
        train_correct = 0
        train_total = 0
        train_loss_sum = 0.0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()
            out = model(x)
            loss = criterion(out, y)
            loss.backward()
            optimizer.step()

            train_loss_sum += loss.item() * x.size(0)
            preds = out.argmax(dim=1)
            train_correct += (preds == y).sum().item()
            train_total += y.size(0)

        train_acc = train_correct / max(train_total, 1)
        train_loss = train_loss_sum / max(train_total, 1)

        model.eval()
        val_correct = 0
        val_total = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                out = model(x)
                preds = out.argmax(dim=1)
                val_correct += (preds == y).sum().item()
                val_total += y.size(0)

        val_acc = val_correct / max(val_total, 1)

        print(f"Epoch {epoch+1}/{EPOCHS} | train_loss={train_loss:.4f} | train_acc={train_acc:.4f} | val_acc={val_acc:.4f}")

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                "model_state_dict": model.state_dict(),
                "class_names": dataset.classes,
                "img_size": IMG_SIZE,
            }, MODEL_OUT)

    print("Saved best model to:", MODEL_OUT)
    print("Best val acc:", best_val_acc)

if __name__ == "__main__":
    main()