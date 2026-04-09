from pathlib import Path
import yaml

DATASET_DIR = Path("data/traffic_sign_dataset")


def convert_label_file(label_path: Path):
    if not label_path.exists():
        return

    lines = label_path.read_text().strip().splitlines()
    new_lines = []

    for line in lines:
        parts = line.strip().split()
        if len(parts) >= 5:
            parts[0] = "0"  # all classes -> single class
            new_lines.append(" ".join(parts))

    label_path.write_text("\n".join(new_lines) + ("\n" if new_lines else ""))


def convert_split(split_name: str):
    label_dir = DATASET_DIR / split_name / "labels"
    if not label_dir.exists():
        print(f"Skipping missing split: {label_dir}")
        return

    txt_files = list(label_dir.glob("*.txt"))
    print(f"Converting {split_name}: {len(txt_files)} label files")

    for txt_file in txt_files:
        convert_label_file(txt_file)


def update_data_yaml():
    yaml_path = DATASET_DIR / "data.yaml"
    if not yaml_path.exists():
        raise FileNotFoundError(f"Missing {yaml_path}")

    with open(yaml_path, "r") as f:
        data = yaml.safe_load(f)

    # keep paths simple and local
    data["train"] = "train/images"
    data["val"] = "valid/images"
    data["test"] = "test/images"
    data["nc"] = 1
    data["names"] = ["traffic_sign"]

    # optional cleanup if old keys exist
    if "roboflow" in data:
        pass

    with open(yaml_path, "w") as f:
        yaml.dump(data, f, sort_keys=False)

    print(f"Updated YAML: {yaml_path}")


def main():
    if not DATASET_DIR.exists():
        raise FileNotFoundError(f"Dataset folder not found: {DATASET_DIR}")

    for split in ["train", "valid", "test"]:
        convert_split(split)

    update_data_yaml()
    print("\nDone. Dataset is now single-class: traffic_sign")


if __name__ == "__main__":
    main()