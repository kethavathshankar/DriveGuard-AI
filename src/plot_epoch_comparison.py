import os
import pandas as pd
import matplotlib.pyplot as plt

RUNS = {
    "E20": "runs/speed_sign/yolo11_speed_sign_e20/results.csv",
    "E50": "runs/speed_sign/yolo11_speed_sign_e50/results.csv",
    "E80": "runs/speed_sign/yolo11_speed_sign_e80/results.csv",
}

OUT_DIR = "outputs/experiments/speed_sign_epoch_compare"
os.makedirs(OUT_DIR, exist_ok=True)


def load_results(path):
    df = pd.read_csv(path)
    df.columns = [c.strip() for c in df.columns]
    return df


def find_col(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    raise ValueError(f"Could not find any of {candidates} in columns: {list(df.columns)}")


def main():
    run_dfs = {name: load_results(path) for name, path in RUNS.items()}

    # Likely YOLO column names
    precision_key_candidates = ["metrics/precision(B)", "metrics/precision"]
    recall_key_candidates = ["metrics/recall(B)", "metrics/recall"]
    map50_key_candidates = ["metrics/mAP50(B)", "metrics/mAP50"]
    map5095_key_candidates = ["metrics/mAP50-95(B)", "metrics/mAP50-95"]
    train_box_loss_candidates = ["train/box_loss"]
    val_box_loss_candidates = ["val/box_loss"]

    # Get keys from first available df
    sample_df = next(iter(run_dfs.values()))
    precision_col = find_col(sample_df, precision_key_candidates)
    recall_col = find_col(sample_df, recall_key_candidates)
    map50_col = find_col(sample_df, map50_key_candidates)
    map5095_col = find_col(sample_df, map5095_key_candidates)
    train_box_col = find_col(sample_df, train_box_loss_candidates)
    val_box_col = find_col(sample_df, val_box_loss_candidates)

    # ---------- Plot 1: mAP50 vs epoch ----------
    plt.figure(figsize=(8, 5))
    for run_name, df in run_dfs.items():
        plt.plot(df["epoch"], df[map50_col], label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("mAP50")
    plt.title("mAP50 vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "map50_vs_epoch.png"))
    plt.close()

    # ---------- Plot 2: Precision vs epoch ----------
    plt.figure(figsize=(8, 5))
    for run_name, df in run_dfs.items():
        plt.plot(df["epoch"], df[precision_col], label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Precision")
    plt.title("Precision vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "precision_vs_epoch.png"))
    plt.close()

    # ---------- Plot 3: Recall vs epoch ----------
    plt.figure(figsize=(8, 5))
    for run_name, df in run_dfs.items():
        plt.plot(df["epoch"], df[recall_col], label=run_name)
    plt.xlabel("Epoch")
    plt.ylabel("Recall")
    plt.title("Recall vs Epoch")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "recall_vs_epoch.png"))
    plt.close()

    # ---------- Plot 4: Train / Val box loss ----------
    plt.figure(figsize=(8, 5))
    for run_name, df in run_dfs.items():
        plt.plot(df["epoch"], df[train_box_col], label=f"{run_name} train")
        plt.plot(df["epoch"], df[val_box_col], linestyle="--", label=f"{run_name} val")
    plt.xlabel("Epoch")
    plt.ylabel("Box Loss")
    plt.title("Train and Validation Box Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, "box_loss_vs_epoch.png"))
    plt.close()

    # ---------- Best metrics summary ----------
    summary_rows = []
    for run_name, df in run_dfs.items():
        best_idx = df[map50_col].idxmax()
        summary_rows.append({
            "run": run_name,
            "best_epoch": int(df.loc[best_idx, "epoch"]),
            "best_precision": float(df.loc[best_idx, precision_col]),
            "best_recall": float(df.loc[best_idx, recall_col]),
            "best_map50": float(df.loc[best_idx, map50_col]),
            "best_map50_95": float(df.loc[best_idx, map5095_col]),
            "final_train_box_loss": float(df[train_box_col].iloc[-1]),
            "final_val_box_loss": float(df[val_box_col].iloc[-1]),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_path = os.path.join(OUT_DIR, "epoch_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("Saved plots to:", OUT_DIR)
    print("Saved summary CSV to:", summary_path)
    print("\nEpoch Summary:")
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()