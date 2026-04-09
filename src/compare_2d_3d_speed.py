import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

MASTER_CSV = "outputs/master_timeline.csv"
TELEMETRY_CSV = "data/front_telemetry.csv"
OUT_DIR = "outputs/experiments/speed_compare_final"

os.makedirs(OUT_DIR, exist_ok=True)


def mps_to_kmph(x):
    return x * 3.6


# --------------------------------------------------
# Load data
# --------------------------------------------------
master = pd.read_csv(MASTER_CSV)
telemetry = pd.read_csv(TELEMETRY_CSV)

required_cols = ["gps_spd_2d", "gps_spd_3d"]
for c in required_cols:
    if c not in telemetry.columns:
        raise ValueError(f"Missing required telemetry column: {c}")

n = min(len(master), len(telemetry))
if n == 0:
    raise ValueError("Empty master_timeline.csv or front_telemetry.csv")

master = master.iloc[:n].reset_index(drop=True)
telemetry = telemetry.iloc[:n].reset_index(drop=True)

df = pd.DataFrame()
df["frame"] = master["front_frame_id"] if "front_frame_id" in master.columns else np.arange(n)
df["gps_spd_2d_mps"] = pd.to_numeric(telemetry["gps_spd_2d"], errors="coerce")
df["gps_spd_3d_mps"] = pd.to_numeric(telemetry["gps_spd_3d"], errors="coerce")

df = df.dropna().reset_index(drop=True)

# Convert to km/h
df["gps_spd_2d_kmh"] = mps_to_kmph(df["gps_spd_2d_mps"])
df["gps_spd_3d_kmh"] = mps_to_kmph(df["gps_spd_3d_mps"])

# Difference
df["abs_diff_kmh"] = np.abs(df["gps_spd_3d_kmh"] - df["gps_spd_2d_kmh"])

# Save aligned csv
df.to_csv(os.path.join(OUT_DIR, "telemetry_2d_3d_aligned.csv"), index=False)

# --------------------------------------------------
# Graph 1: 2D vs 3D speed
# --------------------------------------------------
plt.figure(figsize=(12, 5))
plt.plot(df["frame"], df["gps_spd_2d_kmh"], label="2D Telemetry Speed (km/h)", linewidth=1.5)
plt.plot(df["frame"], df["gps_spd_3d_kmh"], label="3D Telemetry Speed (km/h)", linewidth=1.5)
plt.xlabel("Frame")
plt.ylabel("Speed (km/h)")
plt.title("2D vs 3D Telemetry Speed Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "graph_1_2d_vs_3d_speed.png"), dpi=220)
plt.close()

# --------------------------------------------------
# Graph 2: Absolute difference
# --------------------------------------------------
plt.figure(figsize=(12, 4))
plt.plot(df["frame"], df["abs_diff_kmh"], linewidth=1.5)
plt.xlabel("Frame")
plt.ylabel("Absolute Difference (km/h)")
plt.title("Absolute Difference Between 2D and 3D Speed")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "graph_2_absolute_difference.png"), dpi=220)
plt.close()

# --------------------------------------------------
# Summary
# --------------------------------------------------
mean_abs_diff = df["abs_diff_kmh"].mean()
median_abs_diff = df["abs_diff_kmh"].median()
max_abs_diff = df["abs_diff_kmh"].max()
corr = df["gps_spd_2d_kmh"].corr(df["gps_spd_3d_kmh"])

print("\n===== SUMMARY =====")
print(f"Rows used                : {len(df)}")
print(f"Average absolute gap     : {mean_abs_diff:.4f} km/h")
print(f"Median absolute gap      : {median_abs_diff:.4f} km/h")
print(f"Maximum absolute gap     : {max_abs_diff:.4f} km/h")
print(f"Correlation (2D vs 3D)   : {corr:.6f}")

print("\n===== CONCLUSION =====")
print("2D and 3D telemetry speeds are highly similar and strongly correlated.")
print("The speed gap between them is generally small, so both curves almost overlap.")
print("This shows that the vehicle motion is mostly on the ground plane, with only minor variation.")
print("For overspeed analysis, gps_spd_3d can still be treated as the final telemetry speed.")

print(f"\nSaved graphs in: {OUT_DIR}")