import os
from typing import Dict, Any

import pandas as pd


# =========================================================
# Helpers
# =========================================================
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)


def find_column(df, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None


def build_maps_link(lat, lon):
    if lat is None or lon is None:
        return ""
    return f"https://www.google.com/maps?q={lat},{lon}"


# =========================================================
# Main
# =========================================================
def run_phase5_telemetry_location(
    geometry_csv_path: str,
    telemetry_csv_path: str,
    output_dir="outputs/traffic_sign_audit_system",
    output_csv_name="traffic_sign_with_location.csv",
    progress_callback=None,
) -> Dict[str, Any]:

    if not os.path.exists(geometry_csv_path):
        raise FileNotFoundError("Geometry CSV not found")

    if not os.path.exists(telemetry_csv_path):
        raise FileNotFoundError("Telemetry CSV not found")

    ensure_dir(output_dir)

    out_csv = os.path.join(output_dir, output_csv_name)

    geo_df = pd.read_csv(geometry_csv_path)
    tel_df = pd.read_csv(telemetry_csv_path)

    print("[Phase5] Loaded geometry + telemetry")

    # -------------------------
    # detect telemetry columns
    # -------------------------
    lat_col = find_column(tel_df, ["gps_lat", "lat", "latitude"])
    lon_col = find_column(tel_df, ["gps_lon", "lon", "longitude"])
    alt_col = find_column(tel_df, ["gps_alt", "alt", "altitude"])
    spd_col = find_column(tel_df, ["gps_spd_3d", "gps_spd_2d"])

    print("[Phase5] Columns detected:")
    print("lat:", lat_col, "lon:", lon_col, "alt:", alt_col, "speed:", spd_col)

    # -------------------------
    # create time alignment
    # -------------------------
    # assume telemetry aligned by index ~ frame
    tel_len = len(tel_df)

    rows_out = []
    total = len(geo_df)

    for idx, row in geo_df.iterrows():
        frame = int(row["frame"])

        tel_idx = min(frame, tel_len - 1)

        tel_row = tel_df.iloc[tel_idx]

        lat = tel_row[lat_col] if lat_col else None
        lon = tel_row[lon_col] if lon_col else None
        alt = tel_row[alt_col] if alt_col else None
        speed = tel_row[spd_col] if spd_col else None

        maps_link = build_maps_link(lat, lon)

        rows_out.append({
            **row.to_dict(),

            "latitude": lat,
            "longitude": lon,
            "altitude": alt,
            "vehicle_speed": speed,
            "maps_link": maps_link,
        })

        if progress_callback:
            progress_callback(idx + 1, total)

        if (idx + 1) % 100 == 0:
            print(f"[Phase5] processed {idx+1}/{total}")

    out_df = pd.DataFrame(rows_out)
    out_df.to_csv(out_csv, index=False)

    return {
        "location_csv": out_csv,
        "num_rows": len(out_df),
        "message": "Phase 5 complete"
    }


# =========================================================
# Run
# =========================================================
def _progress(done, total):
    print(f"[Phase5] processed={done}/{total}")


if __name__ == "__main__":
    geometry_csv = "outputs/traffic_sign_audit_system/traffic_sign_geometry.csv"
    telemetry_csv = "data/front_telemetry.csv"

    result = run_phase5_telemetry_location(
        geometry_csv_path=geometry_csv,
        telemetry_csv_path=telemetry_csv,
        output_dir="outputs/traffic_sign_audit_system",
        output_csv_name="traffic_sign_with_location.csv",
        progress_callback=_progress,
    )

    print("\n[Phase5] Done.")
    print("[Phase5] CSV :", result["location_csv"])