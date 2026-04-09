import pandas as pd
import os


def stabilize_speed_signs(
    input_csv,
    output_csv,
    min_repeat=2,
    frame_gap=10,
):
    if not os.path.exists(input_csv):
        raise FileNotFoundError(f"Input not found: {input_csv}")

    df = pd.read_csv(input_csv)

    if len(df) == 0:
        out_df = pd.DataFrame(columns=["front_frame_id", "detected_speed_sign"])
        out_df.to_csv(output_csv, index=False)
        return out_df

    df = df.sort_values(["front_frame_id"]).reset_index(drop=True)

    stable_rows = []
    i = 0

    while i < len(df):
        sign_val = int(df.loc[i, "detected_speed_sign"])
        start_frame = int(df.loc[i, "front_frame_id"])

        group = [i]
        j = i + 1

        while j < len(df):
            f = int(df.loc[j, "front_frame_id"])
            s = int(df.loc[j, "detected_speed_sign"])

            if s == sign_val and (f - int(df.loc[group[-1], "front_frame_id"])) <= frame_gap:
                group.append(j)
                j += 1
            else:
                break

        if len(group) >= min_repeat:
            stable_rows.append({
                "front_frame_id": start_frame,
                "detected_speed_sign": sign_val
            })

        i = j

    out_df = pd.DataFrame(stable_rows)
    out_df.to_csv(output_csv, index=False)
    return out_df


if __name__ == "__main__":
    input_csv = "outputs/speed_sign_detections.csv"
    output_csv = "outputs/speed_sign_detections_stable.csv"

    out_df = stabilize_speed_signs(
        input_csv=input_csv,
        output_csv=output_csv,
        min_repeat=2,
        frame_gap=10,
    )

    print(f"Saved stable speed-sign detections to: {output_csv}")
    print(out_df.head())
    print(f"Total stable sign events: {len(out_df)}")