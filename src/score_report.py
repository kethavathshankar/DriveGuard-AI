import argparse
import json
import os
import pandas as pd


def clamp(x, lo, hi):
    return max(lo, min(hi, x))


def safe_float(x, default=0.0):
    try:
        if pd.isna(x):
            return default
        return float(x)
    except Exception:
        return default


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--overspeed_csv", default="outputs/events.csv",
                    help="Overspeed segments CSV (your events.csv)")
    ap.add_argument("--red_csv", default="outputs/min3_to_min10_red_violations.csv",
                    help="Red light violations CSV")
    ap.add_argument("--out_json", default="outputs/final_report.json",
                    help="Output report JSON")
    ap.add_argument("--trip_name", default="trip", help="Name to show in report")

    # scoring knobs (easy to tune)
    ap.add_argument("--start_score", type=float, default=100.0)
    ap.add_argument("--overspeed_event_penalty", type=float, default=3.0)
    ap.add_argument("--overspeed_per_sec_penalty", type=float, default=0.2)
    ap.add_argument("--overspeed_big_extra_penalty", type=float, default=5.0)
    ap.add_argument("--overspeed_big_delta_kmh", type=float, default=15.0)

    ap.add_argument("--red_light_penalty", type=float, default=25.0)

    args = ap.parse_args()

    # -------- Load CSVs safely --------
    overspeed_df = None
    if args.overspeed_csv and os.path.exists(args.overspeed_csv):
        overspeed_df = pd.read_csv(args.overspeed_csv)
    else:
        overspeed_df = pd.DataFrame()

    red_df = None
    if args.red_csv and os.path.exists(args.red_csv):
        red_df = pd.read_csv(args.red_csv)
    else:
        red_df = pd.DataFrame()

    # -------- Overspeed metrics --------
    overspeed_count = 0
    overspeed_total_duration = 0.0
    overspeed_max_speed = 0.0
    overspeed_big_count = 0

    if len(overspeed_df) > 0:
        # Expect columns like: duration_sec, max_speed_kmh, limit_thr_kmh
        overspeed_count = int(len(overspeed_df))

        if "duration_sec" in overspeed_df.columns:
            overspeed_total_duration = float(overspeed_df["duration_sec"].fillna(0).sum())
        else:
            overspeed_total_duration = 0.0

        if "max_speed_kmh" in overspeed_df.columns:
            overspeed_max_speed = float(overspeed_df["max_speed_kmh"].fillna(0).max())
        else:
            overspeed_max_speed = 0.0

        # big overspeed = max_speed - limit_thr > overspeed_big_delta_kmh
        if "max_speed_kmh" in overspeed_df.columns and "limit_thr_kmh" in overspeed_df.columns:
            delta = (overspeed_df["max_speed_kmh"] - overspeed_df["limit_thr_kmh"]).fillna(0)
            overspeed_big_count = int((delta > args.overspeed_big_delta_kmh).sum())
        else:
            overspeed_big_count = 0

    # -------- Red light metrics --------
    red_count = int(len(red_df)) if len(red_df) > 0 else 0

    # -------- Score computation --------
    score = args.start_score

    # Overspeed penalties
    score -= overspeed_count * args.overspeed_event_penalty
    score -= overspeed_total_duration * args.overspeed_per_sec_penalty
    score -= overspeed_big_count * args.overspeed_big_extra_penalty

    # Red light penalty
    score -= red_count * args.red_light_penalty

    score = clamp(score, 0.0, 100.0)

    # Verdict
    if score >= 80:
        verdict = "SAFE"
    elif score >= 50:
        verdict = "MODERATE_RISK"
    else:
        verdict = "UNSAFE"

    # Suggestions (simple, explainable)
    suggestions = []
    if red_count > 0:
        suggestions.append("Stop fully at RED signals and proceed only on GREEN.")
    if overspeed_count > 0:
        suggestions.append("Maintain speed within limit; avoid long overspeed durations.")
    if overspeed_big_count > 0:
        suggestions.append("Avoid large overspeed spikes; they sharply increase risk.")
    if not suggestions:
        suggestions.append("Good driving behavior in this segment. Keep consistent lane discipline and safe speed.")

    report = {
        "trip_name": args.trip_name,
        "score_0_100": round(score, 2),
        "verdict": verdict,
        "summary": {
            "overspeed_events": overspeed_count,
            "overspeed_total_duration_sec": round(overspeed_total_duration, 2),
            "overspeed_max_speed_kmh": round(overspeed_max_speed, 2),
            "overspeed_big_events": overspeed_big_count,
            "red_light_violations": red_count
        },
        "penalty_config": {
            "start_score": args.start_score,
            "overspeed_event_penalty": args.overspeed_event_penalty,
            "overspeed_per_sec_penalty": args.overspeed_per_sec_penalty,
            "overspeed_big_extra_penalty": args.overspeed_big_extra_penalty,
            "overspeed_big_delta_kmh": args.overspeed_big_delta_kmh,
            "red_light_penalty": args.red_light_penalty
        },
        "suggestions": suggestions
    }

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Final Report Saved:", args.out_json)
    print("Score:", report["score_0_100"], "| Verdict:", report["verdict"])
    print("Summary:", report["summary"])
    print("Suggestions:")
    for s in report["suggestions"]:
        print("-", s)


if __name__ == "__main__":
    main()