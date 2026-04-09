import os
import uuid
from io import BytesIO
from collections import deque

import cv2
import numpy as np
import pandas as pd
import streamlit as st
from ultralytics import YOLO
import plotly.graph_objects as go

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import cm
from reportlab.lib import colors
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from speed_sign_live import run_speed_sign_detector


# ------------------ helpers ------------------
def kmh_from_mps(mps: float) -> float:
    return float(mps) * 3.6


def smooth_1d(x: np.ndarray, win: int = 11) -> np.ndarray:
    if win <= 1 or len(x) < 3:
        return x
    if win % 2 == 0:
        win += 1
    win = min(win, len(x) if len(x) % 2 == 1 else len(x) - 1)
    if win < 3:
        return x
    kernel = np.ones(win) / win
    return np.convolve(x, kernel, mode="same")


def estimate_dt(num_samples: int, fps: float, num_frames: int):
    if num_samples <= 1 or fps <= 0:
        return 1.0 / 20.0
    total_time = (num_frames - 1) / fps
    return total_time / max(num_samples - 1, 1)


def get_first_available_column(df: pd.DataFrame, names):
    for name in names:
        if name in df.columns:
            return df[name].to_numpy(dtype=float), name
    raise ValueError(f"Missing any of these columns: {names}")


def get_tilt_columns(df: pd.DataFrame):
    ax, ax_name = get_first_available_column(df, ["accl_x", "accel_x"])
    ay, ay_name = get_first_available_column(df, ["accl_y", "accel_y"])
    az, az_name = get_first_available_column(df, ["accl_z", "accel_z"])
    gx, gx_name = get_first_available_column(df, ["gyro_x"])
    gy, gy_name = get_first_available_column(df, ["gyro_y"])
    gz, gz_name = get_first_available_column(df, ["gyro_z"])
    return {
        "ax": ax, "ay": ay, "az": az,
        "gx": gx, "gy": gy, "gz": gz,
        "ax_name": ax_name, "ay_name": ay_name, "az_name": az_name,
        "gx_name": gx_name, "gy_name": gy_name, "gz_name": gz_name,
    }


def compute_tilt_from_arrays(ax, ay, az, gx, gy, dt: float, alpha=0.98, smooth_win=9):
    roll_acc = np.degrees(np.arctan2(ay, az + 1e-6))
    pitch_acc = np.degrees(np.arctan2(-ax, np.sqrt(ay ** 2 + az ** 2) + 1e-6))

    pitch = np.zeros(len(ax), dtype=float)
    roll = np.zeros(len(ax), dtype=float)

    for i in range(1, len(ax)):
        roll_gyro = roll[i - 1] + gx[i] * dt
        pitch_gyro = pitch[i - 1] + gy[i] * dt

        roll[i] = alpha * roll_gyro + (1.0 - alpha) * roll_acc[i]
        pitch[i] = alpha * pitch_gyro + (1.0 - alpha) * pitch_acc[i]

    pitch = smooth_1d(pitch, smooth_win)
    roll = smooth_1d(roll, smooth_win)

    n0 = min(100, len(pitch))
    if n0 > 0:
        pitch = pitch - np.median(pitch[:n0])
        roll = roll - np.median(roll[:n0])

    return pitch, roll


def pick_speed_column(df: pd.DataFrame):
    if "gps_spd_3d" in df.columns:
        return df["gps_spd_3d"].to_numpy(dtype=float), "gps_spd_3d"
    if "gps_spd_2d" in df.columns:
        return df["gps_spd_2d"].to_numpy(dtype=float), "gps_spd_2d"
    raise ValueError("Telemetry CSV missing gps_spd_3d or gps_spd_2d columns.")


def find_class_id(names: dict, target: str):
    target = target.lower()
    for k, v in names.items():
        if str(v).lower() == target:
            return int(k)
    return None


def clamp_box(x1, y1, x2, y2, w, h):
    x1 = max(0, min(int(x1), w - 1))
    y1 = max(0, min(int(y1), h - 1))
    x2 = max(0, min(int(x2), w - 1))
    y2 = max(0, min(int(y2), h - 1))
    if x2 <= x1:
        x2 = min(w - 1, x1 + 1)
    if y2 <= y1:
        y2 = min(h - 1, y1 + 1)
    return x1, y1, x2, y2


def classify_signal_color(roi_bgr: np.ndarray):
    if roi_bgr is None or roi_bgr.size == 0:
        return "UNKNOWN", 0.0

    roi = cv2.resize(roi_bgr, (64, 64), interpolation=cv2.INTER_AREA)
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

    red1 = cv2.inRange(hsv, (0, 70, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (160, 70, 70), (180, 255, 255))
    red = cv2.bitwise_or(red1, red2)

    yellow = cv2.inRange(hsv, (15, 70, 70), (35, 255, 255))
    green = cv2.inRange(hsv, (40, 70, 70), (90, 255, 255))

    r = int(np.sum(red > 0))
    y = int(np.sum(yellow > 0))
    g = int(np.sum(green > 0))
    total = r + y + g
    if total < 30:
        return "UNKNOWN", 0.0

    mx = max(r, y, g)
    conf = mx / (total + 1e-6)
    if mx == r:
        return "RED", float(conf)
    if mx == y:
        return "YELLOW", float(conf)
    return "GREEN", float(conf)


def verdict_from_score(score: float) -> str:
    if score >= 80:
        return "SAFE"
    if score >= 50:
        return "MODERATE_RISK"
    return "UNSAFE"


def score_gauge(score: float) -> go.Figure:
    score = max(0.0, min(100.0, float(score)))

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score,
        number={"font": {"size": 44}},
        gauge={
            "shape": "angular",
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "rgba(0,0,0,0.35)"},
            "bar": {"thickness": 0.25},
            "bgcolor": "white",
            "steps": [
                {"range": [0, 25], "color": "#7a0000"},
                {"range": [25, 50], "color": "#ff4d4d"},
                {"range": [50, 70], "color": "#ffd166"},
                {"range": [70, 85], "color": "#a7f3a7"},
                {"range": [85, 100], "color": "#0f8a0f"}
            ],
            "threshold": {
                "line": {"color": "black", "width": 3},
                "thickness": 0.8,
                "value": score
            }
        },
        title={"text": "Civil Driving Score"}
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=10, r=10, t=40, b=0),
        paper_bgcolor="white",
        plot_bgcolor="white"
    )
    return fig


def tilt_timeseries_figure(times, pitch_vals, roll_vals, current_idx: int) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=times,
        y=pitch_vals,
        mode="lines",
        name="Pitch (deg)"
    ))
    fig.add_trace(go.Scatter(
        x=times,
        y=roll_vals,
        mode="lines",
        name="Roll (deg)"
    ))
    if len(times) > 0:
        idx = max(0, min(current_idx, len(times) - 1))
        fig.add_vline(x=float(times[idx]), line_width=2, line_dash="dash")
    fig.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=30, b=10),
        xaxis_title="Time (sec)",
        yaxis_title="Angle (deg)",
        legend=dict(orientation="h", yanchor="bottom", y=1.01, xanchor="left", x=0.0),
    )
    return fig


def put_text_outline(
    img,
    text,
    org,
    font=cv2.FONT_HERSHEY_SIMPLEX,
    scale=0.9,
    color=(255, 255, 255),
    thickness=2,
    outline_color=(0, 0, 0),
    outline_thickness=6,
):
    cv2.putText(img, text, org, font, scale, outline_color, outline_thickness, cv2.LINE_AA)
    cv2.putText(img, text, org, font, scale, color, thickness, cv2.LINE_AA)


def draw_center_alert(img, text, box_color=(0, 0, 255), alpha=0.35):
    if not text:
        return

    h, w = img.shape[:2]
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 1.35
    thickness = 4

    (tw, th), _ = cv2.getTextSize(text, font, scale, thickness)
    cx, cy = w // 2, int(h * 0.35)

    x1 = max(0, cx - tw // 2 - 30)
    y1 = max(0, cy - th - 25)
    x2 = min(w - 1, cx + tw // 2 + 30)
    y2 = min(h - 1, cy + 25)

    overlay = img.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, -1)
    img[:] = cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0)

    put_text_outline(
        img,
        text,
        (cx - tw // 2, cy),
        font=font,
        scale=scale,
        color=(255, 255, 255),
        thickness=3,
        outline_color=(0, 0, 0),
        outline_thickness=8,
    )


def report_to_pdf_bytes(report: dict) -> bytes:
    styles = getSampleStyleSheet()
    story = []

    title = f"CVIT Driver Monitor Report — {report.get('trip_name', '')}"
    story.append(Paragraph(title, styles["Title"]))
    story.append(Spacer(1, 0.4 * cm))

    score = float(report.get("score_0_100", 0) or 0)
    verdict = report.get("verdict", "UNKNOWN")
    summary = report.get("summary", {})

    overview_tbl = Table([
        ["Score (0–100)", f"{score:.2f}"],
        ["Verdict", f"{verdict}"],
        ["Telemetry source", str(report.get("telemetry_source", ""))],
        ["Tilt telemetry columns", str(report.get("tilt_columns_used", ""))],
        ["Tilt alpha", str(report.get("tilt_alpha", ""))],
        ["Tilt smooth win", str(report.get("tilt_smooth_win", ""))],
        ["Fallback limit (km/h)", f"{report.get('fallback_default_limit', '')}"],
        ["Sign valid distance (m)", f"{report.get('sign_valid_distance_m', '')}"],
        ["Overspeed events", f"{summary.get('overspeed_events', 0)}"],
        ["Overspeed total duration (sec)", f"{summary.get('overspeed_total_duration_sec', 0)}"],
        ["Overspeed max speed (km/h)", f"{summary.get('overspeed_max_speed_kmh', 0)}"],
        ["Red light violations", f"{summary.get('red_light_violations', 0)}"],
        ["High pitch events", f"{summary.get('high_pitch_events', 0)}"],
        ["High roll events", f"{summary.get('high_roll_events', 0)}"],
        ["Max |pitch| (deg)", f"{summary.get('max_abs_pitch_deg', 0)}"],
        ["Max |roll| (deg)", f"{summary.get('max_abs_roll_deg', 0)}"],
    ], colWidths=[7 * cm, 9 * cm])

    overview_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("PADDING", (0, 0), (-1, -1), 6),
    ]))

    story.append(Paragraph("<b>Overview</b>", styles["Heading2"]))
    story.append(overview_tbl)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Suggestions</b>", styles["Heading2"]))
    suggestions = report.get("suggestions", [])
    if not suggestions:
        story.append(Paragraph("No suggestions.", styles["BodyText"]))
    else:
        for s in suggestions:
            story.append(Paragraph(f"• {s}", styles["BodyText"]))
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Violations</b>", styles["Heading2"]))
    violations = report.get("violations", [])
    if not violations:
        story.append(Paragraph("No violations logged.", styles["BodyText"]))
    else:
        header = ["time_sec", "frame", "violation_type", "severity", "value", "message"]
        rows = [header]
        for v in violations:
            rows.append([
                f"{v.get('time_sec', '')}",
                f"{v.get('frame', '')}",
                str(v.get("violation_type", "")),
                str(v.get("severity", "")),
                f"{v.get('value', '')}",
                str(v.get("message", ""))[:140],
            ])

        vtbl = Table(rows, colWidths=[2 * cm, 2 * cm, 3.2 * cm, 2.2 * cm, 2 * cm, 6.2 * cm])
        vtbl.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.lightgrey),
            ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
            ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("PADDING", (0, 0), (-1, -1), 4),
        ]))
        story.append(vtbl)

    buf = BytesIO()
    doc = SimpleDocTemplate(
        buf,
        pagesize=A4,
        rightMargin=1.3 * cm,
        leftMargin=1.3 * cm,
        topMargin=1.3 * cm,
        bottomMargin=1.3 * cm,
        title=title,
    )
    doc.build(story)
    return buf.getvalue()


def get_best_device():
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def load_speed_sign_events(csv_path: str) -> pd.DataFrame:
    if not csv_path or not os.path.exists(csv_path):
        return pd.DataFrame(columns=["frame", "speed_limit"])

    df = pd.read_csv(csv_path)
    required = {"frame", "speed_limit"}
    if not required.issubset(df.columns):
        raise ValueError(f"Speed sign CSV must contain columns: {required}")

    keep_cols = [c for c in [
        "frame", "time_sec", "speed_limit", "ocr_confidence",
        "track_start_frame", "track_end_frame", "first_positive_frame", "track_id"
    ] if c in df.columns]
    df = df[keep_cols].copy()
    df["frame"] = df["frame"].astype(int)
    df["speed_limit"] = df["speed_limit"].astype(float)
    df = df.sort_values("frame").drop_duplicates(subset=["frame"], keep="last").reset_index(drop=True)
    return df


def filter_sign_events_for_window(sign_df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    if sign_df.empty:
        return sign_df.copy()

    df = sign_df[sign_df["frame"] <= int(end_frame)].copy()

    before = df[df["frame"] < int(start_frame)]
    within = df[(df["frame"] >= int(start_frame)) & (df["frame"] <= int(end_frame))]

    rows = []
    if len(before) > 0:
        rows.append(before.iloc[[-1]])
    if len(within) > 0:
        rows.append(within)

    if not rows:
        return pd.DataFrame(columns=df.columns)

    out = pd.concat(rows, axis=0).sort_values("frame").reset_index(drop=True)
    return out


def build_sign_event_map(sign_df: pd.DataFrame) -> dict:
    if sign_df.empty:
        return {}
    return {int(row["frame"]): float(row["speed_limit"]) for _, row in sign_df.iterrows()}


def compute_cumulative_distance_m(times_sec: np.ndarray, speeds_kmph: np.ndarray) -> np.ndarray:
    speeds_mps = speeds_kmph / 3.6
    n = len(times_sec)
    cumulative = np.zeros(n, dtype=float)
    for i in range(1, n):
        dt = max(0.0, times_sec[i] - times_sec[i - 1])
        cumulative[i] = cumulative[i - 1] + speeds_mps[i] * dt
    return cumulative


def get_tilt_status(pitch, roll, pitch_th=8.0, roll_th=10.0):
    status = []

    if pitch > pitch_th:
        status.append(f"Hard Braking / Uphill (Pitch: +{pitch:.2f}°)")
    elif pitch < -pitch_th:
        status.append(f"Rapid Acceleration / Downhill (Pitch: {pitch:.2f}°)")

    if roll > roll_th:
        status.append(f"Turning Right (Roll: +{roll:.2f}°)")
    elif roll < -roll_th:
        status.append(f"Turning Left (Roll: {roll:.2f}°)")

    if not status:
        return f"Stable (Pitch: {pitch:+.2f}°, Roll: {roll:+.2f}°)"

    return " | ".join(status)


# ------------------ Streamlit UI ------------------
st.set_page_config(page_title="CVIT Driver Monitor Dashboard", layout="wide")
st.title("🚗 CVIT Driver Monitor Dashboard (Live Violations + Score + Vehicle Tilt)")

if "run_id" not in st.session_state:
    st.session_state.run_id = str(uuid.uuid4())

with st.sidebar:
    st.header("Inputs")
    video_path = st.text_input("Front Video Path", "data/front.mp4")
    telemetry_path = st.text_input("Front Telemetry CSV Path", "data/front_telemetry.csv")
    front_model_path = st.text_input("Front Detection Model", "models/yolo26x.pt")

    st.subheader("Automatic Speed Sign Detection")
    speed_sign_model_path = st.text_input("Speed Sign Detector Model Path", "models/speed_sign_detector.pt")
    sign_process_every = st.slider("Sign detect every N frames", 1, 30, 1, 1)
    sign_conf_thres = st.slider("Sign detector conf", 0.05, 0.9, 0.10, 0.05)
    sign_min_repeat = st.slider("Stable sign repeat count", 1, 5, 2, 1)
    sign_max_gap_frames = st.slider("Stable sign max gap frames", 1, 60, 12, 1)

    st.header("Time Window")
    start_min = st.number_input("Start minute", min_value=0.0, value=2.0, step=0.5)
    end_min = st.number_input("End minute", min_value=0.5, value=2.50, step=0.5)

    st.header("Performance")
    conf = st.slider("YOLO conf", 0.05, 0.8, 0.30, 0.05)
    imgsz = st.select_slider("imgsz", options=[640, 800, 960, 1280], value=960)
    show_every_n = st.slider("UI refresh every N frames", 1, 10, 2, 1)
    process_every_k = st.slider("Process every K frames (speed-up)", 1, 6, 1, 1)

    st.header("Speed Rule")
    fallback_default_limit = st.number_input(
        "Fallback Default Limit (used only if no recent sign is active)",
        min_value=0.0,
        value=100.0,
        step=1.0,
    )
    sign_valid_distance_m = st.number_input(
        "Sign Valid Distance (meters)",
        min_value=1.0,
        value=60.0,
        step=10.0,
    )

    st.header("Vehicle Tilt")
    tilt_alpha = st.slider("Tilt complementary filter alpha", 0.80, 0.999, 0.98, 0.001)
    tilt_smooth_win = st.slider("Tilt smoothing window", 1, 31, 9, 2)
    pitch_warn_deg = st.slider("Pitch warning threshold (deg)", 2.0, 30.0, 8.0, 0.5)
    roll_warn_deg = st.slider("Roll warning threshold (deg)", 2.0, 30.0, 10.0, 0.5)
    show_tilt_plot = st.checkbox("Show tilt plot", value=True)

    st.header("Red Light Logic")
    red_hold = st.slider("RED stable frames", 3, 30, 12, 1)
    near_ratio = st.slider("Near-intersection TL height ratio", 0.01, 0.12, 0.03, 0.005)
    min_tl_area = st.slider("Min TL bbox area", 10, 200, 40, 5)
    min_move_kmh = st.number_input("Moving threshold km/h", min_value=0.0, value=3.0, step=1.0)
    cooldown_sec = st.number_input("Red violation cooldown (sec)", min_value=0.0, value=8.0, step=1.0)

    st.header("Score")
    start_score = st.number_input("Start score", value=100.0, step=5.0)
    overspeed_event_penalty = st.number_input("Overspeed penalty per event", value=3.0, step=0.5)
    overspeed_per_sec_penalty = st.number_input("Overspeed penalty per sec", value=0.2, step=0.05)
    red_penalty = st.number_input("Red light penalty", value=25.0, step=1.0)

    st.header("Output")
    save_annotated_video = st.checkbox("Save annotated video (mp4)", value=True)
    save_sign_crops = st.checkbox("Save speed sign crops", value=True)

run_btn = st.button("▶ Run Dashboard Processing", type="primary")


# ------------------ RUN ------------------
if run_btn:
    if not os.path.exists(video_path):
        st.error(f"Video not found: {video_path}")
        st.stop()

    if not os.path.exists(telemetry_path):
        st.error(f"Telemetry not found: {telemetry_path}")
        st.stop()

    if not os.path.exists(front_model_path):
        st.error(f"Front model file not found: {front_model_path}")
        st.stop()

    if not os.path.exists(speed_sign_model_path):
        st.error(f"Speed sign model file not found: {speed_sign_model_path}")
        st.stop()

    os.makedirs("outputs", exist_ok=True)

    for p in [
        "outputs/speed_sign_raw_detections.csv",
        "outputs/speed_sign_events.csv",
        "outputs/speed_sign_debug.mp4",
        "outputs/dashboard_annotated.mp4",
    ]:
        if os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    stage_box = st.empty()
    status_box = st.empty()
    progress_text_box = st.empty()
    live_backend_box = st.empty()
    raw_sign_table_box = st.empty()
    stable_sign_table_box = st.empty()
    overall_progress_bar = st.progress(0.0)
    stage_progress_bar = st.progress(0.0)

    stage_box.info("Stage 1/3: Loading telemetry and video metadata")
    status_box.info("Loading telemetry...")
    progress_text_box.write("Preparing inputs...")
    overall_progress_bar.progress(0.05)

    try:
        tdf = pd.read_csv(telemetry_path)
        spd_mps, speed_source = pick_speed_column(tdf)
        spd_kmh = np.array([kmh_from_mps(v) for v in spd_mps], dtype=float)
        spd_kmh = smooth_1d(spd_kmh, win=11)

        tilt_cols = get_tilt_columns(tdf)
        tilt_columns_used = {
            "accel_x": tilt_cols["ax_name"],
            "accel_y": tilt_cols["ay_name"],
            "accel_z": tilt_cols["az_name"],
            "gyro_x": tilt_cols["gx_name"],
            "gyro_y": tilt_cols["gy_name"],
            "gyro_z": tilt_cols["gz_name"],
        }
    except Exception as e:
        st.error("Failed to read telemetry CSV or compute tilt columns.")
        st.exception(e)
        st.stop()

    cap_meta = cv2.VideoCapture(video_path)
    fps = cap_meta.get(cv2.CAP_PROP_FPS)
    W = int(cap_meta.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap_meta.get(cv2.CAP_PROP_FRAME_HEIGHT))
    N = int(cap_meta.get(cv2.CAP_PROP_FRAME_COUNT))
    cap_meta.release()

    if fps <= 0 or W <= 0 or H <= 0:
        st.error("Could not read FPS/size from video.")
        st.stop()

    tilt_dt = estimate_dt(len(tdf), fps=fps, num_frames=N)
    pitch_all, roll_all = compute_tilt_from_arrays(
        tilt_cols["ax"],
        tilt_cols["ay"],
        tilt_cols["az"],
        tilt_cols["gx"],
        tilt_cols["gy"],
        dt=tilt_dt,
        alpha=float(tilt_alpha),
        smooth_win=int(tilt_smooth_win),
    )

    start_frame = int(start_min * 60 * fps)
    end_frame = int(end_min * 60 * fps)
    start_frame = max(0, min(start_frame, N - 1))
    end_frame = max(start_frame + 1, min(end_frame, N))
    selected_num_frames = end_frame - start_frame

    progress_text_box.write(
        f"Video loaded. FPS: {fps:.2f} | Resolution: {W}x{H} | Total frames: {N} | "
        f"Selected window: frames {start_frame} to {end_frame}"
    )
    overall_progress_bar.progress(0.10)

    stage_box.info("Stage 2/3: Running automatic speed sign detection")
    status_box.info("Running detector + binary classifier in backend...")
    progress_text_box.write("Detecting speed signs using detector + classifier...")
    overall_progress_bar.progress(0.15)
    stage_progress_bar.progress(0.0)

    sign_raw_csv = "outputs/speed_sign_raw_detections.csv"
    sign_events_csv = "outputs/speed_sign_events.csv"
    sign_debug_video = "outputs/speed_sign_debug.mp4"
    sign_crops_dir = "outputs/speed_sign_crops"

    def sign_progress_callback(frame_id, total_frames, raw_count, stage_text):
        frac = min(1.0, frame_id / max(total_frames, 1))
        stage_progress_bar.progress(frac)
        progress_text_box.write(
            f"{stage_text} | Processed frame: {frame_id}/{total_frames} | Raw detections found: {raw_count}"
        )
        live_backend_box.info(
            f"""
**Backend Status**  
**Current Stage:** Speed sign detection  
**Processed Frame:** {frame_id}/{total_frames}  
**Raw Detections So Far:** {raw_count}  
**Detection Device:** {get_best_device()}
"""
        )

        if os.path.exists(sign_raw_csv):
            try:
                temp_df = pd.read_csv(sign_raw_csv)
                if len(temp_df) > 0:
                    show_cols = [
                        c for c in [
                            "frame", "time_sec", "class_name",
                            "ocr_speed_limit", "ocr_confidence", "ocr_text", "track_id"
                        ]
                        if c in temp_df.columns
                    ]
                    raw_sign_table_box.write("### Latest Raw Sign Reads")
                    raw_sign_table_box.dataframe(
                        temp_df[show_cols].tail(10),
                        use_container_width=True,
                        hide_index=True
                    )
                else:
                    raw_sign_table_box.info("No raw sign detections yet.")
            except Exception:
                raw_sign_table_box.info("Reading raw sign table...")

    try:
        raw_sign_df, stable_sign_df = run_speed_sign_detector(
            front_video_path=video_path,
            model_path=speed_sign_model_path,
            detector_model_path=speed_sign_model_path,
            classifier_model_path="models/best_classifier.pth",
            class_to_idx_path="models/class_to_idx.json",
            out_raw_csv_path=sign_raw_csv,
            out_events_csv_path=sign_events_csv,
            out_debug_video_path=sign_debug_video,
            crops_dir=sign_crops_dir,
            process_every=sign_process_every,
            max_frames=end_frame,
            conf_thres=sign_conf_thres,
            save_crops=save_sign_crops,
            min_repeat=sign_min_repeat,
            max_gap_frames=sign_max_gap_frames,
            progress_callback=sign_progress_callback,
            start_frame=start_frame,
            classifier_min_conf=0.60,
            device=get_best_device(),
        )

        sign_df_all = load_speed_sign_events(sign_events_csv)
        sign_df = filter_sign_events_for_window(sign_df_all, start_frame, end_frame)
        stable_sign_df = sign_df.copy()
        sign_event_map = build_sign_event_map(sign_df)
    except Exception as e:
        st.error("Automatic speed sign detection failed.")
        st.exception(e)
        st.stop()

    stage_progress_bar.progress(1.0)
    status_box.success(
        f"Speed sign detection finished. Raw detections: {len(raw_sign_df)} | Stable sign events in window/context: {len(stable_sign_df)}"
    )
    progress_text_box.write("Stable speed sign events built successfully.")
    overall_progress_bar.progress(0.40)

    if len(raw_sign_df) > 0:
        show_cols = [
            c for c in [
                "frame", "time_sec", "class_name",
                "ocr_speed_limit", "ocr_confidence", "ocr_text", "track_id"
            ]
            if c in raw_sign_df.columns
        ]
        raw_sign_table_box.write("### Final Raw Sign Reads")
        raw_sign_table_box.dataframe(
            raw_sign_df[show_cols].tail(20),
            use_container_width=True,
            hide_index=True
        )
    else:
        raw_sign_table_box.warning("No raw sign detections were found.")

    if len(stable_sign_df) > 0:
        stable_sign_table_box.write("### Stable Speed Sign Events")
        stable_sign_table_box.dataframe(
            stable_sign_df,
            use_container_width=True,
            hide_index=True
        )
    else:
        stable_sign_table_box.info("No stable speed sign events yet.")

    stage_box.info("Stage 3/3: Running main dashboard processing")
    status_box.info("Running main violation detection...")
    progress_text_box.write("Processing front video, red-light logic, speed fusion, scoring, and tilt...")
    stage_progress_bar.progress(0.0)

    cap = cv2.VideoCapture(video_path)
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    try:
        model = YOLO(front_model_path)
    except Exception as e:
        st.error(f"Could not load front model: {front_model_path}")
        st.exception(e)
        cap.release()
        st.stop()

    device_name = get_best_device()
    tl_cls_id = find_class_id(model.names, "traffic light")

    out_path = "outputs/dashboard_annotated.mp4"
    writer = None
    if save_annotated_video:
        fourcc = cv2.VideoWriter_fourcc(*"avc1")
        writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))
        if not writer.isOpened():
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(out_path, fourcc, fps, (W, H))

    if len(spd_kmh) > 0:
        full_times_sec = np.linspace(0, max(N - 1, 0) / fps, len(spd_kmh))
        cumulative_distance_full = compute_cumulative_distance_m(full_times_sec, spd_kmh)
    else:
        cumulative_distance_full = np.array([], dtype=float)

    tilt_times_sec = np.linspace(0, max(N - 1, 0) / fps, len(pitch_all)) if len(pitch_all) > 0 else np.array([])
    tilt_window_mask = (tilt_times_sec >= (start_frame / fps)) & (tilt_times_sec <= (end_frame / fps)) if len(tilt_times_sec) > 0 else np.array([], dtype=bool)
    pitch_window = pitch_all[tilt_window_mask] if len(pitch_all) > 0 else np.array([])
    roll_window = roll_all[tilt_window_mask] if len(roll_all) > 0 else np.array([])
    tilt_window_times = tilt_times_sec[tilt_window_mask] if len(tilt_times_sec) > 0 else np.array([])

    latest_sign_limit = None
    latest_sign_distance_m = None

    if len(sign_df) > 0:
        pre_start = sign_df[sign_df["frame"] <= start_frame]
        if len(pre_start) > 0:
            latest_sign_limit = float(pre_start.iloc[-1]["speed_limit"])
            ti0 = int(start_frame * (len(spd_kmh) - 1) / max(N - 1, 1))
            ti0 = max(0, min(ti0, len(spd_kmh) - 1))
            latest_sign_distance_m = float(cumulative_distance_full[ti0]) if len(cumulative_distance_full) > 0 else 0.0

    state_window = deque(maxlen=red_hold)

    fsm = 0
    last_trigger_time = -1e9

    overspeed_on = False
    overspeed_cur = None
    overspeed_max = 0.0

    high_pitch_events = 0
    high_roll_events = 0
    pitch_above_prev = False
    roll_above_prev = False

    violations = []
    overspeed_events = []
    red_violation_count = 0

    col1, col2 = st.columns([2.0, 1.1])
    with col1:
        frame_slot = st.empty()
        tilt_plot_slot = st.empty()
    with col2:
        score_slot = st.empty()
        verdict_slot = st.empty()
        active_slot = st.empty()
        tilt_metric_slot = st.empty()
        counters_slot = st.empty()
        progress_slot = st.empty()

    score = float(start_score)

    last_tl_state = "UNKNOWN"
    last_tl_conf = 0.0
    last_near_intersection = False

    for i in range(selected_num_frames):
        ret, frame = cap.read()
        if not ret:
            break

        global_frame = start_frame + i
        t_sec = global_frame / fps
        dt = 1.0 / fps

        ti = int(global_frame * (len(spd_kmh) - 1) / max(N - 1, 1))
        ti = max(0, min(ti, len(spd_kmh) - 1))
        speed_now = float(spd_kmh[ti])

        tilt_i = int(global_frame * (len(pitch_all) - 1) / max(N - 1, 1))
        tilt_i = max(0, min(tilt_i, len(pitch_all) - 1))
        pitch_now = float(pitch_all[tilt_i])
        roll_now = float(roll_all[tilt_i])

        current_distance_m = float(cumulative_distance_full[ti]) if len(cumulative_distance_full) > 0 else 0.0

        if global_frame in sign_event_map:
            latest_sign_limit = float(sign_event_map[global_frame])
            latest_sign_distance_m = current_distance_m

        if latest_sign_limit is not None and latest_sign_distance_m is not None:
            distance_since_last_sign_m = current_distance_m - latest_sign_distance_m
        else:
            distance_since_last_sign_m = np.nan

        if (
            latest_sign_limit is not None
            and latest_sign_distance_m is not None
            and distance_since_last_sign_m <= float(sign_valid_distance_m)
        ):
            active_speed_limit = float(latest_sign_limit)
            limit_source = "SIGN"
        else:
            active_speed_limit = float(fallback_default_limit)
            limit_source = "DEFAULT"

        do_process = (i % int(process_every_k) == 0)

        annotated = frame.copy()
        boxes = None

        if do_process:
            try:
                results = model.track(
                    frame,
                    conf=conf,
                    imgsz=imgsz,
                    device=device_name,
                    tracker="bytetrack.yaml",
                    persist=True,
                    verbose=False
                )
                annotated = results[0].plot()
                boxes = results[0].boxes
            except Exception:
                try:
                    results = model.track(
                        frame,
                        conf=conf,
                        imgsz=imgsz,
                        device="cpu",
                        tracker="bytetrack.yaml",
                        persist=True,
                        verbose=False
                    )
                    annotated = results[0].plot()
                    boxes = results[0].boxes
                except Exception as e:
                    st.error("Front model inference failed.")
                    st.exception(e)
                    cap.release()
                    if writer is not None:
                        writer.release()
                    st.stop()

        put_text_outline(
            annotated,
            f"Speed: {speed_now:.1f} km/h | Limit: {active_speed_limit:.0f} ({limit_source})",
            (20, 40),
            scale=0.9
        )

        if not np.isnan(distance_since_last_sign_m):
            put_text_outline(
                annotated,
                f"Distance since sign: {distance_since_last_sign_m:.1f} m",
                (20, 70),
                scale=0.8
            )

        tilt_status = get_tilt_status(
            pitch_now,
            roll_now,
            pitch_th=float(pitch_warn_deg),
            roll_th=float(roll_warn_deg),
        )

        tilt_color = (0, 0, 255) if (
            abs(pitch_now) >= float(pitch_warn_deg) or abs(roll_now) >= float(roll_warn_deg)
        ) else (255, 255, 255)

        put_text_outline(
            annotated,
            f"Driving: {tilt_status}",
            (20, 100),
            scale=0.75,
            color=tilt_color
        )

        pitch_above_now = abs(pitch_now) >= float(pitch_warn_deg)
        roll_above_now = abs(roll_now) >= float(roll_warn_deg)

        if pitch_above_now and not pitch_above_prev:
            high_pitch_events += 1
            pitch_msg = (
                f"Hard Braking / Uphill detected (Pitch: +{pitch_now:.2f}°)"
                if pitch_now > 0
                else f"Rapid Acceleration / Downhill detected (Pitch: {pitch_now:.2f}°)"
            )
            violations.append({
                "time_sec": float(t_sec),
                "frame": int(global_frame),
                "violation_type": "high_pitch",
                "severity": "LOW",
                "value": float(pitch_now),
                "message": pitch_msg,
            })

        if roll_above_now and not roll_above_prev:
            high_roll_events += 1
            roll_msg = (
                f"Turning Right detected (Roll: +{roll_now:.2f}°)"
                if roll_now > 0
                else f"Turning Left detected (Roll: {roll_now:.2f}°)"
            )
            violations.append({
                "time_sec": float(t_sec),
                "frame": int(global_frame),
                "violation_type": "high_roll",
                "severity": "LOW",
                "value": float(roll_now),
                "message": roll_msg,
            })

        pitch_above_prev = pitch_above_now
        roll_above_prev = roll_above_now

        is_overspeed = speed_now > active_speed_limit

        if is_overspeed:
            if not overspeed_on:
                overspeed_on = True
                overspeed_cur = {
                    "start_frame": int(global_frame),
                    "start_time_sec": float(t_sec),
                    "speed_limit_kmh": float(active_speed_limit),
                    "limit_source": limit_source
                }
                overspeed_max = speed_now
            else:
                overspeed_max = max(overspeed_max, speed_now)
        else:
            if overspeed_on and overspeed_cur is not None:
                overspeed_on = False
                overspeed_cur["end_frame"] = int(global_frame)
                overspeed_cur["end_time_sec"] = float(t_sec)
                overspeed_cur["duration_sec"] = overspeed_cur["end_time_sec"] - overspeed_cur["start_time_sec"]
                overspeed_cur["max_speed_kmh"] = float(overspeed_max)
                overspeed_cur["message"] = (
                    f"Overspeed detected: max {overspeed_cur['max_speed_kmh']:.1f} km/h "
                    f"against limit {overspeed_cur['speed_limit_kmh']:.1f} km/h ({overspeed_cur['limit_source']})"
                )
                overspeed_events.append(overspeed_cur)

                violations.append({
                    "time_sec": float(overspeed_cur["start_time_sec"]),
                    "frame": int(overspeed_cur["start_frame"]),
                    "violation_type": "overspeed",
                    "severity": "MEDIUM",
                    "value": float(overspeed_cur["max_speed_kmh"]),
                    "message": overspeed_cur["message"],
                })

                score -= float(overspeed_event_penalty)
                overspeed_cur = None
                overspeed_max = 0.0

        tl_state, tl_conf = last_tl_state, last_tl_conf
        near_intersection = last_near_intersection

        if do_process and tl_cls_id is not None and boxes is not None and boxes.xyxy is not None:
            xyxy = boxes.xyxy.cpu().numpy()
            cls = boxes.cls.cpu().numpy().astype(int)
            confs = boxes.conf.cpu().numpy().astype(float)

            best = None
            for b, c, cf in zip(xyxy, cls, confs):
                if c != tl_cls_id:
                    continue
                x1, y1, x2, y2 = b.tolist()
                area = max(0.0, (x2 - x1)) * max(0.0, (y2 - y1))
                if area < float(min_tl_area):
                    continue
                if best is None or cf > best[0]:
                    best = (cf, (x1, y1, x2, y2))

            if best is not None:
                _, (x1, y1, x2, y2) = best
                x1, y1, x2, y2 = clamp_box(x1, y1, x2, y2, W, H)

                roi = frame[y1:y2, x1:x2]
                tl_state, tl_conf = classify_signal_color(roi)

                tl_h_ratio = (y2 - y1) / float(H)
                near_intersection = tl_h_ratio >= float(near_ratio)

                color = (0, 0, 255) if tl_state == "RED" else (255, 255, 255)
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

                last_tl_state = tl_state
                last_tl_conf = tl_conf
                last_near_intersection = near_intersection

        sig_color = (0, 0, 255) if tl_state == "RED" else (255, 255, 255)
        put_text_outline(
            annotated,
            f"SIGNAL: {tl_state} ({tl_conf:.2f})",
            (20, 140),
            scale=0.9,
            color=sig_color
        )

        if tl_state != "UNKNOWN":
            state_window.append(tl_state)

        red_count = sum(1 for s in state_window if s == "RED")
        red_stable = (len(state_window) == state_window.maxlen) and (red_count == state_window.maxlen)
        cooldown_ok = (t_sec - last_trigger_time) >= float(cooldown_sec)

        red_triggered_now = False
        if fsm == 0:
            if red_stable and near_intersection:
                fsm = 1
        elif fsm == 1:
            if not red_stable:
                fsm = 0
            else:
                if speed_now >= float(min_move_kmh) and cooldown_ok:
                    last_trigger_time = float(t_sec)
                    red_triggered_now = True
                    red_violation_count += 1
                    score -= float(red_penalty)

                    violations.append({
                        "time_sec": float(t_sec),
                        "frame": int(global_frame),
                        "violation_type": "red_light_violation",
                        "severity": "HIGH",
                        "value": float(speed_now),
                        "message": "Moved while signal was RED.",
                    })
                    fsm = 0

        if overspeed_on:
            score -= float(overspeed_per_sec_penalty) * dt

        score = max(0.0, min(100.0, score))

        active = []
        if is_overspeed:
            active.append(f"OVERSPEEDING: {speed_now:.1f} > {active_speed_limit:.0f} km/h ({limit_source})")

        if tl_state == "RED" and near_intersection and speed_now >= float(min_move_kmh):
            active.append("WARNING: MOVING ON RED")
        if red_triggered_now:
            active.append("RED LIGHT VIOLATION!")

        center_alert = ""
        if red_triggered_now:
            center_alert = "CROSSED RED SIGNAL!"
        elif tl_state == "RED" and near_intersection and speed_now >= float(min_move_kmh):
            center_alert = "STOP! RED SIGNAL"
        elif is_overspeed:
            center_alert = f"OVERSPEED: {speed_now:.0f} > {active_speed_limit:.0f} KM/H"

        draw_center_alert(annotated, center_alert)

        y0 = 205
        if active:
            for k, msg in enumerate(active[:4]):
                put_text_outline(
                    annotated,
                    msg,
                    (20, y0 + 35 * k),
                    scale=0.85,
                    color=(0, 0, 255)
                )
        else:
            put_text_outline(annotated, "DRIVING SAFE", (20, y0), scale=1.0)

        if writer is not None:
            writer.write(annotated)

        if i % int(show_every_n) == 0:
            rgb = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)
            frame_slot.image(rgb, caption=f"Frame {global_frame}  (t={t_sec:.2f}s)", use_container_width=True)

            if show_tilt_plot and len(tilt_window_times) > 0:
                local_idx = int((t_sec - tilt_window_times[0]) / max((tilt_window_times[-1] - tilt_window_times[0]) / max(len(tilt_window_times) - 1, 1), 1e-6))
                local_idx = max(0, min(local_idx, len(tilt_window_times) - 1))
                tilt_plot_slot.plotly_chart(
                    tilt_timeseries_figure(tilt_window_times, pitch_window, roll_window, local_idx),
                    use_container_width=True,
                    key=f"tilt_plot_{st.session_state.run_id}_{i}"
                )

            gauge_key = f"score_gauge_{st.session_state.run_id}_{i}"
            score_slot.plotly_chart(score_gauge(score), use_container_width=True, key=gauge_key)

            verdict = verdict_from_score(score)
            verdict_slot.metric("Verdict", verdict)

            active_slot.write("### Active Status")

            tilt_status_now = get_tilt_status(
                pitch_now,
                roll_now,
                pitch_th=float(pitch_warn_deg),
                roll_th=float(roll_warn_deg),
            )

            status_text = (
                f"**Speed:** {speed_now:.1f} km/h  \n"
                f"**Active Limit:** {active_speed_limit:.0f} km/h  \n"
                f"**Limit Source:** {limit_source}  \n"
                f"**Driving State:** {tilt_status_now}  \n"
                f"**Distance Since Sign:** {0.0 if np.isnan(distance_since_last_sign_m) else distance_since_last_sign_m:.1f} m  \n"
            )

            if active:
                active_slot.error(status_text + f"**Status:** {' | '.join(active)}")
            else:
                active_slot.success(status_text + "**Status:** SAFE")

            tilt_metric_slot.write("### Driving Interpretation")
            tilt_metric_slot.info(tilt_status_now)

            counters_slot.write("### Counters")
            counters_slot.write(
                f"- Stable sign events: **{len(stable_sign_df)}**\n"
                f"- Overspeed events: **{len(overspeed_events) + (1 if overspeed_on else 0)}**\n"
                f"- Red light violations: **{red_violation_count}**\n"
                f"- High pitch events: **{high_pitch_events}**\n"
                f"- High roll events: **{high_roll_events}**"
            )

            progress_slot.progress(min(1.0, (i + 1) / selected_num_frames))
            stage_progress_bar.progress(min(1.0, (i + 1) / selected_num_frames))
            overall_progress_bar.progress(0.40 + 0.55 * min(1.0, (i + 1) / selected_num_frames))

            progress_text_box.write(
                f"Running main dashboard processing... Frame {i + 1}/{selected_num_frames} | "
                f"Overspeed events: {len(overspeed_events) + (1 if overspeed_on else 0)} | "
                f"Red-light violations: {red_violation_count} | "
                f"High pitch events: {high_pitch_events} | "
                f"High roll events: {high_roll_events}"
            )

            live_backend_box.info(
                f"""
**Backend Status**  
**Current Stage:** Main dashboard processing  
**Frame:** {i + 1}/{selected_num_frames}  
**Speed:** {speed_now:.1f} km/h  
**Active Limit:** {active_speed_limit:.1f} km/h  
**Limit Source:** {limit_source}  
**Pitch:** {pitch_now:+.2f} deg  
**Roll:** {roll_now:+.2f} deg  
**Stable Sign Events:** {len(stable_sign_df)}  
**Overspeed Events:** {len(overspeed_events) + (1 if overspeed_on else 0)}  
**Red Light Violations:** {red_violation_count}  
**Detection Device:** {device_name}
"""
            )

    cap.release()
    if writer is not None:
        writer.release()

    if overspeed_on and overspeed_cur is not None:
        overspeed_cur["end_frame"] = int(start_frame + selected_num_frames - 1)
        overspeed_cur["end_time_sec"] = float((start_frame + selected_num_frames - 1) / fps)
        overspeed_cur["duration_sec"] = overspeed_cur["end_time_sec"] - overspeed_cur["start_time_sec"]
        overspeed_cur["max_speed_kmh"] = float(overspeed_max)
        overspeed_cur["message"] = (
            f"Overspeed detected: max {overspeed_cur['max_speed_kmh']:.1f} km/h "
            f"against limit {overspeed_cur['speed_limit_kmh']:.1f} km/h ({overspeed_cur['limit_source']})"
        )
        overspeed_events.append(overspeed_cur)

        violations.append({
            "time_sec": float(overspeed_cur["start_time_sec"]),
            "frame": int(overspeed_cur["start_frame"]),
            "violation_type": "overspeed",
            "severity": "MEDIUM",
            "value": float(overspeed_cur["max_speed_kmh"]),
            "message": overspeed_cur["message"],
        })

        score -= float(overspeed_event_penalty)
        score = max(0.0, min(100.0, score))

    stage_progress_bar.progress(1.0)
    overall_progress_bar.progress(1.0)
    stage_box.success("All stages completed")
    status_box.success("Processing complete.")
    progress_text_box.write("Generating final report, summary tables, and annotated video...")
    live_backend_box.success("Backend processing finished successfully.")

    report = {
        "trip_name": f"dashboard_min{start_min}_to_min{end_min}",
        "score_0_100": round(float(score), 2),
        "verdict": verdict_from_score(score),
        "telemetry_source": speed_source,
        "tilt_columns_used": str(tilt_columns_used),
        "tilt_alpha": float(tilt_alpha),
        "tilt_smooth_win": int(tilt_smooth_win),
        "fallback_default_limit": float(fallback_default_limit),
        "sign_valid_distance_m": float(sign_valid_distance_m),
        "summary": {
            "overspeed_events": int(len(overspeed_events)),
            "overspeed_total_duration_sec": round(
                float(sum([e.get("duration_sec", 0.0) or 0.0 for e in overspeed_events])), 2
            ),
            "overspeed_max_speed_kmh": round(
                float(max([e.get("max_speed_kmh", 0.0) or 0.0 for e in overspeed_events] + [0.0])), 2
            ),
            "red_light_violations": int(red_violation_count),
            "high_pitch_events": int(high_pitch_events),
            "high_roll_events": int(high_roll_events),
            "max_abs_pitch_deg": round(float(np.max(np.abs(pitch_window))) if len(pitch_window) > 0 else 0.0, 2),
            "max_abs_roll_deg": round(float(np.max(np.abs(roll_window))) if len(roll_window) > 0 else 0.0, 2),
        },
        "suggestions": [
            "Maintain speed within the active detected speed limit."
            if len(overspeed_events) > 0 else
            "Good speed control.",
            "Stop fully at RED signals and proceed only on GREEN."
            if red_violation_count > 0 else
            "Good signal compliance.",
            "High pitch was observed. Check harsh braking / acceleration behavior and road slope sections."
            if high_pitch_events > 0 else
            "Pitch remained within the selected threshold.",
            "High roll was observed. Check sharp turns, lane changes, or body sway."
            if high_roll_events > 0 else
            "Roll remained within the selected threshold."
        ],
        "violations": violations
    }

    st.success("✅ Processing finished!")
    st.write("## Final Report")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Telemetry Source", speed_source)
    c2.metric("Score", f"{report['score_0_100']:.2f}")
    c3.metric("Verdict", report["verdict"])
    c4.metric("Red Violations", report["summary"]["red_light_violations"])

    if report["verdict"] == "SAFE":
        st.success("✅ Overall driving is SAFE.")
    elif report["verdict"] == "MODERATE_RISK":
        st.warning("⚠️ Driving has MODERATE risk. Improve compliance.")
    else:
        st.error("🛑 Driving is UNSAFE. Please drive carefully.")

    st.write("---")
    st.subheader("Tilt Summary")
    t1, t2, t3, t4 = st.columns(4)
    t1.metric("High Pitch Events", report["summary"]["high_pitch_events"])
    t2.metric("High Roll Events", report["summary"]["high_roll_events"])
    t3.metric("Max |Pitch|", f"{report['summary']['max_abs_pitch_deg']:.2f}°")
    t4.metric("Max |Roll|", f"{report['summary']['max_abs_roll_deg']:.2f}°")

    if show_tilt_plot and len(tilt_window_times) > 0:
        st.plotly_chart(
            tilt_timeseries_figure(tilt_window_times, pitch_window, roll_window, len(tilt_window_times) - 1),
            use_container_width=True,
            key=f"tilt_final_{st.session_state.run_id}"
        )

    st.write("---")
    st.subheader("Speed Rule Summary")
    s1, s2, s3 = st.columns(3)
    s1.metric("Fallback Limit", f"{fallback_default_limit:.0f} km/h")
    s2.metric("Sign Valid Distance", f"{sign_valid_distance_m:.0f} m")
    s3.metric("Stable Sign Events", len(stable_sign_df))

    st.write("---")
    st.subheader("Tilt Telemetry Columns Used")
    st.json(tilt_columns_used)

    st.write("---")
    st.subheader("Suggestions")
    for s in report.get("suggestions", []):
        st.info(s)

    st.write("---")
    st.subheader("Detected Stable Speed Sign Events")
    if len(stable_sign_df) == 0:
        st.warning("No stable speed sign events were detected. Fallback limit was used.")
    else:
        st.dataframe(stable_sign_df, use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("Raw Sign Reads")
    if len(raw_sign_df) == 0:
        st.warning("No raw sign detections were found.")
    else:
        show_cols = [
            c for c in [
                "frame", "time_sec", "class_name",
                "ocr_speed_limit", "ocr_confidence", "ocr_text", "track_id"
            ]
            if c in raw_sign_df.columns
        ]
        st.dataframe(raw_sign_df[show_cols].tail(50), use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("Overspeed Events")
    if len(overspeed_events) == 0:
        st.success("No overspeed events detected.")
    else:
        st.dataframe(pd.DataFrame(overspeed_events), use_container_width=True, hide_index=True)

    st.write("---")
    st.subheader("Violations Table")
    if len(violations) == 0:
        st.success("No violations logged.")
    else:
        vdf = pd.DataFrame(violations)
        keep = [c for c in ["time_sec", "frame", "violation_type", "severity", "value", "message"] if c in vdf.columns]
        st.dataframe(vdf[keep], use_container_width=True, hide_index=True)

    pdf_bytes = report_to_pdf_bytes(report)
    st.download_button(
        "⬇️ Download Final Report (PDF)",
        data=pdf_bytes,
        file_name=f"{report['trip_name']}_report.pdf",
        mime="application/pdf"
    )

    st.write("---")
    st.subheader("Annotated Video")
    if save_annotated_video and os.path.exists(out_path) and os.path.getsize(out_path) > 0:
        st.caption(f"Saved at: {out_path} ({os.path.getsize(out_path)/1024/1024:.2f} MB)")
        with open(out_path, "rb") as f:
            video_bytes = f.read()
        st.video(video_bytes)
    else:
        st.info("Annotated video not saved (or file is empty).")

else:
    st.info("Set inputs and click **Run Dashboard Processing**.")