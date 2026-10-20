"""
Pose-estimation engine for SportsSight AI (MediaPipe PoseLandmarker, Tasks API).

- Extracts 33 body landmarks per sampled frame.
- Turns landmark time-series into sport-aware metrics (joint angles, sway, speed,
  direction changes, recovery time, consistency).
- Falls back gracefully: if MediaPipe or the model is unavailable, `load_landmarker`
  returns None and main.py uses the older OpenCV heuristics.

The model file (pose_landmarker_*.task) is downloaded automatically on first use into
./models/ . Set POSE_MODEL_PATH to use your own copy (offline / production), and
POSE_MODEL_VARIANT = lite | full | heavy (default: full).
"""
from __future__ import annotations

import math
import os
import shutil
import time
import urllib.request
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

MODEL_DIR = Path(__file__).parent / "models"
VARIANT = os.getenv("POSE_MODEL_VARIANT", "full").lower()
MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/pose_landmarker/"
    f"pose_landmarker_{VARIANT}/float16/latest/pose_landmarker_{VARIANT}.task"
)

# MediaPipe landmark indices
L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK = 11, 12, 23, 24, 25, 26, 27, 28
KEY_JOINTS = [L_SH, R_SH, L_HIP, R_HIP, L_KNEE, R_KNEE, L_ANK, R_ANK]

MIN_VISIBILITY = 0.5
MIN_VALID_FRAMES_VIDEO = 6
TARGET_SAMPLES = 150          # max frames we run pose on for one video
LOAD_ERROR: Optional[str] = None
_last_download_fail = 0.0       # avoid re-trying a failing download on every request
DOWNLOAD_RETRY_SECONDS = 120
DOWNLOAD_TIMEOUT = 25


# --------------------------------------------------------------------------- model loading
def _model_path() -> Optional[Path]:
    global LOAD_ERROR
    custom = os.getenv("POSE_MODEL_PATH")
    if custom:
        p = Path(custom)
        if p.exists():
            return p
        LOAD_ERROR = f"POSE_MODEL_PATH not found: {custom}"
        return None
    p = MODEL_DIR / f"pose_landmarker_{VARIANT}.task"
    if p.exists() and p.stat().st_size > 1_000_000:
        return p
    global _last_download_fail
    if time.time() - _last_download_fail < DOWNLOAD_RETRY_SECONDS:
        return None  # LOAD_ERROR already describes the last failure
    try:
        MODEL_DIR.mkdir(exist_ok=True)
        tmp = p.with_suffix(".part")
        with urllib.request.urlopen(MODEL_URL, timeout=DOWNLOAD_TIMEOUT) as resp, open(tmp, "wb") as out:
            shutil.copyfileobj(resp, out)
        tmp.replace(p)
        return p
    except Exception as e:  # no internet, blocked, etc.
        _last_download_fail = time.time()
        LOAD_ERROR = f"Could not download pose model ({e}). Place the .task file in ./models or set POSE_MODEL_PATH."
        return None


def load_landmarker(video: bool):
    """Return a PoseLandmarker or None if unavailable."""
    global LOAD_ERROR
    try:
        import mediapipe as mp  # noqa: F401
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision
    except Exception as e:
        LOAD_ERROR = f"mediapipe not installed ({e})"
        return None
    path = _model_path()
    if path is None:
        return None
    try:
        opts = vision.PoseLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_poses=1,
            min_pose_detection_confidence=0.5,
            min_pose_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        return vision.PoseLandmarker.create_from_options(opts)
    except Exception as e:
        LOAD_ERROR = f"Could not start pose model ({e})"
        return None


def _to_array(result) -> Optional[np.ndarray]:
    """PoseLandmarkerResult -> (33, 3) array of x, y, visibility (normalised image coords)."""
    if not result or not result.pose_landmarks:
        return None
    lms = result.pose_landmarks[0]
    return np.array([[lm.x, lm.y, getattr(lm, "visibility", 1.0) or 0.0] for lm in lms], dtype=float)


def _valid(arr: Optional[np.ndarray]) -> bool:
    return arr is not None and float(np.mean(arr[KEY_JOINTS, 2])) >= MIN_VISIBILITY


# --------------------------------------------------------------------------- extraction
def extract_image(path: str):
    import mediapipe as mp
    img = cv2.imread(path)
    if img is None:
        return None, None
    lm = load_landmarker(video=False)
    if lm is None:
        return None, None
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        res = lm.detect(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb))
        arr = _to_array(res)
        sharp = float(cv2.Laplacian(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var())
        return ([arr] if _valid(arr) else []), {"sharpness": sharp, "frames_total": 1, "dt": None, "duration": None, "fps": None}
    finally:
        lm.close()


def extract_video(path: str):
    """Returns (list_of_valid_landmark_arrays, info) or (None, None) if the engine is unavailable."""
    import mediapipe as mp
    lm = load_landmarker(video=True)
    if lm is None:
        return None, None
    cap = cv2.VideoCapture(path)
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        if fps <= 0 or fps > 240:
            fps = 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        if total <= 0 or total > 10_000_000:  # webm from browsers often has no frame count
            total = 0
        stride = max(1, int(round(total / TARGET_SAMPLES))) if total else max(1, int(round(fps / 8)))
        dt = stride / fps
        frames, sharps, idx, sampled, last_ts = [], [], 0, 0, -1
        while sampled < TARGET_SAMPLES * 2:
            if not cap.grab():
                break
            idx += 1
            if (idx - 1) % stride:
                continue
            ok, frame = cap.retrieve()
            if not ok:
                break
            sampled += 1
            h, w = frame.shape[:2]
            if max(h, w) > 720:
                s = 720 / max(h, w)
                frame = cv2.resize(frame, (int(w * s), int(h * s)))
            ts = max(last_ts + 1, int(round(((idx - 1) / fps) * 1000)))
            last_ts = ts
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ts)
            arr = _to_array(res)
            if _valid(arr):
                frames.append(arr)
                sharps.append(float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()))
        info = {
            "sharpness": float(np.mean(sharps)) if sharps else 0.0,
            "frames_total": sampled,
            "dt": dt,
            "fps": fps,
            "duration": round(idx / fps, 1) if idx else None,
        }
        return frames, info
    finally:
        cap.release()
        lm.close()


# --------------------------------------------------------------------------- metrics
def _clamp(v, lo=0, hi=100):
    return int(max(lo, min(hi, round(float(v)))))


def _label(score):
    if score >= 85: return "Excellent"
    if score >= 75: return "Strong"
    if score >= 62: return "Good"
    if score >= 50: return "Developing"
    return "Needs work"


def _angle(a, b, c):
    """Angle ABC in degrees (b is the vertex). Inputs are (N,2) arrays."""
    ba, bc = a - b, c - b
    cosang = np.sum(ba * bc, axis=1) / (np.linalg.norm(ba, axis=1) * np.linalg.norm(bc, axis=1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cosang, -1, 1)))


# Ideal ranges per sport (average knee angle in degrees, trunk lean in degrees from vertical)
SPORT_TARGETS = {
    "Kabaddi":   {"knee": 135, "lean": 20, "width": 1.6},
    "Wrestling": {"knee": 125, "lean": 25, "width": 1.8},
    "Sprint":    {"knee": 150, "lean": 15, "width": None},
}

SPORT_DRILLS = {
    "Kabaddi": [
        "Cone-weave raids: 5 sets x 6 cones, keep knees bent and hips low.",
        "Resisted retreats: 5 x 10 m, twice a week, to speed up recovery after a raid.",
        "Single-leg balance holds: 3 x 30 s per leg to improve stability on turns.",
    ],
    "Wrestling": [
        "Stance-and-motion drill: 4 x 2 min holding knees bent and hips low.",
        "Level-change shots: 5 x 8 reps, focus on fast, balanced entries.",
        "Core stability circuit: planks and Russian twists, 3 rounds.",
    ],
    "Sprint": [
        "Block starts: 6 x 20 m, focus on an explosive first 3 steps.",
        "A-skips and B-skips: 3 x 20 m to improve knee drive and stride mechanics.",
        "Acceleration runs: 5 x 30 m with full recovery between reps.",
    ],
}


def compute_pose_report(frames: list, info: dict, sport: str, meta: dict) -> Optional[dict]:
    """
    frames: list of (33,3) landmark arrays (only valid frames).
    info:   {"dt": seconds between samples or None for a single image, "sharpness":..., "frames_total":...}
    Returns a report dict in the same shape the frontend already renders, or None if not enough data.
    """
    sport = sport if sport in SPORT_TARGETS else "Kabaddi"
    tgt = SPORT_TARGETS[sport]
    is_video = info.get("dt") is not None
    n = len(frames)
    if n == 0 or (is_video and n < MIN_VALID_FRAMES_VIDEO):
        return None

    A = np.stack(frames)  # (T,33,3)
    xy = A[:, :, :2]
    sh_mid = (xy[:, L_SH] + xy[:, R_SH]) / 2
    hip_mid = (xy[:, L_HIP] + xy[:, R_HIP]) / 2
    ank_mid = (xy[:, L_ANK] + xy[:, R_ANK]) / 2
    torso = np.linalg.norm(sh_mid - hip_mid, axis=1)
    torso = np.maximum(torso, 1e-3)
    torso_ref = float(np.median(torso))
    sh_width = np.maximum(np.linalg.norm(xy[:, L_SH] - xy[:, R_SH], axis=1), 0.5 * torso)

    # posture signals
    vec = sh_mid - hip_mid
    lean = np.degrees(np.arctan2(np.abs(vec[:, 0]), -vec[:, 1]))  # 0 = upright, y axis points down
    knee_l = _angle(xy[:, L_HIP], xy[:, L_KNEE], xy[:, L_ANK])
    knee_r = _angle(xy[:, R_HIP], xy[:, R_KNEE], xy[:, R_ANK])
    knee = (knee_l + knee_r) / 2
    width_ratio = np.abs(xy[:, L_ANK, 0] - xy[:, R_ANK, 0]) / sh_width
    balance_off = np.abs(hip_mid[:, 0] - ank_mid[:, 0]) / torso

    mean_knee, mean_lean = float(np.mean(knee)), float(np.mean(lean))
    mean_off, mean_width = float(np.mean(balance_off)), float(np.mean(width_ratio))

    # --- static (posture) scores
    balance = _clamp(100 - mean_off * 220 - max(0.0, mean_lean - 30) * 0.9)
    knee_score = 100 - abs(mean_knee - tgt["knee"]) * 1.3
    if tgt["width"]:
        width_score = 100 - abs(mean_width - tgt["width"]) * 35
        stance = _clamp(0.6 * knee_score + 0.4 * width_score, 30, 98)
    else:
        stance = _clamp(0.7 * knee_score + 0.3 * (100 - abs(mean_lean - tgt["lean"]) * 2.2), 30, 98)
    cv_posture = float(np.std(knee) / (mean_knee + 1e-6)) if n > 2 else 0.0
    consistency = _clamp(97 - cv_posture * 260, 30, 98)

    # --- motion scores (video only)
    motion = {}
    speed = np.array([])
    if is_video and n >= MIN_VALID_FRAMES_VIDEO:
        dt = max(float(info["dt"]), 1e-3)
        step = np.linalg.norm(np.diff(hip_mid, axis=0), axis=1) / ((torso[1:] + torso[:-1]) / 2) / dt
        k = 3 if len(step) >= 6 else 1
        speed = np.convolve(step, np.ones(k) / k, mode="same") if k > 1 else step  # torso-lengths / second

        p75, p90, p95, peak = (float(np.percentile(speed, q)) for q in (75, 90, 95, 100))
        vx = np.diff(hip_mid[:, 0]) / torso[1:]
        thr = 0.02
        sig = vx[np.abs(vx) > thr]
        dir_changes = int(np.sum(np.diff(np.sign(sig)) != 0)) if len(sig) > 1 else 0
        dur = max(dt * len(step), 1.0)
        changes_per_10s = dir_changes / dur * 10

        if sport == "Sprint":
            agility = _clamp(30 + min(p90, 5.0) / 5.0 * 65)
        else:
            agility = _clamp(30 + min(p75, 3.0) / 3.0 * 50 + min(changes_per_10s, 6.0) / 6.0 * 15)
        vert_range = float((np.percentile(hip_mid[:, 1], 95) - np.percentile(hip_mid[:, 1], 5)) / torso_ref)
        power = _clamp(35 + min(p95, 6.0) / 6.0 * 50 + min(vert_range, 1.0) * 15)

        # recovery: seconds for speed to fall under 50 % of a burst's peak (needs >= 2 bursts to be meaningful)
        rec_times = []
        if peak > 0.5:
            thr_burst = 0.6 * peak
            i = 0
            while i < len(speed):
                if speed[i] >= thr_burst:
                    j = i
                    while j < len(speed) and speed[j] >= thr_burst:
                        j += 1
                    pk = float(np.max(speed[i:j]))
                    m = j
                    while m < len(speed) and speed[m] > 0.5 * pk:
                        m += 1
                    if m < len(speed):
                        rec_times.append((m - j) * dt)
                    i = max(m, j) + 1
                else:
                    i += 1
        recovery = _clamp(100 - float(np.median(rec_times)) * 35, 35, 98) if len(rec_times) >= 2 else None
        motion = dict(agility=agility, power=power, recovery=recovery, p75=p75, p95=p95,
                      changes=changes_per_10s, vert=vert_range, rec_times=rec_times)

    # --- assemble metrics per sport
    M = []
    def add(label, score):
        if score is not None:
            M.append({"label": label, "score": int(score), "note": _label(int(score))})

    if sport == "Sprint":
        if motion:
            add("Speed & acceleration", motion["agility"]); add("Explosive power", motion["power"])
        add("Posture & knee drive", stance); add("Balance & control", balance); add("Consistency", consistency)
    elif sport == "Wrestling":
        add("Stance & posture", stance); add("Balance & control", balance)
        if motion:
            add("Attack speed", motion["power"]); add("Recovery speed", motion["recovery"])
        add("Consistency", consistency)
    else:  # Kabaddi
        if motion:
            add("Agility & footwork", motion["agility"]); add("Explosive power", motion["power"])
        add("Balance & control", balance)
        if motion:
            add("Recovery speed", motion["recovery"])
        add("Stance & posture", stance); add("Consistency", consistency)

    detect_rate = n / max(1, info.get("frames_total") or n)
    sharp = float(info.get("sharpness", 0.0))
    sharp_pct = _clamp(min(sharp, 300) / 300 * 100)
    quality_score = _clamp(0.6 * detect_rate * 100 + 0.4 * sharp_pct)
    M.append({"label": "Video quality", "score": quality_score, "note": _label(quality_score)})

    core = [m["score"] for m in M if m["label"] != "Video quality"]
    overall = int(min(max(round(np.mean(core) * (0.85 + 0.15 * min(1.0, detect_rate))), 40), 96))

    ranked = sorted([m for m in M if m["label"] != "Video quality"], key=lambda m: m["score"])
    weakest, strongest = ranked[0], ranked[-1]

    # --- feedback grounded in measured values
    strengths, improvements = [], []
    if strongest["score"] >= 62:
        strengths.append(f"{strongest['label']} is your best area ({strongest['score']}/100).")
    if abs(mean_knee - tgt["knee"]) <= 12:
        strengths.append(f"Knee bend (~{mean_knee:.0f} deg) is close to the ideal athletic range for {sport.lower()}.")
    if mean_off < 0.12:
        strengths.append("Your body weight stays well centred over your feet.")
    if consistency >= 75:
        strengths.append("Posture stays consistent through the recording.")
    if motion and motion["agility"] >= 70:
        strengths.append("Good movement speed with quick changes of direction." if sport != "Sprint" else "Strong running speed through the clip.")
    if not strengths:
        strengths.append("A valid, trackable recording - a solid baseline to improve from.")

    if weakest["score"] < 75:
        improvements.append(f"Biggest opportunity: {weakest['label']} ({weakest['score']}/100).")
    if mean_knee > tgt["knee"] + 12:
        improvements.append(f"Your stance is tall (average knee angle ~{mean_knee:.0f} deg). Bend the knees more, aiming for roughly {tgt['knee']} deg, to lower your centre of gravity.")
    elif mean_knee < tgt["knee"] - 20:
        improvements.append(f"Knees are bent very deeply (~{mean_knee:.0f} deg), which can slow movement. Aim for roughly {tgt['knee']} deg.")
    if mean_off > 0.18:
        improvements.append("Your hips drift away from your feet. Keep weight centred over a stable base, especially when changing direction.")
    if mean_lean > tgt["lean"] + 15:
        improvements.append(f"Upper body leans a lot (~{mean_lean:.0f} deg from vertical). Keep the chest tall and controlled.")
    if tgt["width"] and mean_width < tgt["width"] - 0.7:
        improvements.append("Feet are close together. A wider base (about shoulder width or more) gives better balance.")
    if tgt["width"] and mean_width > tgt["width"] + 1.0:
        improvements.append("Feet are very wide, which can slow footwork. Narrow your stance slightly.")
    if motion and motion["recovery"] is not None and motion["recovery"] < 65:
        improvements.append(f"Recovery after fast bursts is slow (~{np.median(motion['rec_times']):.1f}s to settle). Practise resisted retreats and quick resets.")
    if consistency < 60:
        improvements.append("Posture varies a lot between frames. Work on repeating the same stance and technique.")
    if not improvements:
        improvements.append("No major technical issues detected. Confirm details with a coach.")

    quality_tips = []
    if sharp < 40: quality_tips.append("Video looks blurry or dark. Use better lighting and keep the camera steady.")
    if detect_rate < 0.6: quality_tips.append("Your full body was not tracked in many frames. Record from further away with head-to-feet visible.")
    if not quality_tips: quality_tips.append("Recording quality is good for pose tracking.")

    timeline = []
    if len(speed) > 1:
        peak = float(np.max(speed)) or 1.0
        idx = np.linspace(0, len(speed) - 1, min(40, len(speed))).astype(int)
        timeline = [_clamp(speed[i] / peak * 100) for i in idx]

    summary = (
        f"Pose tracking was run on {n} frame(s) of your {sport} {'video' if is_video else 'photo'}. "
        f"Overall score {overall}/100. Strongest: {strongest['label']} ({strongest['score']}). "
        f"Biggest opportunity: {weakest['label']} ({weakest['score']}). "
        f"Average knee angle {mean_knee:.0f} deg, trunk lean {mean_lean:.0f} deg."
    )

    meta = dict(meta)
    meta.update({"engine": "pose", "frames_analysed": n,
                 "duration_sec": info.get("duration"), "fps": round(info["fps"], 1) if info.get("fps") else None})
    return {
        "overall_score": overall,
        "grade": _label(overall),
        "summary": summary,
        "metrics": M,
        "strengths": strengths,
        "improvements": improvements,
        "drills": SPORT_DRILLS[sport],
        "quality": {"detection_rate": _clamp(detect_rate * 100), "sharpness": sharp_pct, "tips": quality_tips},
        "timeline": timeline,
        "meta": meta,
        "flat_feedback": [*improvements, *strengths],
        "measurements": {
            "avg_knee_angle_deg": round(mean_knee, 1),
            "avg_trunk_lean_deg": round(mean_lean, 1),
            "avg_stance_width_ratio": round(mean_width, 2),
            "avg_balance_offset": round(mean_off, 3),
        },
    }
