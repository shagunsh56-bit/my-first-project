from fastapi import FastAPI, File, Form, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.concurrency import run_in_threadpool
from pathlib import Path
from datetime import datetime, timezone
import cv2
import os
import shutil
import tempfile
import numpy as np

import pose_engine

app = FastAPI(title="SportsSight AI - Fair Athlete Assessment")
# For local development. In production, replace '*' with your deployed frontend origin.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

hog = cv2.HOGDescriptor()
hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_FRAMES_READ = 600      # hard cap on frames read from a video
SAMPLE_EVERY = 5           # analyse every Nth frame

# Sport-specific copy: what is assessed + drills to improve
SPORT_PROFILES = {
    "Kabaddi": {
        "focus": "raid movement, agility, balance and recovery",
        "drills": [
            "Cone-weave raids: 5 sets x 6 cones, focus on low centre of gravity.",
            "Resisted retreats: 5 x 10 m, twice a week, to improve post-raid recovery.",
            "Single-leg balance holds: 3 x 30 s per leg to improve stability on turns.",
        ],
    },
    "Wrestling": {
        "focus": "stance, balance, attack speed and technique",
        "drills": [
            "Stance-and-motion drill: 4 x 2 min keeping knees bent and hips low.",
            "Level-change shots: 5 x 8 reps, focus on fast, balanced entries.",
            "Core stability circuit: planks and Russian twists, 3 rounds.",
        ],
    },
    "Sprint": {
        "focus": "start reaction, acceleration, stride mechanics and speed",
        "drills": [
            "Block starts: 6 x 20 m, focus on explosive first 3 steps.",
            "A-skips and B-skips: 3 x 20 m for stride mechanics.",
            "Acceleration runs: 5 x 30 m with full recovery between reps.",
        ],
    },
}
DEFAULT_PROFILE = SPORT_PROFILES["Kabaddi"]


@app.get("/")
def home():
    return {"message": "SportsSight AI Engine is Running!"}


@app.get("/engine-status")
async def engine_status():
    """Tells you whether the MediaPipe pose model is loaded (it downloads on first use)."""
    def check():
        lm = pose_engine.load_landmarker(video=False)
        if lm is None:
            return {"pose_engine": "unavailable", "reason": pose_engine.LOAD_ERROR, "fallback": "OpenCV heuristics"}
        lm.close()
        return {"pose_engine": "ready", "model_variant": pose_engine.VARIANT}
    return await run_in_threadpool(check)


def clamp(v, lo=0, hi=100):
    return int(max(lo, min(hi, round(v))))


def label_for(score):
    if score >= 85: return "Excellent"
    if score >= 75: return "Strong"
    if score >= 62: return "Good"
    if score >= 50: return "Developing"
    return "Needs work"


def detect_human(gray):
    boxes, _ = hog.detectMultiScale(gray, winStride=(8, 8), padding=(8, 8), scale=1.05)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5) if not face_cascade.empty() else ()
    return len(boxes) > 0 or len(faces) > 0, boxes


def frame_metrics(frame):
    """Per-frame visual measurements (heuristic, not true pose estimation)."""
    h0, w0 = frame.shape[:2]
    if max(h0, w0) > 640:  # speed: downscale big frames
        s = 640 / max(h0, w0)
        frame = cv2.resize(frame, (int(w0 * s), int(h0 * s)))
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    has_human, boxes = detect_human(gray)
    edges = cv2.Canny(gray, 50, 150)
    if not has_human and np.mean(cv2.Canny(gray, 100, 200)) < 2.0:
        return None
    h, w = gray.shape
    left, right = np.sum(edges[:, : w // 2]), np.sum(edges[:, w // 2:])
    top, bottom = np.sum(edges[: h // 2, :]), np.sum(edges[h // 2:, :])
    symmetry = min(left, right) / (max(left, right) + 1e-5)
    lower_ratio = bottom / (top + 1e-5)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()
    framing = 0.0
    if len(boxes) > 0:
        x, y, bw, bh = max(boxes, key=lambda b: b[2] * b[3])
        framing = min(1.0, (bh / h))  # how much of frame height the athlete fills
    return {
        "gray_small": cv2.resize(gray, (160, 90)),
        "symmetry": float(symmetry),
        "lower_ratio": float(lower_ratio),
        "sharpness": float(sharpness),
        "human": bool(has_human),
        "framing": float(framing),
    }


def build_report(samples, motion_levels, meta, sport):
    n = len(samples)
    detect_rate = sum(1 for s in samples if s["human"]) / n
    sym = float(np.mean([s["symmetry"] for s in samples]))
    low = float(np.mean([s["lower_ratio"] for s in samples]))
    sharp = float(np.mean([s["sharpness"] for s in samples]))
    framing = float(np.mean([s["framing"] for s in samples]))

    has_motion = len(motion_levels) > 0
    motion_avg = float(np.mean(motion_levels)) if has_motion else 0.0
    motion_std = float(np.std(motion_levels)) if has_motion else 0.0

    balance = clamp(35 + sym * 65)
    stance = clamp(40 + min(low, 1.4) / 1.4 * 55)
    video_quality = clamp(min(sharp, 300) / 300 * 70 + 30 * (0.5 + framing / 2 if framing else 0.5))
    if has_motion:
        agility = clamp(30 + min(motion_avg, 25) / 25 * 65)
        consistency = clamp(95 - min(motion_std / (motion_avg + 1e-5), 1.5) / 1.5 * 55)
        recovery = clamp(45 + consistency * 0.45)
    else:  # single image
        agility = consistency = recovery = None

    metrics = [
        {"label": "Balance & symmetry", "score": balance, "note": label_for(balance)},
        {"label": "Stance & lower-body", "score": stance, "note": label_for(stance)},
    ]
    if has_motion:
        metrics = [
            {"label": "Agility & movement", "score": agility, "note": label_for(agility)},
            *metrics,
            {"label": "Consistency", "score": consistency, "note": label_for(consistency)},
            {"label": "Recovery rhythm", "score": recovery, "note": label_for(recovery)},
        ]
    metrics.append({"label": "Video quality", "score": video_quality, "note": label_for(video_quality)})

    core = [m["score"] for m in metrics if m["label"] != "Video quality"]
    overall = clamp(np.mean(core) * (0.85 + 0.15 * detect_rate))
    overall = int(min(max(overall, 40), 96))

    ranked = sorted([m for m in metrics if m["label"] != "Video quality"], key=lambda m: m["score"])
    weakest, strongest = ranked[0], ranked[-1]

    strengths, improvements = [], []
    if strongest["score"] >= 62:
        strengths.append(f"{strongest['label']} is your best area ({strongest['score']}/100). Keep building on it.")
    if sym >= 0.65:
        strengths.append("Left-right visual balance looks even across the recording.")
    if detect_rate >= 0.7:
        strengths.append("Athlete was clearly visible in most analysed frames.")
    if has_motion and motion_std / (motion_avg + 1e-5) < 0.7:
        strengths.append("Movement intensity stays steady through the clip.")
    if not strengths:
        strengths.append("You submitted a usable recording - a good baseline to improve from.")

    if weakest["score"] < 75:
        improvements.append(f"Focus area: {weakest['label']} ({weakest['score']}/100) needs the most work.")
    if sym < 0.65:
        improvements.append("Possible uneven balance: review lateral stability and weight shift.")
    if low < 0.8:
        improvements.append("Review stance width and lower-body positioning.")
    if sharp < 40:
        improvements.append("Video looks blurry or low-light. Use better lighting and a steady camera.")
    if detect_rate < 0.5:
        improvements.append("Athlete was not clearly visible in many frames. Keep the full body in frame.")
    if not improvements:
        improvements.append("No major issues detected. Confirm technique details with a coach.")

    quality_tips = []
    if sharp < 40: quality_tips.append("Increase lighting / clean the camera lens.")
    if framing and framing < 0.4: quality_tips.append("Move the camera back or zoom so the athlete fills more of the frame.")
    if detect_rate < 0.5: quality_tips.append("Record from the side with the full body visible.")
    if not quality_tips: quality_tips.append("Recording quality is good for analysis.")

    profile = SPORT_PROFILES.get(sport, DEFAULT_PROFILE)
    summary = (
        f"Your {sport} {'video' if has_motion else 'photo'} scored {overall}/100 overall. "
        f"Strongest area: {strongest['label']} ({strongest['score']}). "
        f"Biggest opportunity: {weakest['label']} ({weakest['score']}). "
        f"This assessment looks at {profile['focus']}."
    )

    # timeline of activity (video only)
    timeline = []
    if has_motion:
        peak = max(motion_levels) or 1.0
        timeline = [clamp(m / peak * 100) for m in motion_levels][:40]

    flat_feedback = [*improvements, *strengths]
    return {
        "overall_score": overall,
        "grade": label_for(overall),
        "summary": summary,
        "metrics": metrics,
        "strengths": strengths,
        "improvements": improvements,
        "drills": profile["drills"],
        "quality": {
            "detection_rate": clamp(detect_rate * 100),
            "sharpness": clamp(min(sharp, 300) / 300 * 100),
            "tips": quality_tips,
        },
        "timeline": timeline,
        "meta": meta,
        "flat_feedback": flat_feedback,
    }


@app.post("/analyze-file")
async def analyze_file(file: UploadFile = File(...), sport: str = Form("Kabaddi")):
    filename = Path(file.filename or "upload").name
    ext = Path(filename).suffix.lower()
    # Browser MediaRecorder may upload "blob"/no extension; infer from content type.
    if not ext and file.content_type:
        ext = {"video/webm": ".webm", "video/mp4": ".mp4", "image/jpeg": ".jpg", "image/png": ".png"}.get(file.content_type.split(";")[0], "")
    if ext not in ALLOWED_EXT:
        raise HTTPException(status_code=415, detail="Unsupported file type. Upload JPG, PNG, MP4, MOV, AVI, MKV, or WEBM.")

    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as temp:
            temp_path = temp.name
            shutil.copyfileobj(file.file, temp)

        samples, motion_levels = [], []
        meta = {
            "filename": filename,
            "sport": sport,
            "analysed_at": datetime.now(timezone.utc).isoformat(),
            "type": "image" if ext in {".jpg", ".jpeg", ".png"} else "video",
        }

        # ---- 1) Preferred path: real pose estimation (MediaPipe) -------------------------
        pose_note = None
        try:
            extractor = pose_engine.extract_image if meta["type"] == "image" else pose_engine.extract_video
            pose_frames, pose_info = await run_in_threadpool(extractor, temp_path)
            if pose_frames is None:
                pose_note = pose_engine.LOAD_ERROR or "Pose engine unavailable"
            else:
                pose_report = pose_engine.compute_pose_report(pose_frames, pose_info, sport, meta)
                if pose_report is not None:
                    return {
                        "status": "success",
                        "filename": filename,
                        "talent_score": pose_report["overall_score"],
                        "assessment": f"Pose-tracked {pose_report['meta']['frames_analysed']} frame(s)",
                        "actionable_feedback": pose_report["flat_feedback"],
                        "report": pose_report,
                    }
                pose_note = "Full body was not clearly tracked; used basic visual analysis instead."
        except Exception as e:  # never let the pose path break the whole request
            pose_note = f"Pose engine error: {e}"

        # ---- 2) Fallback: basic OpenCV heuristics ----------------------------------------
        meta["engine"] = "heuristic"
        meta["engine_note"] = pose_note

        if meta["type"] == "image":
            img = cv2.imread(temp_path)
            if img is None:
                raise HTTPException(status_code=400, detail="Invalid image file format.")
            m = frame_metrics(img)
            if m:
                samples.append(m)
        else:
            cap = cv2.VideoCapture(temp_path)
            fps = cap.get(cv2.CAP_PROP_FPS) or 0
            total = cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0
            meta["fps"] = round(fps, 1) if fps and fps < 240 else None
            meta["duration_sec"] = round(total / fps, 1) if fps and 0 < total < 1e8 else None
            prev, idx = None, 0
            while cap.isOpened() and idx < MAX_FRAMES_READ:
                ret, frame = cap.read()
                if not ret:
                    break
                idx += 1
                if idx % SAMPLE_EVERY == 0:
                    m = frame_metrics(frame)
                    if m:
                        samples.append(m)
                        if prev is not None:
                            motion_levels.append(float(np.mean(cv2.absdiff(prev, m["gray_small"]))))
                        prev = m["gray_small"]
            cap.release()
            meta["frames_analysed"] = len(samples)

        for s in samples:
            s.pop("gray_small", None)

        if not samples or not any(s["human"] for s in samples):
            return {
                "status": "rejected",
                "filename": filename,
                "talent_score": 0,
                "error": "No Athlete Detected",
                "assessment": "Invalid Input",
                "feedback": ["No athlete detected. Record a clear, well-lit performance with the full body visible."],
            }

        report = build_report(samples, motion_levels, meta, sport)
        return {
            "status": "success",
            "filename": filename,
            "talent_score": report["overall_score"],
            "assessment": f"Analysed {len(samples)} sampled frame(s)",
            "actionable_feedback": report["flat_feedback"],
            "report": report,
        }
    finally:
        await file.close()
        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)
