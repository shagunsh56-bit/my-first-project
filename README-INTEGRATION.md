# SportsSight AI — Frontend + FastAPI integration

## What is connected
- The React/Vite assessment screen now lets the athlete select a photo/video from the gallery or open a camera-capable file picker.
- On **Analyse my performance**, the frontend sends the selected file as `multipart/form-data` field `file` to `POST /analyze-file`.
- The returned score/status/feedback is displayed on the report screen instead of showing the sample report after upload.
- The FastAPI backend enables CORS for local prototype development and removes temporary uploads even if analysis fails.

## Run locally (Windows PowerShell)

### 1. Backend
Open a terminal in this project folder:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
uvicorn main:app --reload --host 127.0.0.1 --port 8000
```

Check `http://127.0.0.1:8000/` — it should return `SportsSight AI Engine is Running!`.

### 2. Frontend
Open a second terminal in the same folder:

```powershell
npm install
npm run dev
```

Open the Vite URL shown in the terminal. The frontend defaults to `http://127.0.0.1:8000` for the backend. If the backend runs on another host/port, create a `.env` file using `.env.example` and set `VITE_API_BASE_URL` to that API origin, then restart Vite.

## API contract
`POST /analyze-file` expects a multipart file field named `file`. Supported extensions: JPG, JPEG, PNG, MP4, MOV, AVI, MKV, WEBM. The response contains `status`, `talent_score`, and feedback fields.

## Important prototype limitations
- The current OpenCV logic is a basic human detector plus image-edge heuristics. It is **not** a validated sports pose-estimation model, and its score should not be used for athlete selection or as a scientifically valid performance rating.
- The backend does not currently use the selected sport to change its algorithm, and the frontend's detailed sample metric cards are hidden when a real backend response is shown.
- For production, restrict CORS to your deployed frontend origin, add upload-size limits/authentication, and use a sport-specific pose model (e.g. keypoint detection + temporal action analysis) validated with coaches and athletes.

## Update: in-app camera + full feedback report
- **Record now** now opens an in-app camera (getUserMedia + MediaRecorder): live preview, start/stop, timer (max 3 min), front/back camera flip, retake / use video. The recorded clip is sent to `/analyze-file` like an uploaded file.
- Camera needs **HTTPS or localhost** and the user must allow camera permission. On a phone testing over LAN IP (http://192.168.x.x) the browser will block the camera - use HTTPS (e.g. `vite --host` with a tunnel such as ngrok/cloudflared) or deploy.
- `/analyze-file` now also accepts the form field `sport` and returns a `report` object: overall score + grade, summary, skill metrics, strengths, focus areas, sport-specific drills, recording-quality stats, movement timeline. The Report screen renders all of it and the top-right button downloads it as a .txt.
- Scores are still computed from OpenCV visual heuristics (not true pose estimation) - use as a training aid only.

## Update: MediaPipe pose estimation (`pose_engine.py`)
- `/analyze-file` first runs **MediaPipe PoseLandmarker** (33 body landmarks per frame) and computes real, sport-aware metrics: knee angle, trunk lean, stance width, balance (hips vs feet), movement speed, direction changes, burst recovery time, posture consistency. Feedback text quotes the measured values.
- If MediaPipe / the model is unavailable, or the full body is not tracked, it **falls back** to the older OpenCV heuristics. The report shows which engine was used.
- The model file (~6-30 MB) downloads automatically on first use into `./models/`. For offline/production, download `pose_landmarker_full.task` (or `_lite` / `_heavy`) yourself and set `POSE_MODEL_PATH`, or set `POSE_MODEL_VARIANT=lite` for faster CPU inference.
- Check status any time: `GET http://127.0.0.1:8000/engine-status`.
- Needs Python 3.9-3.12 for MediaPipe wheels. `pip install -r requirements.txt` installs it.
- Limitations: metrics assume a reasonably fixed camera with the full body in frame. A panning camera under-estimates speed (speed is measured from hip movement in the image). Target angle/lean ranges per sport (`SPORT_TARGETS`) are sensible starting values and should be tuned with coaches.


## Firebase Firestore database integration

The frontend now saves assessment metadata and the AI report to Cloud Firestore after the FastAPI analysis completes. It stores the sport, filename, status, score, feedback, report metrics, anonymous Firebase user ID, and server timestamp. It does **not** upload/store the raw video in Firestore; the backend's temporary upload is deleted after processing.

### One-time Firebase setup

1. Open https://console.firebase.google.com/ and create/select a Firebase project.
2. In **Build → Firestore Database**, create a database.
3. In **Project settings → General → Your apps**, add a **Web app** and copy its config.
4. In **Build → Authentication → Sign-in method**, enable **Anonymous** sign-in.
5. Copy `.env.example` to a new file named `.env` in this project root and fill in the six `VITE_FIREBASE_*` values from the Firebase web app config. Keep `.env` private and do not commit it.
6. In Firebase Console → Firestore Database → **Rules**, publish the contents of `firestore.rules`.
7. Install dependencies and restart Vite:

```powershell
npm install
npm run dev
```

The backend run commands remain unchanged. Firestore is optional until those environment values are filled in; without them, AI analysis still works but is not saved to Firebase.

### Database shape

Collection: `assessments`

Each document contains `userId`, `sport`, `filename`, `status`, `talent_score`, `assessment`, `report`, feedback fields, and `createdAt`. Anonymous authentication gives each browser session/user an identity so Firestore rules can scope documents to their owner. Clearing browser site data or changing browser/device can create a different anonymous identity.

### Security notes

- Never put Firebase Admin/service-account credentials in the frontend. This integration uses the Firebase Web SDK.
- The provided rules restrict documents to their owner. Test rules in Firebase's Rules Playground before production.
- For a real deployment, add an explicit account/recovery flow, validate payloads, configure authorized domains and backend CORS, and consider Firebase Storage for videos if you need to retain them. Do not store video bytes in Firestore documents.
