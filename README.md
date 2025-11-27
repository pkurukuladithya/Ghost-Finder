# Ghost Finder — Real-Time Presence Counter 👻📹

Ghost Finder is a Flask + YOLO web app that watches a live camera stream and tells you how many people are currently visible in the frame. It stores snapshots of every change to the headcount so you can review when the crowd size went up or down. The UI is built with Tailwind (CDN) for a clean, modern look.

Owner: Praveena Kurukuladithya — Computer Engineering Student

## Features ✨
- 📺 Live video with on-frame bounding boxes from YOLOv8.
- 🧍 Presence-only counting: shows “people in view” and logs each change with timestamp.
- 🗂️ History page lists the latest 200 presence changes.
- 🔄 Auto-updating stats card (AJAX) so the dashboard stays in sync.
- ⚙️ Configurable thresholds, model path, camera index, resolution, and frame skipping to balance accuracy vs. speed.

## Tech Stack 🛠️
- 🐍 Python, Flask
- 🤖 YOLOv8 (ultralytics) + OpenCV
- 🗃️ SQLite via SQLAlchemy
- 🎨 Tailwind CSS (CDN) + Manrope font for the frontend

## Project Structure 📁
- `app.py` — Flask app, routes (`/`, `/video_feed`, `/history`, `/api/stats`).
- `counter.py` — YOLO inference, lightweight centroid tracker, presence counting, DB logging.
- `config.py` — Configuration and environment variable defaults.
- `models.py` — SQLAlchemy models (`CountEvent`).
- `templates/` — Jinja templates (`base.html`, `index.html`, `history.html`).
- `static/` — Static assets (you can add custom CSS if needed).
- `requirements.txt` — Python dependencies.
- `yolov8n.pt` — Default YOLOv8 model weights (you can swap in another checkpoint).
- `people_counter.db` — SQLite database (auto-created on first run).

## Prerequisites ✅
- Python 3.9+ recommended.
- A webcam (or USB/IP camera exposed as a local index) accessible from the machine.
- Optional GPU (CUDA) for faster inference; falls back to CPU automatically.

## Setup 🚀
```bash
# 1) Create a virtual environment
python -m venv .venv

# 2) Activate it (Windows)
.\.venv\Scripts\activate
# (macOS/Linux) source .venv/bin/activate

# 3) Install dependencies
pip install -r requirements.txt
```

## Configuration ⚙️
All settings have sane defaults but can be overridden via environment variables:
- `CAMERA_INDEX` — Which camera to use (0 = default).
- `YOLO_MODEL_PATH` — Path to weights (e.g., `yolov8n.pt`, `yolov8s.pt`, or your trained `best.pt`).
- `CONF_THRESHOLD` — Detection confidence (default 0.5). Lower to ~0.35 if people are missed; raise to reduce false positives.
- `IOU_THRESHOLD` — NMS IoU (default 0.45).
- `LINE_POSITION` — Visual guide line (0.0 left … 1.0 right). Purely cosmetic now.
- `FRAME_WIDTH` / `FRAME_HEIGHT` — Resize for performance (default 960x540).
- `SKIP_FRAMES` — Process every Nth frame (1 = every frame; increase to reduce lag).

Example (Windows PowerShell):
```powershell
$env:CONF_THRESHOLD="0.4"; $env:FRAME_WIDTH="1280"; python app.py
```

## Running the App ▶️
```bash
python app.py
# Open http://localhost:5000/
```

## Using the UI 🖥️
- **Live dashboard (`/`)**: Shows the camera feed with boxes and the live “people in view” count. The stat card updates automatically every few seconds.
- **History (`/history`)**: Lists the most recent 200 count changes with timestamps and the count at that moment.

## Accuracy & Performance Tips 🎯
- Use good, even lighting; avoid backlight and motion blur.
- Frame people fully (head to torso) and keep the camera steady.
- If detections miss: lower `CONF_THRESHOLD` (0.35–0.45) or switch to a stronger model (`yolov8s.pt`).
- If video lags: lower `FRAME_WIDTH/HEIGHT` or raise `SKIP_FRAMES` (e.g., 2 or 3).
- For busy scenes, a GPU model (`yolov8s.pt` or larger) will help, but may need more VRAM.

## Training Your Own Model (Optional) 🧠
1) Collect and label images of your specific entrance/scene in YOLO format.  
2) Create `data.yaml` pointing to train/val image folders.  
3) Train: `yolo detect train model=yolov8n.pt data=data.yaml imgsz=640 epochs=50`.  
4) Point `YOLO_MODEL_PATH` to the resulting `best.pt`.

## Database 🗄️
- SQLite database lives at `people_counter.db` by default.
- `CountEvent` rows are created whenever the visible headcount changes; `lobby_count` stores the “people in view” at that moment.
- To reset data, stop the app and delete `people_counter.db` (or point `SQLALCHEMY_DATABASE_URI` elsewhere).

## Notes 📌
- The app uses a simple centroid tracker to keep IDs stable for short gaps. For very crowded scenes, consider higher-resolution input and a stronger model.
- Tailwind is loaded via CDN; no build step required.

## Contact ✉️
For questions or improvements, reach out to the project owner:
- Praveena Kurukuladithya — Computer Engineering Student
