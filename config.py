import os

BASE_DIR = os.path.abspath(os.path.dirname(__file__))

class Config:
    SECRET_KEY = "change-this-secret-key"
    SQLALCHEMY_DATABASE_URI = "sqlite:///" + os.path.join(BASE_DIR, "people_counter.db")
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Camera / detection tuning
    CAMERA_INDEX = int(os.getenv("CAMERA_INDEX", 0))  # 0 = default webcam
    YOLO_MODEL_PATH = os.getenv("YOLO_MODEL_PATH", "yolov8n.pt")
    CONF_THRESHOLD = float(os.getenv("CONF_THRESHOLD", 0.5))
    IOU_THRESHOLD = float(os.getenv("IOU_THRESHOLD", 0.45))
    LINE_POSITION = float(os.getenv("LINE_POSITION", 0.5))  # percent of frame width
    FRAME_WIDTH = int(os.getenv("FRAME_WIDTH", 960))        # lower for less lag
    FRAME_HEIGHT = int(os.getenv("FRAME_HEIGHT", 540))
    SKIP_FRAMES = int(os.getenv("SKIP_FRAMES", 1))          # process every Nth frame (1 = every frame)
