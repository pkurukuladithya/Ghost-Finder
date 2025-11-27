import cv2
import math
import torch
from ultralytics import YOLO

from models import db, CountEvent

class CentroidTracker:
    """
    Lightweight centroid tracker. Keeps IDs alive for a few frames to reduce flicker
    and re-assigns based on nearest neighbor.
    """
    def __init__(self, max_distance=50, max_disappeared=10):
        self.next_id = 0
        self.objects = {}        # id -> (cx, cy)
        self.disappeared = {}    # id -> frames since last seen
        self.max_distance = max_distance
        self.max_disappeared = max_disappeared

    def update(self, detections):
        """
        Args:
            detections: list of (cx, cy)
        Returns:
            objects dict and list mapping each detection index -> object id
        """
        assignments = [-1] * len(detections)

        # If no objects currently tracked, register all detections
        if len(self.objects) == 0:
            for i, (cx, cy) in enumerate(detections):
                self.objects[self.next_id] = (cx, cy)
                self.disappeared[self.next_id] = 0
                assignments[i] = self.next_id
                self.next_id += 1
            return self.objects, assignments

        new_objects = {}
        used_ids = set()

        # Greedy match detections to existing objects
        for i, (cx, cy) in enumerate(detections):
            best_id = None
            best_dist = float("inf")
            for obj_id, (ox, oy) in self.objects.items():
                if obj_id in used_ids:
                    continue
                dist = math.hypot(cx - ox, cy - oy)
                if dist < best_dist and dist < self.max_distance:
                    best_dist = dist
                    best_id = obj_id

            if best_id is None:
                # new object
                new_objects[self.next_id] = (cx, cy)
                self.disappeared[self.next_id] = 0
                assignments[i] = self.next_id
                used_ids.add(self.next_id)
                self.next_id += 1
            else:
                new_objects[best_id] = (cx, cy)
                self.disappeared[best_id] = 0
                assignments[i] = best_id
                used_ids.add(best_id)

        # Handle disappeared objects
        for obj_id in self.objects.keys():
            if obj_id in new_objects:
                continue
            self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
            if self.disappeared[obj_id] <= self.max_disappeared:
                new_objects[obj_id] = self.objects[obj_id]

        self.objects = new_objects
        return self.objects, assignments

class PeopleCounter:
    def __init__(
        self,
        camera_index=0,
        model_path="yolov8n.pt",
        conf_threshold=0.5,
        iou_threshold=0.45,
        line_position=0.5,
        frame_width=960,
        frame_height=540,
        skip_frames=1,
    ):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = YOLO(model_path)

        self.conf_threshold = conf_threshold
        self.iou_threshold = iou_threshold
        self.line_position = line_position
        self.frame_width = frame_width
        self.frame_height = frame_height
        self.skip_frames = max(1, int(skip_frames))

        self.cap = cv2.VideoCapture(camera_index)
        if not self.cap.isOpened():
            raise RuntimeError("Cannot open camera")
        # Reduce lag and load
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.cap.set(cv2.CAP_PROP_FPS, 30)

        self.tracker = CentroidTracker(max_distance=70, max_disappeared=12)
        self.lobby_count = 0
        self.last_seen = {}  # id -> frame index
        self.frame_index = 0
        self.line_margin_px = 12  # keep for visualization

    def _log_event(self):
        """Save lobby presence snapshot to database when count changes."""
        event = CountEvent(direction="PRESENT", lobby_count=self.lobby_count)
        db.session.add(event)
        db.session.commit()

    def _cleanup_stale_ids(self, active_ids):
        """Drop old memory for objects that disappeared long time ago."""
        stale = []
        for obj_id in list(self.last_seen.keys()):
            if obj_id not in active_ids:
                last_seen_frame = self.last_seen.get(obj_id, -1)
                if self.frame_index - last_seen_frame > 90:
                    stale.append(obj_id)
        for obj_id in stale:
            self.last_seen.pop(obj_id, None)

    def generate_frames(self):
        """Generator that yields JPEG frames with boxes and counters drawn."""
        while True:
            success, frame = self.cap.read()
            if not success:
                break

            self.frame_index += 1
            h, w = frame.shape[:2]
            line_x = int(w * self.line_position)

            # Optionally drop frames to reduce load
            if self.frame_index % self.skip_frames != 0:
                continue

            # YOLO person detection
            results = self.model.predict(
                frame,
                classes=[0],  # person class
                device=self.device,
                conf=self.conf_threshold,
                iou=self.iou_threshold,
                imgsz=640,
                verbose=False,
            )

            detections = []  # centroids
            boxes_list = []

            for r in results:
                for box in r.boxes:
                    conf = float(box.conf[0])
                    if conf < self.conf_threshold:
                        continue
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    cx = int((x1 + x2) / 2)
                    cy = int((y1 + y2) / 2)
                    detections.append((cx, cy))
                    boxes_list.append((x1, y1, x2, y2, cx, cy, conf))

            objects, assignments = self.tracker.update(detections)
            active_ids = set(objects.keys())
            self._cleanup_stale_ids(active_ids)

            # Update lobby count based on active IDs (people currently visible)
            current_lobby = len(active_ids)
            if current_lobby != self.lobby_count:
                self.lobby_count = current_lobby
                self._log_event()

            # Draw vertical line (visual guide only)
            cv2.line(frame, (line_x, 0), (line_x, h), (90, 200, 255), 2)

            # Go through detections with assigned IDs
            for (x1, y1, x2, y2, cx, cy, conf), obj_id in zip(boxes_list, assignments):
                if obj_id == -1:
                    continue

                self.last_seen[obj_id] = self.frame_index

                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 0), 2)
                cv2.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
                label = f"ID {obj_id} {conf:.2f}"
                cv2.putText(frame, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 200, 0), 2)

            # Draw counts overlay
            cv2.putText(frame, f"In Lobby: {self.lobby_count}", (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 200, 0), 2)

            # Encode frame as JPEG
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()

            # Yield as HTTP multipart
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
