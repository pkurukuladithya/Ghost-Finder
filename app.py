from flask import Flask, render_template, Response, jsonify, stream_with_context
from config import Config
from models import db, CountEvent
from counter import PeopleCounter

app = Flask(__name__)
app.config.from_object(Config)
db.init_app(app)

# Initialize database and YOLO counter
with app.app_context():
    db.create_all()
    people_counter = PeopleCounter(
        camera_index=Config.CAMERA_INDEX,
        model_path=Config.YOLO_MODEL_PATH,
        conf_threshold=Config.CONF_THRESHOLD,
        iou_threshold=Config.IOU_THRESHOLD,
        line_position=Config.LINE_POSITION,
        frame_width=Config.FRAME_WIDTH,
        frame_height=Config.FRAME_HEIGHT,
        skip_frames=Config.SKIP_FRAMES,
    )  # default webcam


def current_stats():
    """Fetch aggregated stats from DB and keep counter object in sync."""
    last_event = CountEvent.query.order_by(CountEvent.timestamp.desc()).first()
    lobby_count = last_event.lobby_count if last_event else 0

    people_counter.lobby_count = lobby_count

    return lobby_count


@app.route("/")
def index():
    lobby_count = current_stats()
    return render_template(
        "index.html",
        lobby_count=lobby_count
    )


@app.route("/video_feed")
def video_feed():
    return Response(
        stream_with_context(people_counter.generate_frames()),
        mimetype="multipart/x-mixed-replace; boundary=frame"
    )


@app.route("/history")
def history():
    events = CountEvent.query.order_by(CountEvent.timestamp.desc()).limit(200).all()
    return render_template("history.html", events=events)


@app.route("/api/stats")
def stats():
    lobby_count = current_stats()
    return jsonify(
        lobby_count=lobby_count,
    )


if __name__ == "__main__":
    app.run(debug=True)
