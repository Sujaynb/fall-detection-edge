# YOLO_Video.py
from ultralytics import YOLO
import cv2
import math
import os
import csv
from datetime import datetime
import threading
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("YOLO_Video")

# CONFIG
MODEL_PATH = os.environ.get("FALL_MODEL_PATH",
                            "runs/detect/FallDetector_PhoneOptimized/weights/best.pt")
CONF_THRESHOLD = float(os.environ.get("FALL_CONF_THRESHOLD", 0.60))
FALL_FOLDER = os.environ.get("FALL_FOLDER", "static/falls")
LOG_FILE = os.environ.get("FALL_LOG_FILE", "falls_log.csv")

os.makedirs(FALL_FOLDER, exist_ok=True)

# GLOBALS used by Flask app
FALL_ALERT_FLAG = threading.Event()   # set() when fall occurs
LAST_FALL_META = {}                    # dict with last fall's metadata (timestamp, ip, conf, path)

# Ensure CSV header
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "ip", "location", "confidence", "image_path"])

# Load model ONCE (thread-safe)
try:
    model = YOLO(MODEL_PATH)
    logger.info("Loaded YOLO model from %s", MODEL_PATH)
except Exception as e:
    model = None
    logger.exception("Failed to load YOLO model: %s", e)


def log_fall(ip_address, location, confidence, image_path):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([timestamp, ip_address, location, f"{confidence:.3f}", image_path])
    logger.info("Logged fall: %s %s %s", timestamp, ip_address, image_path)


def save_fall_frame(frame, prefix="FALL"):
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    filename = f"{prefix}_{ts}.jpg"
    path = os.path.join(FALL_FOLDER, filename)
    # save compressed jpg
    cv2.imwrite(path, frame, [int(cv2.IMWRITE_JPEG_QUALITY), 85])
    return path


def video_detection(source, client_ip="Unknown"):
    """
    Generator that yields annotated frames.
    source: video file path or camera url (IP Webcam like http://192.168.x.x:8080/video or 0)
    client_ip: string containing ip to log
    """
    global FALL_ALERT_FLAG, LAST_FALL_META, model

    if model is None:
        # yield a black frame with error text if model failed to load
        black = 255 * (cv2.imread("placeholder.png") is None)  # no-op to avoid linter
        while True:
            frame = 255 * (cv2.imread("placeholder.png") is None)  # keep generator semantics
            yield frame

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        logger.error("Cannot open video source: %s", source)
        # produce nothing
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # if streaming and frame fails, keep looping to try again
                logger.debug("Empty frame from source: %s", source)
                break

            # Inference (Ultralytics returns results[0])
            try:
                results = model(frame)[0]
            except Exception as e:
                logger.exception("Model inference failed: %s", e)
                yield frame
                continue

            fall_detected = False
            fall_confidence = 0.0

            # draw boxes
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                conf = float(box.conf[0])
                cls = int(box.cls[0])
                # NOTE: your datayaml order matters: assume 0=fall, 1=nonfall
                class_name = "fall" if cls == 0 else "nonfall"
                color = (0, 0, 255) if class_name == "fall" else (0, 255, 0)
                label = f"{class_name} {conf:.2f}"

                if conf >= CONF_THRESHOLD:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    cv2.putText(frame, label, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

                if class_name == "fall" and conf >= CONF_THRESHOLD:
                    fall_detected = True
                    fall_confidence = max(fall_confidence, conf)

            if fall_detected:
                # save frame and log (do this once per detection)
                image_path = save_fall_frame(frame)
                location = "Unknown"  # stub — can be replaced by GeoIP service later
                log_fall(client_ip, location, fall_confidence, image_path)

                # set global meta for flask to include in alert
                LAST_FALL_META = {
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "ip": client_ip,
                    "location": location,
                    "confidence": fall_confidence,
                    "image_path": image_path
                }

                # fire alert event (Flask consumes this)
                FALL_ALERT_FLAG.set()

                # overlay a visible message on the frame
                overlay_text = f"FALL DETECTED {LAST_FALL_META['timestamp']}"
                cv2.putText(frame, overlay_text, (10, 40),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)

            yield frame

    finally:
        cap.release()
