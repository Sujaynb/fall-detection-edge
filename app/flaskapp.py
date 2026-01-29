import os
import time
import csv
import logging
import cv2
import threading
import smtplib
from email.mime.text import MIMEText
from flask import Flask, render_template, Response, request, session, redirect, url_for, send_from_directory, jsonify
from werkzeug.utils import secure_filename
from YOLO_Video import video_detection, FALL_ALERT_FLAG, LAST_FALL_META

                                                               
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("flaskapp")

                                                               
SENDER_EMAIL = "sujaynb07@gmail.com"
APP_PASSWORD = "gqqd kosr xwot mzbf"
RECEIVER_EMAIL = SENDER_EMAIL
ALERT_COOLDOWN = 60           

UPLOAD_FOLDER = "static/files"
FALL_FOLDER = "static/falls"
LOG_FILE = "falls_log.csv"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(FALL_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["SECRET_KEY"] = "super-secret-key"
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

_last_alert_time = 0
_last_alert_lock = threading.Lock()

                                                                

def send_email_alert(meta):
    global _last_alert_time

    with _last_alert_lock:
        now = time.time()
        if now - _last_alert_time < ALERT_COOLDOWN:
            logger.info("Cooldown active, skipping alert.")
            return
        _last_alert_time = now

    body = f"""
FALL DETECTED!

Time: {meta.get('timestamp')}
Source IP: {meta.get('ip')}
Location: {meta.get('location')}
Confidence: {meta.get('confidence')}
Saved Image: {meta.get('image_path')}
"""

    msg = MIMEText(body)
    msg["Subject"] = "Fall Alert"
    msg["From"] = SENDER_EMAIL
    msg["To"] = RECEIVER_EMAIL

    try:
        with smtplib.SMTP_SSL("smtp.gmail.com", 465) as s:
            s.login(SENDER_EMAIL, APP_PASSWORD)
            s.send_message(msg)
        logger.info("EMAIL SENT SUCCESSFULLY")
    except Exception as e:
        logger.error("EMAIL ERROR: %s", e)

                                                                

def gen_stream(source, ip):
    for frame in video_detection(source, ip):

        if FALL_ALERT_FLAG.is_set():

                                                                 
            retries = 0
            while (
                LAST_FALL_META.get("timestamp") is None or
                LAST_FALL_META.get("image_path") is None or
                LAST_FALL_META.get("confidence") is None
            ) and retries < 5:
                time.sleep(0.05)             
                retries += 1

                                    
            meta = {
                "timestamp": LAST_FALL_META.get("timestamp", "UNKNOWN"),
                "ip": LAST_FALL_META.get("ip", ip),
                "location": LAST_FALL_META.get("location", "UNKNOWN"),
                "confidence": LAST_FALL_META.get("confidence", 0),
                "image_path": LAST_FALL_META.get("image_path", "UNKNOWN")
            }

                                       
            threading.Thread(target=send_email_alert, args=(meta,), daemon=True).start()

                                                       
            FALL_ALERT_FLAG.clear()

        ok, buf = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" +
            buf.tobytes() +
            b"\r\n"
        )

                                                                

@app.route("/")
def home():
    return render_template("indexproject.html")

@app.route("/dashboard")
def dashboard():
    return render_template("dashboard.html")

                                                               

@app.route("/enter_ip", methods=["GET", "POST"])
def enter_ip():
    if request.method == "POST":
        ip = request.form.get("ip")
        return redirect(url_for("stream_page", ip=ip))
    return render_template("enter_ip.html")

@app.route("/stream/<ip>")
def stream_page(ip):
    return render_template("ui.html", ip=ip)

@app.route("/stream_video/<ip>")
def stream_video(ip):
    url = f"http://{ip}:8080/video"
    return Response(gen_stream(url, ip), mimetype="multipart/x-mixed-replace; boundary=frame")

                                                               

@app.route("/webcam")
def webcam_page():
    return render_template("ui.html", ip=None)

@app.route("/webcam_stream")
def webcam_stream():
    return Response(gen_stream(0, "local"), mimetype="multipart/x-mixed-replace; boundary=frame")

                                                               

@app.route("/FrontPage", methods=["GET", "POST"])
def frontpage():
    if request.method == "POST":
        if "file" not in request.files:
            return "No file uploaded", 400

        file = request.files["file"]
        if file.filename == "":
            return "Empty filename", 400

        filename = secure_filename(file.filename)
        save_path = os.path.join(UPLOAD_FOLDER, filename)
        file.save(save_path)

        session["video_path"] = save_path
        return redirect(url_for("video"))

    return render_template("videoprojectnew.html")

@app.route("/video")
def video():
    path = session.get("video_path")
    if not path or not os.path.exists(path):
        return "No uploaded video found"
    return Response(gen_stream(path, "uploaded"), mimetype="multipart/x-mixed-replace; boundary=frame")

                                                               

@app.route("/about")
def about():
    return render_template("about.html")

                                                               

@app.route("/falls/<filename>")
def falls(filename):
    return send_from_directory(FALL_FOLDER, filename)

@app.route("/api/falls/recent")
def recent_falls():
    if not os.path.exists(LOG_FILE):
        return jsonify([])
    rows = list(csv.DictReader(open(LOG_FILE)))
    return jsonify(rows[::-1][:50])

                                                              

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
