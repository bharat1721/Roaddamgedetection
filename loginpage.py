from flask import Flask, render_template, request, redirect, url_for, session, Response
import os
import cv2
import torch
import sqlite3
import numpy as np
import smtplib
import random
from email.message import EmailMessage
from ultralytics import YOLO
from PIL import Image
import io

# App config
app = Flask(
    __name__,
    static_folder=r"C:\Users\Koush\Downloads\project\static",
    template_folder=r"C:\Users\Koush\Downloads\project\templates"
)
app.secret_key = "your_secret_key"
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load model
MODEL_PATH = r"C:\Users\Koush\Downloads\project\best.pt"
model = YOLO(MODEL_PATH)

# Database
DB_PATH = r"C:\Users\Koush\Downloads\project\signup.db"

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            email TEXT NOT NULL UNIQUE,
            password TEXT NOT NULL,
            mobile TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

init_db()

# Globals for OTP (not safe in production)
otp = None
username = ""
email = ""
number = ""
password = ""

# 1. Process image
def process_image(image_path):
    results = model(image_path)
    output_image = results[0].plot()
    return output_image

# 2. Homepage
@app.route("/")
def index():
    if "user" not in session:
        return redirect(url_for("signin"))
    return render_template("index.html", user=session["user"])

# 3. Signup
@app.route("/signup", methods=["GET", "POST"])
def signup():
    global otp, username, email, number, password

    if request.method == "POST":
        username = request.form.get("username")
        email = request.form.get("email")
        number = request.form.get("mobile")
        password = request.form.get("password")

        otp = random.randint(1000, 9999)
        print("Generated OTP:", otp)

        msg = EmailMessage()
        msg.set_content(f"Your OTP is: {otp}")
        msg["Subject"] = "OTP Verification"
        msg["From"] = "youremail@gmail.com"
        msg["To"] = email

        try:
            s = smtplib.SMTP("smtp.gmail.com", 587)
            s.starttls()
            s.login("youremail@gmail.com", "yourpassword")
            s.send_message(msg)
            s.quit()
            return render_template("val.html")  # OTP input page
        except:
            return "Error sending OTP. Check your email settings."

    return render_template("signup.html")

# 4. OTP Verification
@app.route('/verify_otp', methods=['POST'])
def verify_otp():
    global otp, username, email, number, password
    entered_otp = request.form.get("otp")

    if int(entered_otp) == otp:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("INSERT INTO users (username, email, password, mobile) VALUES (?, ?, ?, ?)",
                       (username, email, password, number))
        conn.commit()
        conn.close()
        return redirect(url_for("signin"))
    else:
        return "Invalid OTP. Try again.", 400

# 5. Signin
@app.route("/signin", methods=["GET", "POST"])
def signin():
    if request.method == "POST":
        email = request.form.get("email")
        password = request.form.get("password")

        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("SELECT username FROM users WHERE email = ? AND password = ?", (email, password))
        user = cursor.fetchone()
        conn.close()

        if user:
            session["user"] = user[0]
            return redirect(url_for("index"))
        else:
            return "Invalid credentials. Try again.", 401

    return render_template("signin.html")

# 6. Logout
@app.route("/logout")
def logout():
    session.pop("user", None)
    return redirect(url_for("signin"))



@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return "No file uploaded", 400

    file = request.files["file"]
    if file.filename == "":
        return "No selected file", 400

    if file:
        filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
        file.save(filepath)

        # Process with YOLO
        output_image = process_image(filepath)
        output_path = os.path.join(app.config["UPLOAD_FOLDER"], "output.jpg")

        # Convert RGB (from YOLO) to BGR (for OpenCV saving)
        output_image = cv2.cvtColor(output_image, cv2.COLOR_RGB2BGR)
        cv2.imwrite(output_path, output_image)

        return render_template("result.html", input_image=file.filename, output_image="output.jpg")


@app.route("/video")
def video():
    return Response(gen(), mimetype='multipart/x-mixed-replace; boundary=frame')


def gen():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break

        # Convert frame to PIL image and pass through YOLO
        img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        results = model(img)
        processed = results[0].plot()  # YOLO-rendered image

        # Encode the result to JPEG
        processed = cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)
        ret, buffer = cv2.imencode('.jpg', processed)
        if not ret:
            continue

        frame = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()



@app.route("/about")
def about():
    return render_template("about.html")

@app.route("/notebook")
def notebook():
    return render_template("RoadDamage.html")
