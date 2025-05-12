from flask import Flask, render_template, request, redirect, url_for, session, Response
import os
import cv2
import torch
import sqlite3
import numpy as np
from PIL import Image
import io
import smtplib
import random
from email.message import EmailMessage
from ultralytics import YOLO

# Initialize Flask app
app = Flask(
    __name__,
    static_folder=r"C:\Users\Koush\Downloads\project\static",
    template_folder=r"C:\Users\Koush\Downloads\project\templates"
)
app.secret_key = "your_secret_key"  # Replace with a secure key

# Uploads
UPLOAD_FOLDER = os.path.join(app.static_folder, "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# Load YOLO model
MODEL_PATH = r"C:\Users\Koush\Downloads\project\best.pt"
model = YOLO(MODEL_PATH)

# Database setup
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
