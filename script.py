import cv2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt
from reportlab.pdfgen import canvas
from datetime import datetime
import os
import multiprocessing
from flask import Flask, render_template, request, redirect, url_for, session, send_file

app = Flask(__name__)
app.secret_key = "supersecretkey"

# Initialize Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.7,
    model_complexity=1
)
mp_draw = mp.solutions.drawing_utils

# Initial circle sizes
left_radius = 50
right_radius = 50
max_radius = 100
min_radius = 30

# Finger tip indexes
tips_ids = [8, 12, 16, 20]


def is_hand_open(landmarks):
    open_fingers = 0
    for tip_id in tips_ids:
        if landmarks.landmark[tip_id].y < landmarks.landmark[tip_id - 2].y:
            open_fingers += 1
    return open_fingers >= 4


# Data storage for graph
time_stamps = []
left_sizes = []
right_sizes = []
start_time = datetime.now()


def generate_report():
    plt.figure(figsize=(8, 4))
    plt.plot(time_stamps, left_sizes, label="Left Circle Size", color="purple")
    plt.plot(time_stamps, right_sizes, label="Right Circle Size", color="magenta")
    plt.xlabel("Time (seconds)")
    plt.ylabel("Circle Size (radius)")
    plt.title("Hand Gesture-Controlled Circle Size Over Time")
    plt.legend()
    plt.grid()
    plt.savefig("static/circle_size_graph.png")

    pdf_filename = "Hand_Tracking_Report.pdf"
    c = canvas.Canvas(pdf_filename)
    c.setFont("Helvetica", 14)
    c.drawString(100, 800, "Hand Gesture-Controlled Circle Size Report")
    c.drawString(100, 780, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    c.drawImage("static/circle_size_graph.png", 100, 500, width=400, height=250)
    c.drawString(100, 470, "Summary:")
    c.drawString(100, 450, f"Max Left Circle Size: {max(left_sizes)}")
    c.drawString(100, 430, f"Min Left Circle Size: {min(left_sizes)}")
    c.drawString(100, 410, f"Max Right Circle Size: {max(right_sizes)}")
    c.drawString(100, 390, f"Min Right Circle Size: {min(right_sizes)}")
    c.save()
    return pdf_filename


def track_hand():
    global left_radius, right_radius
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open camera.")
        return
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Failed to capture image.")
                break
            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            black_screen = np.zeros((h, w, 3), dtype=np.uint8)

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks and result.multi_handedness:
                for hand_landmarks, handedness in zip(result.multi_hand_landmarks, result.multi_handedness):
                    label = handedness.classification[0].label
                    hand_open = is_hand_open(hand_landmarks)
                    mp_draw.draw_landmarks(black_screen, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                    if label == 'Left':
                        left_radius = min(left_radius + 3, max_radius) if hand_open else max(left_radius - 3,
                                                                                             min_radius)
                    elif label == 'Right':
                        right_radius = min(right_radius + 3, max_radius) if hand_open else max(right_radius - 3,
                                                                                               min_radius)

            elapsed_time = (datetime.now() - start_time).total_seconds()
            time_stamps.append(elapsed_time)
            left_sizes.append(left_radius)
            right_sizes.append(right_radius)

            cv2.circle(black_screen, (int(w * 0.3), int(h / 2)), left_radius, (255, 0, 255), -1)
            cv2.circle(black_screen, (int(w * 0.7), int(h / 2)), right_radius, (255, 0, 255), -1)

            cv2.imshow("Hand Controlled Circles", black_screen)
            if cv2.waitKey(1) & 0xFF == 27:
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


def start_tracking():
    process = multiprocessing.Process(target=track_hand)
    process.start()


@app.route('/', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        if username == "admin" and password == "admin":
            session['logged_in'] = True
            return redirect(url_for('dashboard'))
        else:
            return "Invalid credentials!"
    return render_template('login.html')


@app.route('/dashboard')
def dashboard():
    if not session.get('logged_in'):
        return redirect(url_for('login'))
    return render_template('dashboard.html')


@app.route('/start_tracking')
def start_tracking_route():
    start_tracking()
    return redirect(url_for('dashboard'))


@app.route('/download_report')
def download_report():
    pdf_filename = generate_report()
    return send_file(pdf_filename, as_attachment=True)


@app.route('/logout')
def logout():
    session.pop('logged_in', None)
    return redirect(url_for('login'))


if __name__ == '__main__':
    app.run(debug=True, threaded=True)