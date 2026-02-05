import cv2
import threading
import time
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO

# =============================
# LOAD MODEL (PAKAI NANO!)
# =============================
model = YOLO("yolov8n.pt")  # GANTI ke best.pt jika sudah ringan

# =============================
# CAMERA
# =============================
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# =============================
# GLOBAL STATE
# =============================
latest_frame = None
latest_annotated = None
running = False
lock = threading.Lock()

# =============================
# CAMERA THREAD
# =============================
def camera_loop():
    global latest_frame
    while True:
        if not running:
            time.sleep(0.1)
            continue

        ret, frame = cap.read()
        if ret:
            with lock:
                latest_frame = frame

# =============================
# YOLO THREAD
# =============================
def yolo_loop():
    global latest_annotated
    prev = time.time()

    while True:
        if not running or latest_frame is None:
            time.sleep(0.01)
            continue

        with lock:
            frame = latest_frame.copy()

        frame = cv2.resize(frame, (416, 416))  # 🔥 kecil = cepat
        results = model(frame, conf=0.4, verbose=False)
        annotated = results[0].plot()

        fps = int(1 / (time.time() - prev))
        prev = time.time()

        status = "AMAN"
        for box in results[0].boxes:
            cls = int(box.cls[0])
            label = model.names[cls].lower()
            if label in ["fire", "api", "smoke", "asap"]:
                status = "KEBAKARAN"

        cv2.putText(annotated, f"FPS: {fps}",
                    (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,255,0), 2)

        cv2.putText(annotated, f"STATUS: {status}",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX,
                    0.7, (0,0,255) if status=="KEBAKARAN" else (0,255,0), 2)

        with lock:
            latest_annotated = annotated

# =============================
# TKINTER UI
# =============================
root = tk.Tk()
root.title("YOLO Fire Detection - Desktop")

label = tk.Label(root)
label.pack()

def update_ui():
    if latest_annotated is not None:
        frame = cv2.cvtColor(latest_annotated, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(frame)
        imgtk = ImageTk.PhotoImage(image=img)
        label.imgtk = imgtk
        label.config(image=imgtk)

    root.after(30, update_ui)  # ~30 FPS UI

def start():
    global running
    running = True

def stop():
    global running
    running = False

btn_start = tk.Button(root, text="▶ START", width=20, command=start)
btn_start.pack(pady=5)

btn_stop = tk.Button(root, text="⏹ STOP", width=20, command=stop)
btn_stop.pack(pady=5)

# =============================
# THREAD START
# =============================
threading.Thread(target=camera_loop, daemon=True).start()
threading.Thread(target=yolo_loop, daemon=True).start()

update_ui()
root.mainloop()

cap.release()
