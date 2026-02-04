
from ultralytics import YOLO
import cv2
import time

MODEL_PATH = "runs/detect/train/weights/best.pt"
model = YOLO(MODEL_PATH)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("❌ Kamera laptop tidak bisa dibuka")
    exit()

print("✅ Kamera aktif, mulai deteksi api...")

while True:
    start_time = time.time()

    ret, frame = cap.read()
    if not ret:
        print("❌ Gagal membaca frame")
        break

    results = model(
        frame,
        imgsz=416,
        conf=0.4,   # confidence threshold
        iou=0.5,
        verbose=False
    )


    fire_detected = False

    for box in results[0].boxes:
        cls_id = int(box.cls[0])
        conf = float(box.conf[0])
        label = model.names[cls_id]

        if label == "fire and smoke":
            fire_detected = True

    annotated_frame = results[0].plot(labels = False , conf = True)

    # HITUNG FPS
    fps = 1 / (time.time() - start_time)

    # TEXT INFO
    status_text = "🔥 KEBAKARAN TERDETEKSI" if fire_detected else "✅ AMAN"
    status_color = (0, 0, 255) if fire_detected else (0, 255, 0)

    cv2.putText(
        annotated_frame,
        status_text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        status_color,
        2
    )

    cv2.putText(
        annotated_frame,
        f"FPS: {int(fps)}",
        (20, 80),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (255, 255, 0),
        2
    )

 
    cv2.imshow("YOLOv8 Fire Detection - Webcam", annotated_frame)

    # =====================================
    # 8. KELUAR DENGAN TOMBOL 'q'
    # =====================================
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("🛑 Program dihentikan")
