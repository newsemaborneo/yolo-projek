from ultralytics import YOLO
import pandas as pd
import os


model = YOLO("yolov8n.pt")  # lightweight & cepat

model.train(
    data="Datasets-Fire/data.yaml",
    epochs=50,
    imgsz=512,
    batch=4,
    device="cpu",
    workers=2  
)


trained_model_path = "runs/detect/train/weights/best.pt"
model = YOLO(trained_model_path)

metrics = model.val(
    data="Datasets-Fire/data.yaml",
    imgsz=415,
    batch=16,
    conf=0.25,
    iou=0.5,
    save=True
)


# evaluation_results = {
#     "Precision": metrics.box.precision.mean(),
#     "Recall": metrics.box.recall.mean(),
#     "mAP50": metrics.box.map50,
#     "mAP50-95": metrics.box.map
# }

# df = pd.DataFrame([evaluation_results])
# df.to_csv("fire_model_evaluation.csv", index=False)

# print("\n=== HASIL EVALUASI MODEL ===")
# print(df)


print("\nMatriks evaluasi otomatis tersimpan di:")
print("runs/detect/val/")
