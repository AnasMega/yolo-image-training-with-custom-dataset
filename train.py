from ultralytics import YOLO

model = YOLO("yolov8n.pt")  # small model for small dataset

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)