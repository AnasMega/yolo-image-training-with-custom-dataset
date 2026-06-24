from ultralytics import YOLO

# model = YOLO("yolo11n.pt")
# model = YOLO("yolo11s.pt")

# Load a model
model = YOLO("yolo26n.yaml")  # build a new model from YAML
model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

#model = YOLO("yolov8n.pt")  # small model for small dataset

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640
)