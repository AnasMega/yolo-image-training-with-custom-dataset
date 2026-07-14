# from ultralytics import YOLO
# import psutil
# import os

# p = psutil.Process(os.getpid())
# p.nice(psutil.HIGH_PRIORITY_CLASS)

# # model = YOLO("yolo11n.pt")
# model = YOLO("yolo11s.pt")

# # Load a model
# # model = YOLO("yolo26n.yaml")  # build a new model from YAML
# # model = YOLO("yolo26n.pt")  # load a pretrained model (recommended for training)
# # model = YOLO("yolo26n.yaml").load("yolo26n.pt")  # build from YAML and transfer weights

# #model = YOLO("yolov8n.pt")  # small model for small dataset

# model.train(
#     data="data.yaml",
#     epochs=50,
#     imgsz=640
# )



from ultralytics import YOLO
import psutil
import os

# Set high process priority
p = psutil.Process(os.getpid())
p.nice(psutil.HIGH_PRIORITY_CLASS)

model = YOLO("yolo11s.pt")

model.train(
    data="data.yaml",
    epochs=50,
    imgsz=640,

    # Performance settings
    workers=32,      # Use all logical processors
    batch=64,        # Increase if you have enough RAM
    cache=True,      # Cache dataset in RAM
    device="cpu",    # Change to 0 if using an NVIDIA GPU
)