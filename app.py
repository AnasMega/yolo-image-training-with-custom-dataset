from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import numpy as np
import io

app = FastAPI()

model = YOLO("runs/detect/train-7/weights/best.pt")  # your trained model


def calculate_area(box, img_width, img_height):
    x1, y1, x2, y2 = box
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    return round((bbox_area / img_area) * 100, 2)


@app.post("/detect")
async def detect(file: UploadFile = File(...)):
    image_bytes = await file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

    img_width, img_height = image.size

    results = model(image)

    output = []

    for r in results:
        for box in r.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            area_pct = calculate_area((x1, y1, x2, y2), img_width, img_height)

            output.append({
                "product": label,
                "confidence": float(box.conf[0]),
                "bbox": [x1, y1, x2, y2],
                "area_percent": area_pct
            })

    return {
        "count": len(output),
        "detections": output
    }