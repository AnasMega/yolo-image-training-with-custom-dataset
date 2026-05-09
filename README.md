# AI Detection Service using YOLOv8, YOLOv11 & YOLOv12

Developed by: **Anas Hussain (Mr Rumi)**

 

## Overview

This project is an AI-powered object detection service built with Python, FastAPI, and multiple YOLO models including:

* YOLOv8
* YOLOv11
* YOLOv12

The system is designed for high-performance object detection using datasets annotated in CVAT and exported in YOLO 1.1 format.

Each model was trained using different dataset sizes and image collections to compare accuracy, performance, and detection quality.

---

## Features

* FastAPI REST API
* Object Detection using YOLO
* Multiple model versions
* Bounding box detection
* Confidence scoring
* Area percentage calculation
* Annotated image generation
* CVAT annotation support
* YOLO 1.1 dataset format
* Multi-branch model management

---

## Technologies Used

* Python
* FastAPI
* Ultralytics YOLO
* OpenCV
* NumPy
* Pillow (PIL)
* CVAT
* Uvicorn

---

## Model Information

| Model   | Dataset Size   | Annotation Tool | Format   |
| ------- | -------------- | --------------- | -------- |
| YOLOv8  | Small Dataset  | CVAT            | YOLO 1.1 |
| YOLOv11 | Medium Dataset | CVAT            | YOLO 1.1 |
| YOLOv12 | Large Dataset  | CVAT            | YOLO 1.1 |

---

## Dataset Training

All datasets were:

* Annotated manually in CVAT
* Exported in YOLO 1.1 format
* Trained with different image sizes
* Organized in separate branches for model management

---

## Branch Structure

Different models and datasets are maintained in separate Git branches.

Example:

```bash
main
yolov8-model
yolov11-model
yolov12-model
custom-dataset-v1
large-dataset-training
```

---

## API Endpoint
http://127.0.0.1:8000/docs#/default/detect_detect_post
### Detect Objects

```http
POST /detect
```

### Request

Upload an image file using multipart/form-data.

### Response Example

```json
{
  "count": 2,
  "detections": [
    {
      "product": "bottle",
      "confidence": 0.94,
      "bbox": [120, 55, 300, 400],
      "area_percent": 12.44
    }
  ],
  "image_url": "/outputs/result.jpg"
}
```

---

## Run Project

Install dependencies:


Run server:
 uvicorn main:app --reload
 

 

## Future Improvements

* Real-time video detection
* Docker deployment
* GPU optimization
* Model comparison dashboard
* Multi-class analytics
* Cloud deployment

---

## Author

**Anas Hussain (Mr Rumi)**
AI Developer/Engineer | Computer Data/Vision Engineer | Team Lead Software Engineering
Linkedin:  https://www.linkedin.com/in/anas-hussain-00b658180?originalSubdomain=pk
 

## License

This project is open-source and available for research and development purposes.
