# from fastapi import FastAPI, UploadFile, File
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import io

# app = FastAPI()

# model = YOLO("runs/detect/train-9/weights/best.pt")  # your trained model


# def calculate_area(box, img_width, img_height):
#     x1, y1, x2, y2 = box
#     bbox_area = (x2 - x1) * (y2 - y1)
#     img_area = img_width * img_height
#     return round((bbox_area / img_area) * 100, 2)


# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):
#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     img_width, img_height = image.size

#     results = model(image)

#     output = []

#     for r in results:
#         for box in r.boxes:
#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]

#             x1, y1, x2, y2 = box.xyxy[0].tolist()
#             area_pct = calculate_area((x1, y1, x2, y2), img_width, img_height)

#             output.append({
#                 "product": label,
#                 "confidence": float(box.conf[0]),
#                 "bbox": [x1, y1, x2, y2],
#                 "area_percent": area_pct
#             })

#     return {
#         "count": len(output),
#         "detections": output
#     }


# ------------------------------------------------------------------------------------------------------


# from fastapi import FastAPI, UploadFile, File
# from fastapi.responses import JSONResponse
# from fastapi.staticfiles import StaticFiles
# from ultralytics import YOLO
# from PIL import Image
# import numpy as np
# import io
# import os
# import uuid

# app = FastAPI()

# # Load YOLO model
# model = YOLO("runs/detect/train-9/weights/best.pt")

# # Output folder for detected images
# OUTPUT_DIR = "outputs"
# os.makedirs(OUTPUT_DIR, exist_ok=True)

# # Serve output images
# app.mount("/outputs", StaticFiles(directory="outputs"), name="outputs")


# def calculate_area(box, img_width, img_height):
#     x1, y1, x2, y2 = box
#     bbox_area = (x2 - x1) * (y2 - y1)
#     img_area = img_width * img_height
#     return round((bbox_area / img_area) * 100, 2)


# @app.post("/detect")
# async def detect(file: UploadFile = File(...)):

#     # Read uploaded image
#     image_bytes = await file.read()
#     image = Image.open(io.BytesIO(image_bytes)).convert("RGB")

#     img_width, img_height = image.size

#     # Run YOLO detection
#     results = model(image)

#     output = []

#     for r in results:
#         for box in r.boxes:

#             cls_id = int(box.cls[0])
#             label = model.names[cls_id]

#             x1, y1, x2, y2 = box.xyxy[0].tolist()

#             area_pct = calculate_area(
#                 (x1, y1, x2, y2),
#                 img_width,
#                 img_height
#             )

#             output.append({
#                 "product": label,
#                 "confidence": round(float(box.conf[0]), 4),
#                 "bbox": [x1, y1, x2, y2],
#                 "area_percent": area_pct
#             })

#     # Create image with labels + boxes
#     annotated_frame = results[0].plot()

#     # Convert numpy array to image
#     annotated_image = Image.fromarray(annotated_frame)

#     # Unique filename
#     filename = f"{uuid.uuid4().hex}.jpg"
#     filepath = os.path.join(OUTPUT_DIR, filename)

#     # Save annotated image
#     annotated_image.save(filepath)

#     # Return response
#     return JSONResponse({
#         "count": len(output),
#         "detections": output,
#         "detected_image": f"/outputs/{filename}"
#     })



import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Image Detection",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# CUSTOM SMALL UI CSS
# -----------------------------
st.markdown("""
<style>
    .main {
        max-width: 700px;
        margin: auto;
    }

    h1 {
        font-size: 28px !important;
        text-align: center;
    }

    .stImage img {
        border-radius: 10px;
    }

    .stDataFrame {
        font-size: 12px;
    }

    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🤖 YOLO Detection")
st.caption("Upload image for object detection")

# -----------------------------
# LOAD MODEL
# -----------------------------
model = YOLO("runs/detect/train-11/weights/best.pt")

# -----------------------------
# AREA FUNCTION
# -----------------------------
def calculate_area(box, img_width, img_height):
    x1, y1, x2, y2 = box
    bbox_area = (x2 - x1) * (y2 - y1)
    img_area = img_width * img_height
    return round((bbox_area / img_area) * 100, 2)

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# DETECTION
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")

    # Small image preview
    st.image(image, width=300, caption="Original")

    img_width, img_height = image.size

    with st.spinner("Detecting Objects..."):

        results = model(image)

        detections = []

        for r in results:
            for box in r.boxes:

                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                confidence = round(float(box.conf[0]), 3)

                area_pct = calculate_area(
                    (x1, y1, x2, y2),
                    img_width,
                    img_height
                )

                detections.append({
                    "Label": label,
                    "Conf": confidence,
                    "Area %": area_pct
                })

        # -----------------------------
        # DETECTED IMAGE
        # -----------------------------
        annotated_frame = results[0].plot()

        detected_image = Image.fromarray(annotated_frame)

        st.image(
            detected_image,
            width=400,
            caption="Detected Objects"
        )

        # -----------------------------
        # TABLE
        # -----------------------------
        if detections:

            st.success(f"{len(detections)} Object(s) Detected")

            df = pd.DataFrame(detections)

            st.dataframe(
                df,
                use_container_width=True,
                height=180
            )

        else:
            st.warning("No Objects Detected")

        # -----------------------------
        # DOWNLOAD BUTTON
        # -----------------------------
        buf = io.BytesIO()
        detected_image.save(buf, format="JPEG")

        st.download_button(
            "📥 Download Image",
            data=buf.getvalue(),
            file_name="detected.jpg",
            mime="image/jpeg"
        )