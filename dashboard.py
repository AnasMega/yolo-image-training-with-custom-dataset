import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd
import io
import plotly.express as px
import numpy as np
import os

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Smart Shelf Analytics",
    page_icon="🛒",
    layout="wide"
)

# -----------------------------
# TITLE
# -----------------------------
st.title("🛒 Smart Shelf Product Analytics")
st.caption("AI-powered Retail Intelligence using YOLO + Shelf Analytics")
 

# -----------------------------
# SIDEBAR SETTINGS
# -----------------------------
st.sidebar.header("⚙ Shelf Settings")

TOTAL_SHELF_HEIGHT_FT = st.sidebar.slider(
    "Estimated Shelf Height (ft)",
    min_value=6,
    max_value=15,
    value=10
)

EYE_LEVEL_MIN = st.sidebar.slider(
    "Eye-Level Start (ft)",
    min_value=1,
    max_value=10,
    value=4
)

EYE_LEVEL_MAX = st.sidebar.slider(
    "Eye-Level End (ft)",
    min_value=1,
    max_value=12,
    value=8
)

MODEL_PATH = "runs/detect/train-25/weights/best.pt"

if not os.path.exists(MODEL_PATH):
    MODEL_PATH = "yolo11n.pt"  # fallback pretrained model

model = YOLO(MODEL_PATH)
# -----------------------------
# LOAD MODEL
# -----------------------------
# model = YOLO("runs/detect/train3/weights/best.pt")

# -----------------------------
# FILE UPLOAD
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Shelf Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# MAIN PIPELINE
# -----------------------------
if uploaded_file:

    image = Image.open(uploaded_file).convert("RGB")
    img_width, img_height = image.size

    # -----------------------------
    # SHELF SIZE (INCHES)
    # -----------------------------
    dpi = image.info.get("dpi", (96, 96))
    dpi_x, dpi_y = dpi

    shelf_width_in = round(img_width / dpi_x, 2)
    shelf_height_in = round(img_height / dpi_y, 2)

    # -----------------------------
    # SHOW IMAGE
    # -----------------------------
    st.markdown("## 🖼 Original Shelf Image")
    st.image(image, width=700)

    # -----------------------------
    # DETECTION
    # -----------------------------
    with st.spinner("🔍 Detecting Products..."):

        results = model(image, conf=0.35, iou=0.5)

        boxes_data = []
        centers_y = []

        for r in results:
            for box in r.boxes:

                cls_id = int(box.cls[0])
                label = model.names[cls_id]

                x1, y1, x2, y2 = box.xyxy[0].tolist()

                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2

                centers_y.append([cy])

                boxes_data.append({
                    "Label": label,
                    "x1": x1,
                    "y1": y1,
                    "x2": x2,
                    "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "Confidence": float(box.conf[0])
                })

        # -----------------------------
        # NO DETECTION
        # -----------------------------
        if len(boxes_data) == 0:
            st.warning("⚠ No products detected")
            st.stop()

         # -----------------------------
         # SMART ROW DETECTION
         # -----------------------------

        # Sort products by vertical center
        sorted_boxes = sorted(boxes_data, key=lambda x: x["cy"])

        row_threshold = img_height * 0.08
        current_row = 1

        sorted_boxes[0]["Row"] = current_row

        for i in range(1, len(sorted_boxes)):

            prev_cy = sorted_boxes[i - 1]["cy"]
            curr_cy = sorted_boxes[i]["cy"]

            # If vertical distance is large,
            # create a new row
            if abs(curr_cy - prev_cy) > row_threshold:
                current_row += 1

            sorted_boxes[i]["Row"] = current_row

        # Convert back
        boxes_data = sorted_boxes

        df = pd.DataFrame(boxes_data)

        # -----------------------------
        # SORT ROWS (TOP → BOTTOM)
        # -----------------------------
        row_order = (
            df.groupby("Row")["cy"]
            .mean()
            .sort_values()
            .index
        )

        row_map = {
            old: new + 1
            for new, old in enumerate(row_order)
        }

        df["Row"] = df["Row"].map(row_map)

        # -----------------------------
        # SMART HEIGHT DETECTION
        # -----------------------------
        def pixel_to_feet(cy):

            # Convert image pixel position into feet
            # top = highest shelf point
            # bottom = lowest shelf point

            relative_position = 1 - (cy / img_height)

            return round(
                relative_position * TOTAL_SHELF_HEIGHT_FT,
                2
            )

        df["Height_ft"] = df["cy"].apply(pixel_to_feet)

        # -----------------------------
        # EYE LEVEL DETECTION
        # -----------------------------
        df["Eye Catching"] = df["Height_ft"].apply(
            lambda h: (
                "YES"
                if EYE_LEVEL_MIN <= h <= EYE_LEVEL_MAX
                else "NO"
            )
        )

        # -----------------------------
        # OCCUPIED SPACE CALCULATION
        # -----------------------------
        occupied_width_px = df["x2"].max() - df["x1"].min()
        occupied_height_px = df["y2"].max() - df["y1"].min()

        occupied_width_in = round(
            occupied_width_px / dpi_x,
            2
        )

        occupied_height_in = round(
            occupied_height_px / dpi_y,
            2
        )

        width_util = round(
            (occupied_width_in / shelf_width_in) * 100,
            2
        )

        height_util = round(
            (occupied_height_in / shelf_height_in) * 100,
            2
        )

        # -----------------------------
        # IMAGE WITH HIGHLIGHTS
        # -----------------------------
        annotated = results[0].plot()
        annotated_image = Image.fromarray(annotated)

        draw = ImageDraw.Draw(annotated_image)

        for _, row in df.iterrows():

            x1, y1, x2, y2 = (
                row["x1"],
                row["y1"],
                row["x2"],
                row["y2"]
            )

            height = round(row["Height_ft"], 1)

            if row["Eye Catching"] == "YES":
                color = "green"
                label = f"EYE LEVEL ({height} ft)"
            else:
                color = "red"
                label = f"NOT EYE LEVEL ({height} ft)"

            draw.rectangle(
                [x1, y1, x2, y2],
                outline=color,
                width=4
            )

            draw.text(
                (x1, y1 - 15),
                label,
                fill=color
            )

        # -----------------------------
        # SHOW RESULT
        # -----------------------------
        st.markdown("## 🎯 Detection with Shelf Intelligence")


        st.image(
            annotated_image,
            width=900
        )

        # -----------------------------
        # SHELF SIZE & UTILIZATION
        # -----------------------------
        st.markdown("## 📏 Shelf Capacity vs Usage")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Shelf Width (in)",
                shelf_width_in
            )

            st.metric(
                "Shelf Height (in)",
                shelf_height_in
            )

        with col2:
            st.metric(
                "Occupied Width (in)",
                occupied_width_in
            )

            st.metric(
                "Occupied Height (in)",
                occupied_height_in
            )

        with col3:
            st.metric(
                "Width Utilization",
                f"{width_util}%"
            )

            st.metric(
                "Height Utilization",
                f"{height_util}%"
            )

         

        # -----------------------------
        # OVERVIEW
        # -----------------------------
        st.markdown("## 📊 Overview")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Products",
                len(df)
            )

        with col2:
            st.metric(
                "Eye-Level Products",
                len(df[df["Eye Catching"] == "YES"])
            )

        with col3:
            st.metric(
                "Non Eye-Level Products",
                len(df[df["Eye Catching"] == "NO"])
            )

        # -----------------------------
        
        # -----------------------------
        # EYE LEVEL PRODUCTS
        # -----------------------------
        st.markdown("## 👁 Eye-Catching Products")

        eye_df = df[df["Eye Catching"] == "YES"]

        if len(eye_df) > 0:

            st.success(
                f"{len(eye_df)} eye-level products detected"
            )

            st.dataframe(
                eye_df[
                    [
                        "Label",
                        "Height_ft",
                        "Row",
                        "Confidence"
                    ]
                ],
                use_container_width=True
            )

        # -----------------------------
        # NON EYE LEVEL PRODUCTS
        # -----------------------------
        st.markdown("## ⚠ Not Eye-Catching Products")

        not_eye_df = df[df["Eye Catching"] == "NO"]

        if len(not_eye_df) > 0:

            st.error(
                f"{len(not_eye_df)} products outside eye-level zone"
            )

            st.dataframe(
                not_eye_df[
                    [
                        "Label",
                        "Height_ft",
                        "Row",
                        "Confidence"
                    ]
                ],
                use_container_width=True
            )

        else:
            st.success("🎉 All products are eye-level")

        # -----------------------------
        # PRODUCT SUMMARY
        # -----------------------------
        st.markdown("## 📦 Product Summary")

        summary = df.groupby("Label").agg(
            Count=("Label", "count"),
            Avg_Confidence=("Confidence", "mean")
        ).reset_index()

        summary["Avg_Confidence"] = (
            summary["Avg_Confidence"] * 100
        ).round(1)

        st.dataframe(
            summary,
            use_container_width=True
        )

        # -----------------------------
        # CHARTS
        # -----------------------------
        st.markdown("## 📊 Charts")

        fig1 = px.pie(
            summary,
            names="Label",
            values="Count",
            hole=0.4,
            title="Product Distribution"
        )

        st.plotly_chart(
            fig1,
            use_container_width=True
        )

        fig2 = px.bar(
            summary,
            x="Label",
            y="Count",
            text="Count",
            title="Product Count"
        )

        st.plotly_chart(
            fig2,
            use_container_width=True
        )

        # -----------------------------
        # DOWNLOAD RESULT
        # -----------------------------
        st.markdown("## 📥 Download Result")

        buf = io.BytesIO()

        annotated_image.save(
            buf,
            format="JPEG"
        )

        st.download_button(
            label="📸 Download Image",
            data=buf.getvalue(),
            file_name="smart_shelf_result.jpg",
            mime="image/jpeg"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")

st.caption(
    "Smart Shelf Analytics | YOLO + Auto Row Detection + Shelf Utilization Intelligence"
)