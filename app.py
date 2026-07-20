 

 
import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd
import io
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans

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
# LOAD MODEL
# -----------------------------
# model = YOLO("runs/detect/train-25/weights/best.pt")
model = YOLO("runs/detect/train-33/weights/best.pt") # new trained modal to be test
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
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "Confidence": float(box.conf[0])
                })

        if len(boxes_data) == 0:
            st.warning("⚠ No products detected")
            st.stop()

        # -----------------------------
        # AUTO ROW DETECTION
        # -----------------------------
        k = min(5, len(boxes_data))

        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(np.array(centers_y))

        for i in range(len(boxes_data)):
            boxes_data[i]["Row"] = int(labels[i]) + 1

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

        row_map = {old: new + 1 for new, old in enumerate(row_order)}
        df["Row"] = df["Row"].map(row_map)

        # -----------------------------
        # EYE-CATCHING LOGIC
        # -----------------------------
        df["Eye Catching"] = df["Row"].apply(
            lambda x: "YES" if x <= 2 else "NO"
        )

        # -----------------------------
        # OCCUPIED SPACE CALCULATION
        # -----------------------------
        occupied_width_px = df["x2"].max() - df["x1"].min()
        occupied_height_px = df["y2"].max() - df["y1"].min()

        occupied_width_in = round(occupied_width_px / dpi_x, 2)
        occupied_height_in = round(occupied_height_px / dpi_y, 2)

        width_util = round((occupied_width_in / shelf_width_in) * 100, 2)
        height_util = round((occupied_height_in / shelf_height_in) * 100, 2)

        # -----------------------------
        # IMAGE WITH HIGHLIGHTS
        # -----------------------------
        annotated = results[0].plot()
        annotated_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(annotated_image)

        for _, row in df.iterrows():

            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

            if row["Eye Catching"] == "YES":
                color = "green"
                label = "EYE LEVEL"
            else:
                color = "red"
                label = "NOT EYE LEVEL"

            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1, y1 - 10), label, fill=color)

        st.markdown("## 🎯 Detection with Shelf Intelligence")
        st.image(annotated_image, width=900)

        # -----------------------------
        # SHELF SIZE & UTILIZATION
        # -----------------------------
        st.markdown("## 📏 Shelf Capacity vs Usage")

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Shelf Width (in)", shelf_width_in)
            st.metric("Shelf Height (in)", shelf_height_in)

        with col2:
            st.metric("Occupied Width (in)", occupied_width_in)
            st.metric("Occupied Height (in)", occupied_height_in)

        with col3:
            st.metric("Width Utilization", f"{width_util}%")
            st.metric("Height Utilization", f"{height_util}%")

        # -----------------------------
        # INSIGHT
        # -----------------------------
        # st.markdown("## 📊 Shelf Insight")

        # if width_util > 80:
        #     st.error("⚠ Shelf width heavily crowded")
        # elif width_util > 50:
        #     st.warning("Moderate width usage")
        # else:
        #     st.success("Good width spacing")

        # if height_util > 80:
        #     st.error("⚠ Shelf height over-utilized")
        # elif height_util > 50:
        #     st.warning("Moderate height usage")
        # else:
        #     st.success("Good vertical spacing")

        # -----------------------------
        # ANALYTICS
        # -----------------------------
        st.markdown("## 📊 Overview")

        col1, col2 = st.columns(2)

        with col1:
            st.metric("Total Products", len(df))

        with col2:
            st.metric("Eye-Level Products", len(df[df["Eye Catching"] == "YES"]))

        # -----------------------------
        # ROW ANALYSIS
        # -----------------------------
        st.markdown("## 📦 Row Analysis")

        row_summary = df.groupby("Row").agg(
            Products=("Label", "count"),
            Avg_Confidence=("Confidence", "mean")
        ).reset_index()

        st.dataframe(row_summary, use_container_width=True)

        # -----------------------------
        # EYE / NON EYE TABLES
        # -----------------------------
        st.markdown("## 👁 Eye-Catching Products")

        eye_df = df[df["Eye Catching"] == "YES"]
        not_eye_df = df[df["Eye Catching"] == "NO"]

        if len(eye_df) > 0:
            st.success(f"{len(eye_df)} eye-level products")
            st.dataframe(eye_df, use_container_width=True)

        st.markdown("## ⚠ Not Eye-Catching Products")

        if len(not_eye_df) > 0:
            st.error(f"{len(not_eye_df)} not in eye-level zone")
            st.dataframe(not_eye_df, use_container_width=True)
        else:
            st.success("All products are eye-level 🎉")

        # -----------------------------
        # PRODUCT SUMMARY
        # -----------------------------
        st.markdown("## 📦 Product Summary")

        summary = df.groupby("Label").agg(
            Count=("Label", "count"),
            Avg_Confidence=("Confidence", "mean")
        ).reset_index()

        summary["Avg_Confidence"] = (summary["Avg_Confidence"] * 100).round(1)

        st.dataframe(summary, use_container_width=True)

        # -----------------------------
        # CHARTS
        # -----------------------------
        st.markdown("## 📊 Charts")

        fig1 = px.pie(summary, names="Label", values="Count", hole=0.4)
        st.plotly_chart(fig1, use_container_width=True)

        fig2 = px.bar(summary, x="Label", y="Count", text="Count")
        st.plotly_chart(fig2, use_container_width=True)

        # -----------------------------
        # DOWNLOAD IMAGE
        # -----------------------------
        st.markdown("## 📥 Download Result")

        buf = io.BytesIO()
        annotated_image.save(buf, format="JPEG")

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
st.caption("Smart Shelf Analytics | YOLO + Auto Row Detection + Shelf Utilization Intelligence")



 