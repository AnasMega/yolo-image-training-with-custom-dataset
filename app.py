 



import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
import plotly.express as px

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Shelf Product Detection",
    page_icon="🤖",
    layout="centered"
)

# -----------------------------
# CUSTOM UI CSS
# -----------------------------
st.markdown("""
<style>

.main {
    max-width: 850px;
    margin: auto;
}

h1 {
    font-size: 32px !important;
    text-align: center;
    font-weight: 700;
}

.stImage img {
    border-radius: 12px;
    border: 1px solid #ddd;
}

.block-container {
    padding-top: 1rem;
    padding-bottom: 1rem;
}

.metric-card {
    background-color: #f8f9fa;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #e6e6e6;
    text-align: center;
}

.small-text {
    font-size: 13px;
    color: gray;
}

</style>
""", unsafe_allow_html=True)

# -----------------------------
# TITLE
# -----------------------------
st.title("🤖 Shelf Product Detection")
st.caption("Upload an image to detect shelf products category-wise")

# -----------------------------
# LOAD YOLO MODEL
# -----------------------------
model = YOLO("runs/detect/train3/weights/best.pt")

# -----------------------------
# AREA FUNCTION
# -----------------------------
def calculate_area(box, img_width, img_height):
    x1, y1, x2, y2 = box

    bbox_area = (x2 - x1) * (y2 - y1)
    image_area = img_width * img_height

    return round((bbox_area / image_area) * 100, 2)

# -----------------------------
# FILE UPLOADER
# -----------------------------
uploaded_file = st.file_uploader(
    "📤 Upload Shelf Image",
    type=["jpg", "jpeg", "png"]
)

# -----------------------------
# DETECTION SECTION
# -----------------------------
if uploaded_file:

    # Read image
    image = Image.open(uploaded_file).convert("RGB")

    # Show original image
    st.markdown("## 🖼 Original Image")
    st.image(image, width=350)

    img_width, img_height = image.size

    # -----------------------------
    # RUN DETECTION
    # -----------------------------
    with st.spinner("🔍 Detecting Products..."):

        results = model(image)

        detections = []

        for r in results:

            for box in r.boxes:

                # Class ID
                cls_id = int(box.cls[0])

                # Label name
                label = model.names[cls_id]

                # Bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].tolist()

                # Confidence
                confidence = round(float(box.conf[0]), 3)

                # Area %
                area_pct = calculate_area(
                    (x1, y1, x2, y2),
                    img_width,
                    img_height
                )

                detections.append({
                    "Label": label,
                    "Confidence": confidence,
                    "Area %": area_pct
                })

        # -----------------------------
        # DETECTED IMAGE
        # -----------------------------
        annotated_frame = results[0].plot()

        detected_image = Image.fromarray(annotated_frame)

        st.markdown("## 🎯 Detected Objects")

        st.image(
            detected_image,
            width=500
        )

        # -----------------------------
        # NO DETECTION
        # -----------------------------
        if len(detections) == 0:

            st.warning("⚠ No Objects Detected")

        else:

            # -----------------------------
            # DATAFRAME
            # -----------------------------
            df = pd.DataFrame(detections)

            # -----------------------------
            # SUMMARY TABLE
            # -----------------------------
            summary_df = (
                df.groupby("Label")
                .agg(
                    Count=("Label", "count"),
                    Avg_Confidence=("Confidence", "mean"),
                    Avg_Area=("Area %", "mean")
                )
                .reset_index()
            )

            summary_df["Avg_Confidence"] = (
                summary_df["Avg_Confidence"] * 100
            ).round(1)

            summary_df["Avg_Area"] = (
                summary_df["Avg_Area"]
            ).round(2)

            # -----------------------------
            # SUCCESS MESSAGE
            # -----------------------------
            st.success(
                f"✅ {len(detections)} Object(s) Detected"
            )

            # -----------------------------
            # CATEGORY METRICS
            # -----------------------------
            st.markdown("## 📦 Category Summary")

            cols = st.columns(len(summary_df))

            for idx, row in summary_df.iterrows():

                with cols[idx]:

                    st.metric(
                        label=row["Label"],
                        value=f"{row['Count']} Items",
                        delta=f"{row['Avg_Confidence']}% Conf"
                    )

            # -----------------------------
            # SUMMARY TABLE
            # -----------------------------
            st.markdown("## 📊 Detection Overview")

            st.dataframe(
                summary_df,
                use_container_width=True,
                hide_index=True
            )

            # -----------------------------
            # PIE CHART
            # -----------------------------
            st.markdown("## 🥧 Product Distribution")

            fig = px.pie(
                summary_df,
                names="Label",
                values="Count",
                hole=0.4
            )

            st.plotly_chart(
                fig,
                use_container_width=True
            )

            # -----------------------------
            # BAR CHART
            # -----------------------------
            st.markdown("## 📈 Product Count Chart")

            bar_fig = px.bar(
                summary_df,
                x="Label",
                y="Count",
                text="Count"
            )

            st.plotly_chart(
                bar_fig,
                use_container_width=True
            )

            # -----------------------------
            # CATEGORY DETAILS
            # -----------------------------
            st.markdown("## 🔍 Category Wise Details")

            for category in df["Label"].unique():

                category_df = df[df["Label"] == category]

                with st.expander(
                    f"📁 {category} ({len(category_df)} items)"
                ):

                    st.dataframe(
                        category_df,
                        use_container_width=True,
                        hide_index=True
                    )

            # -----------------------------
            # FULL DETECTION TABLE
            # -----------------------------
            st.markdown("## 🧾 All Detection Records")

            st.dataframe(
                df,
                use_container_width=True,
                hide_index=True,
                height=250
            )

        # -----------------------------
        # DOWNLOAD BUTTON
        # -----------------------------
        st.markdown("## 📥 Download Result")

        buf = io.BytesIO()

        detected_image.save(buf, format="JPEG")

        st.download_button(
            label="📸 Download Detected Image",
            data=buf.getvalue(),
            file_name="detected_output.jpg",
            mime="image/jpeg"
        )

# -----------------------------
# FOOTER
# -----------------------------
st.markdown("---")
st.caption("YOLOv8 Shelf Product Detection Dashboard")