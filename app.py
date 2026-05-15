 

# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import pandas as pd
# import io
# import plotly.express as px

# # -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="Shelf Product Detection",
#     page_icon="🤖",
#     layout="centered"
# )

# # -----------------------------
# # CUSTOM UI CSS
# # -----------------------------
# st.markdown("""
# <style>

# .main {
#     max-width: 850px;
#     margin: auto;
# }

# h1 {
#     font-size: 32px !important;
#     text-align: center;
#     font-weight: 700;
# }

# .stImage img {
#     border-radius: 12px;
#     border: 1px solid #ddd;
# }

# .block-container {
#     padding-top: 1rem;
#     padding-bottom: 1rem;
# }

# .metric-card {
#     background-color: #f8f9fa;
#     padding: 15px;
#     border-radius: 12px;
#     border: 1px solid #e6e6e6;
#     text-align: center;
# }

# .small-text {
#     font-size: 13px;
#     color: gray;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------
# # TITLE
# # -----------------------------
# st.title("🤖 Shelf Product Detection")
# st.caption("Upload an image to detect shelf products category-wise")

# # -----------------------------
# # LOAD YOLO MODEL
# # -----------------------------
# model = YOLO("runs/detect/train-17/weights/best.pt")

# # -----------------------------
# # AREA FUNCTION
# # -----------------------------
# def calculate_area(box, img_width, img_height):
#     x1, y1, x2, y2 = box

#     bbox_area = (x2 - x1) * (y2 - y1)
#     image_area = img_width * img_height

#     return round((bbox_area / image_area) * 100, 2)

# # -----------------------------
# # FILE UPLOADER
# # -----------------------------
# uploaded_file = st.file_uploader(
#     "📤 Upload Shelf Image",
#     type=["jpg", "jpeg", "png"]
# )

# # -----------------------------
# # DETECTION SECTION
# # -----------------------------
# if uploaded_file:

#     # Read image
#     image = Image.open(uploaded_file).convert("RGB")

#     # Show original image
#     st.markdown("## 🖼 Original Image")
#     st.image(image, width=350)

#     img_width, img_height = image.size

#     # -----------------------------
#     # RUN DETECTION
#     # -----------------------------
#     with st.spinner("🔍 Detecting Products..."):

#         results = model(image)

#         detections = []

#         for r in results:

#             for box in r.boxes:

#                 # Class ID
#                 cls_id = int(box.cls[0])

#                 # Label name
#                 label = model.names[cls_id]

#                 # Bounding box coordinates
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()

#                 # Confidence
#                 confidence = round(float(box.conf[0]), 3)

#                 # Area %
#                 area_pct = calculate_area(
#                     (x1, y1, x2, y2),
#                     img_width,
#                     img_height
#                 )

#                 detections.append({
#                     "Label": label,
#                     "Confidence": confidence,
#                     "Area %": area_pct
#                 })

#         # -----------------------------
#         # DETECTED IMAGE
#         # -----------------------------
#         annotated_frame = results[0].plot()

#         detected_image = Image.fromarray(annotated_frame)

#         st.markdown("## 🎯 Detected Objects")

#         st.image(
#             detected_image,
#             width=500
#         )

#         # -----------------------------
#         # NO DETECTION
#         # -----------------------------
#         if len(detections) == 0:

#             st.warning("⚠ No Objects Detected")

#         else:

#             # -----------------------------
#             # DATAFRAME
#             # -----------------------------
#             df = pd.DataFrame(detections)

#             # -----------------------------
#             # SUMMARY TABLE
#             # -----------------------------
#             summary_df = (
#                 df.groupby("Label")
#                 .agg(
#                     Count=("Label", "count"),
#                     Avg_Confidence=("Confidence", "mean"),
#                     Avg_Area=("Area %", "mean")
#                 )
#                 .reset_index()
#             )

#             summary_df["Avg_Confidence"] = (
#                 summary_df["Avg_Confidence"] * 100
#             ).round(1)

#             summary_df["Avg_Area"] = (
#                 summary_df["Avg_Area"]
#             ).round(2)

#             # -----------------------------
#             # SUCCESS MESSAGE
#             # -----------------------------
#             st.success(
#                 f"✅ {len(detections)} Object(s) Detected"
#             )

#             # -----------------------------
#             # CATEGORY METRICS
#             # -----------------------------
#             st.markdown("## 📦 Category Summary")

#             cols = st.columns(len(summary_df))

#             for idx, row in summary_df.iterrows():

#                 with cols[idx]:

#                     st.metric(
#                         label=row["Label"],
#                         value=f"{row['Count']} Items",
#                         delta=f"{row['Avg_Confidence']}% Conf"
#                     )

#             # -----------------------------
#             # SUMMARY TABLE
#             # -----------------------------
#             st.markdown("## 📊 Detection Overview")

#             st.dataframe(
#                 summary_df,
#                 use_container_width=True,
#                 hide_index=True
#             )

#             # -----------------------------
#             # PIE CHART
#             # -----------------------------
#             st.markdown("## 🥧 Product Distribution")

#             fig = px.pie(
#                 summary_df,
#                 names="Label",
#                 values="Count",
#                 hole=0.4
#             )

#             st.plotly_chart(
#                 fig,
#                 use_container_width=True
#             )

#             # -----------------------------
#             # BAR CHART
#             # -----------------------------
#             st.markdown("## 📈 Product Count Chart")

#             bar_fig = px.bar(
#                 summary_df,
#                 x="Label",
#                 y="Count",
#                 text="Count"
#             )

#             st.plotly_chart(
#                 bar_fig,
#                 use_container_width=True
#             )

#             # -----------------------------
#             # CATEGORY DETAILS
#             # -----------------------------
#             st.markdown("## 🔍 Category Wise Details")

#             for category in df["Label"].unique():

#                 category_df = df[df["Label"] == category]

#                 with st.expander(
#                     f"📁 {category} ({len(category_df)} items)"
#                 ):

#                     st.dataframe(
#                         category_df,
#                         use_container_width=True,
#                         hide_index=True
#                     )

#             # -----------------------------
#             # FULL DETECTION TABLE
#             # -----------------------------
#             st.markdown("## 🧾 All Detection Records")

#             st.dataframe(
#                 df,
#                 use_container_width=True,
#                 hide_index=True,
#                 height=250
#             )

#         # -----------------------------
#         # DOWNLOAD BUTTON
#         # -----------------------------
#         st.markdown("## 📥 Download Result")

#         buf = io.BytesIO()

#         detected_image.save(buf, format="JPEG")

#         st.download_button(
#             label="📸 Download Detected Image",
#             data=buf.getvalue(),
#             file_name="detected_output.jpg",
#             mime="image/jpeg"
#         )

# # -----------------------------
# # FOOTER
# # -----------------------------
# st.markdown("---")
# st.caption("YOLOv8 Shelf Product Detection Dashboard")

# -----------------------------------------------------------------------------
 
# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image
# import pandas as pd
# import io
# import plotly.express as px

# # -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="Shelf Product Detection",
#     page_icon="🤖",
#     layout="centered"
# )

# # -----------------------------
# # CUSTOM CSS
# # -----------------------------
# st.markdown("""
# <style>

# .main {
#     max-width: 1000px;
#     margin: auto;
# }

# h1 {
#     text-align: center;
#     font-size: 34px !important;
#     font-weight: 700;
# }

# .stImage img {
#     border-radius: 12px;
#     border: 1px solid #ddd;
# }

# .block-container {
#     padding-top: 1rem;
#     padding-bottom: 1rem;
# }

# </style>
# """, unsafe_allow_html=True)

# # -----------------------------
# # TITLE
# # -----------------------------
# st.title("🤖 Shelf Product Detection")
# st.caption("AI Based Shelf Analytics using RT-DETR / YOLO")

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("runs/detect/train-17/weights/best.pt")

# # -----------------------------
# # AREA FUNCTION
# # -----------------------------
# def calculate_area(box, img_width, img_height):

#     x1, y1, x2, y2 = box

#     bbox_area = (x2 - x1) * (y2 - y1)

#     image_area = img_width * img_height

#     return round((bbox_area / image_area) * 100, 2)

# # -----------------------------
# # FILE UPLOADER
# # -----------------------------
# uploaded_file = st.file_uploader(
#     "📤 Upload Shelf Image",
#     type=["jpg", "jpeg", "png"]
# )

# # -----------------------------
# # DETECTION
# # -----------------------------
# if uploaded_file:

#     # Read image
#     image = Image.open(uploaded_file).convert("RGB")

#     # Show Original Image
#     st.markdown("## 🖼 Original Image")

#     st.image(
#         image,
#         width=400
#     )

#     # Image dimensions
#     img_width, img_height = image.size

#     # DPI INFO
#     dpi = image.info.get("dpi", (96, 96))

#     dpi_x, dpi_y = dpi

#     # Inches conversion
#     image_width_inches = round(img_width / dpi_x, 2)
#     image_height_inches = round(img_height / dpi_y, 2)

#     # -----------------------------
#     # IMAGE + SHELF DIMENSIONS
#     # -----------------------------
#     st.markdown("## 📏 Shelf/Image Dimensions")

#     col1, col2, col3, col4 = st.columns(4)

#     with col1:
#         st.metric(
#             "Width (px)",
#             f"{img_width}"
#         )

#     with col2:
#         st.metric(
#             "Height (px)",
#             f"{img_height}"
#         )

#     with col3:
#         st.metric(
#             "Width (in)",
#             f"{image_width_inches}"
#         )

#     with col4:
#         st.metric(
#             "Height (in)",
#             f"{image_height_inches}"
#         )

#     # -----------------------------
#     # DETECTION
#     # -----------------------------
#     with st.spinner("🔍 Detecting Products..."):

#         results = model(
#             image,
#             conf=0.35,
#             iou=0.5
#         )

#         detections = []

#         # Shelf dimensions
#         shelf_width_px = img_width
#         shelf_height_px = img_height

#         # Eye level region
#         eye_level_top = img_height * 0.35
#         eye_level_bottom = img_height * 0.65

#         total_occupied_width = 0

#         # -----------------------------
#         # LOOP DETECTIONS
#         # -----------------------------
#         for r in results:

#             for box in r.boxes:

#                 # Class ID
#                 cls_id = int(box.cls[0])

#                 # Label
#                 label = model.names[cls_id]

#                 # Bounding Box
#                 x1, y1, x2, y2 = box.xyxy[0].tolist()

#                 # Product dimensions
#                 product_width = x2 - x1
#                 product_height = y2 - y1

#                 # Width occupied %
#                 width_pct = round(
#                     (product_width / shelf_width_px) * 100,
#                     2
#                 )

#                 # Height occupied %
#                 height_pct = round(
#                     (product_height / shelf_height_px) * 100,
#                     2
#                 )

#                 # Total occupied width
#                 total_occupied_width += width_pct

#                 # Area %
#                 area_pct = calculate_area(
#                     (x1, y1, x2, y2),
#                     img_width,
#                     img_height
#                 )

#                 # Confidence
#                 confidence = round(
#                     float(box.conf[0]),
#                     3
#                 )

#                 # Center point
#                 center_y = (y1 + y2) / 2

#                 # Eye level check
#                 eye_level = (
#                     "YES"
#                     if eye_level_top <= center_y <= eye_level_bottom
#                     else "NO"
#                 )

#                 detections.append({

#                     "Label": label,
#                     "Confidence": confidence,
#                     "Area %": area_pct,
#                     "Width %": width_pct,
#                     "Height %": height_pct,
#                     "Eye Level": eye_level

#                 })

#         # -----------------------------
#         # DETECTED IMAGE
#         # -----------------------------
#         annotated_frame = results[0].plot()

#         detected_image = Image.fromarray(
#             annotated_frame
#         )

#         st.markdown("## 🎯 Detected Objects")

#         st.image(
#             detected_image,
#             width=700
#         )

#         # -----------------------------
#         # NO DETECTIONS
#         # -----------------------------
#         if len(detections) == 0:

#             st.warning("⚠ No Objects Detected")

#         else:

#             # -----------------------------
#             # DATAFRAME
#             # -----------------------------
#             df = pd.DataFrame(detections)

#             # -----------------------------
#             # SHELF OCCUPANCY
#             # -----------------------------
#             st.markdown("## 🏪 Shelf Occupancy")

#             col1, col2 = st.columns(2)

#             with col1:

#                 st.metric(
#                     "Shelf Width Occupied",
#                     f"{round(total_occupied_width, 2)}%"
#                 )

#             with col2:

#                 free_space = round(
#                     100 - total_occupied_width,
#                     2
#                 )

#                 st.metric(
#                     "Free Shelf Space",
#                     f"{free_space}%"
#                 )

#             # -----------------------------
#             # SUMMARY TABLE
#             # -----------------------------
#             summary_df = (

#                 df.groupby("Label")

#                 .agg(

#                     Count=("Label", "count"),

#                     Avg_Confidence=(
#                         "Confidence",
#                         "mean"
#                     ),

#                     Avg_Area=(
#                         "Area %",
#                         "mean"
#                     ),

#                     Avg_Width=(
#                         "Width %",
#                         "mean"
#                     ),

#                     Avg_Height=(
#                         "Height %",
#                         "mean"
#                     ),

#                     Eye_Level_Count=(
#                         "Eye Level",
#                         lambda x: (x == "YES").sum()
#                     )

#                 )

#                 .reset_index()

#             )

#             # Formatting
#             summary_df["Avg_Confidence"] = (
#                 summary_df["Avg_Confidence"] * 100
#             ).round(1)

#             summary_df["Avg_Area"] = (
#                 summary_df["Avg_Area"]
#             ).round(2)

#             summary_df["Avg_Width"] = (
#                 summary_df["Avg_Width"]
#             ).round(2)

#             summary_df["Avg_Height"] = (
#                 summary_df["Avg_Height"]
#             ).round(2)

#             # -----------------------------
#             # SUCCESS MESSAGE
#             # -----------------------------
#             st.success(
#                 f"✅ {len(detections)} Product(s) Detected"
#             )

#             # -----------------------------
#             # CATEGORY METRICS
#             # -----------------------------
#             st.markdown("## 📦 Category Summary")

#             cols = st.columns(len(summary_df))

#             for idx, row in summary_df.iterrows():

#                 with cols[idx]:

#                     st.metric(
#                         label=row["Label"],
#                         value=f"{row['Count']} Items",
#                         delta=f"{row['Avg_Confidence']}% Conf"
#                     )

#             # -----------------------------
#             # OVERVIEW TABLE
#             # -----------------------------
#             st.markdown("## 📊 Detection Overview")

#             st.dataframe(
#                 summary_df,
#                 use_container_width=True,
#                 hide_index=True
#             )

#             # -----------------------------
#             # PIE CHART
#             # -----------------------------
#             st.markdown("## 🥧 Product Distribution")

#             pie_fig = px.pie(
#                 summary_df,
#                 names="Label",
#                 values="Count",
#                 hole=0.4
#             )

#             st.plotly_chart(
#                 pie_fig,
#                 use_container_width=True
#             )

#             # -----------------------------
#             # BAR CHART
#             # -----------------------------
#             st.markdown("## 📈 Product Count Chart")

#             bar_fig = px.bar(
#                 summary_df,
#                 x="Label",
#                 y="Count",
#                 text="Count"
#             )

#             st.plotly_chart(
#                 bar_fig,
#                 use_container_width=True
#             )

#             # -----------------------------
#             # CATEGORY DETAILS
#             # -----------------------------
#             st.markdown("## 🔍 Category Wise Details")

#             for category in df["Label"].unique():

#                 category_df = df[
#                     df["Label"] == category
#                 ]

#                 with st.expander(
#                     f"📁 {category} ({len(category_df)} items)"
#                 ):

#                     st.dataframe(
#                         category_df,
#                         use_container_width=True,
#                         hide_index=True
#                     )

#             # -----------------------------
#             # FULL TABLE
#             # -----------------------------
#             st.markdown("## 🧾 All Detection Records")

#             st.dataframe(
#                 df,
#                 use_container_width=True,
#                 hide_index=True,
#                 height=350
#             )

#         # -----------------------------
#         # DOWNLOAD BUTTON
#         # -----------------------------
#         st.markdown("## 📥 Download Result")

#         buf = io.BytesIO()

#         detected_image.save(
#             buf,
#             format="JPEG"
#         )

#         st.download_button(
#             label="📸 Download Detected Image",
#             data=buf.getvalue(),
#             file_name="detected_output.jpg",
#             mime="image/jpeg"
#         )

# # -----------------------------
# # FOOTER
# # -----------------------------
# st.markdown("---")

# st.caption(
#     "AI Shelf Analytics Dashboard using RT-DETR / YOLO"
# )
# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image, ImageDraw
# import pandas as pd
# import io
# import plotly.express as px
# import numpy as np
# from sklearn.cluster import KMeans

# # -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="Smart Shelf Analytics",
#     page_icon="🛒",
#     layout="wide"
# )

# # -----------------------------
# # TITLE
# # -----------------------------
# st.title("🛒 Smart Shelf Product Analytics")
# st.caption("AI-powered Shelf Intelligence using YOLO + Auto Row Detection")

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("runs/detect/train-17/weights/best.pt")

# # -----------------------------
# # AREA FUNCTION
# # -----------------------------
# def calculate_area(box, img_width, img_height):
#     x1, y1, x2, y2 = box
#     return round(((x2 - x1) * (y2 - y1)) / (img_width * img_height) * 100, 2)

# # -----------------------------
# # UPLOAD IMAGE
# # -----------------------------
# uploaded_file = st.file_uploader(
#     "📤 Upload Shelf Image",
#     type=["jpg", "jpeg", "png"]
# )

# # -----------------------------
# # MAIN PIPELINE
# # -----------------------------
# if uploaded_file:

#     image = Image.open(uploaded_file).convert("RGB")
#     img_width, img_height = image.size

#     st.markdown("## 🖼 Original Image")
#     st.image(image, width=700)

#     # -----------------------------
#     # DETECTION
#     # -----------------------------
#     with st.spinner("🔍 Detecting Products..."):

#         results = model(image, conf=0.35, iou=0.5)

#         boxes_data = []
#         centers_y = []

#         # -----------------------------
#         # EXTRACT BOXES
#         # -----------------------------
#         for r in results:
#             for box in r.boxes:

#                 cls_id = int(box.cls[0])
#                 label = model.names[cls_id]

#                 x1, y1, x2, y2 = box.xyxy[0].tolist()

#                 cx = (x1 + x2) / 2
#                 cy = (y1 + y2) / 2

#                 centers_y.append([cy])

#                 boxes_data.append({
#                     "Label": label,
#                     "x1": x1, "y1": y1, "x2": x2, "y2": y2,
#                     "cx": cx,
#                     "cy": cy,
#                     "Confidence": float(box.conf[0]),
#                     "Width_px": x2 - x1,
#                     "Height_px": y2 - y1
#                 })

#         # -----------------------------
#         # CHECK IF DETECTIONS EXIST
#         # -----------------------------
#         if len(boxes_data) == 0:
#             st.warning("⚠ No products detected")
#             st.stop()

#         # -----------------------------
#         # AUTO ROW DETECTION
#         # -----------------------------
#         k = min(5, len(boxes_data))
#         kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
#         labels = kmeans.fit_predict(np.array(centers_y))

#         for i in range(len(boxes_data)):
#             boxes_data[i]["Row"] = int(labels[i]) + 1

#         df = pd.DataFrame(boxes_data)

#         # -----------------------------
#         # SORT ROWS (TOP → BOTTOM)
#         # -----------------------------
#         row_order = (
#             df.groupby("Row")["cy"]
#             .mean()
#             .sort_values()
#             .index
#         )

#         row_map = {old: new + 1 for new, old in enumerate(row_order)}
#         df["Row"] = df["Row"].map(row_map)

#         # -----------------------------
#         # EYE-CATCHING LOGIC
#         # -----------------------------
#         df["Eye Catching"] = df["Row"].apply(
#             lambda x: "YES" if x <= 2 else "NO"
#         )

#         # -----------------------------
#         # DETECTED IMAGE WITH HIGHLIGHT
#         # -----------------------------
#         annotated = results[0].plot()
#         annotated_image = Image.fromarray(annotated)
#         draw = ImageDraw.Draw(annotated_image)

#         for _, row in df.iterrows():

#             x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

#             if row["Eye Catching"] == "YES":
#                 color = "green"
#                 label = "EYE LEVEL"
#             else:
#                 color = "red"
#                 label = "NOT EYE LEVEL"

#             draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
#             draw.text((x1, y1 - 10), label, fill=color)

#         st.markdown("## 🎯 Detection with Eye-Level Highlighting")
#         st.image(annotated_image, width=900)

#         # -----------------------------
#         # ANALYTICS
#         # -----------------------------
#         st.markdown("## 📊 Shelf Analytics")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.metric("Total Products", len(df))

#         with col2:
#             st.metric("Eye-Level Products", len(df[df["Eye Catching"] == "YES"]))

#         # -----------------------------
#         # ROW ANALYSIS
#         # -----------------------------
#         st.markdown("## 📦 Row Wise Distribution")

#         row_summary = df.groupby("Row").agg(
#             Products=("Label", "count"),
#             Avg_Confidence=("Confidence", "mean")
#         ).reset_index()

#         st.dataframe(row_summary, use_container_width=True)

#         # -----------------------------
#         # EYE LEVEL PRODUCTS
#         # -----------------------------
#         st.markdown("## 👀 Eye-Catching Products")

#         eye_df = df[df["Eye Catching"] == "YES"]
#         not_eye_df = df[df["Eye Catching"] == "NO"]

#         if len(eye_df) > 0:
#             st.success(f"{len(eye_df)} products in eye-level zone")
#             st.dataframe(eye_df, use_container_width=True)

#         # -----------------------------
#         # NOT EYE LEVEL PRODUCTS
#         # -----------------------------
#         st.markdown("## ⚠ Not Eye-Catching Products")

#         if len(not_eye_df) > 0:
#             st.error(f"{len(not_eye_df)} products NOT in eye-level zone")
#             st.dataframe(not_eye_df, use_container_width=True)
#         else:
#             st.success("All products are eye-level 🎉")

#         # -----------------------------
#         # PRODUCT SUMMARY
#         # -----------------------------
#         st.markdown("## 📦 Product Summary")

#         summary = df.groupby("Label").agg(
#             Count=("Label", "count"),
#             Avg_Confidence=("Confidence", "mean")
#         ).reset_index()

#         summary["Avg_Confidence"] = (summary["Avg_Confidence"] * 100).round(1)

#         st.dataframe(summary, use_container_width=True)

#         # -----------------------------
#         # CHARTS
#         # -----------------------------
#         st.markdown("## 📊 Charts")

#         fig1 = px.pie(summary, names="Label", values="Count", hole=0.4)
#         st.plotly_chart(fig1, use_container_width=True)

#         fig2 = px.bar(summary, x="Label", y="Count", text="Count")
#         st.plotly_chart(fig2, use_container_width=True)

#         # -----------------------------
#         # DOWNLOAD IMAGE
#         # -----------------------------
#         st.markdown("## 📥 Download Result")

#         buf = io.BytesIO()
#         annotated_image.save(buf, format="JPEG")

#         st.download_button(
#             label="📸 Download Annotated Image",
#             data=buf.getvalue(),
#             file_name="shelf_analysis.jpg",
#             mime="image/jpeg"
#         )

# # -----------------------------
# # FOOTER
# # -----------------------------
# st.markdown("---")
# st.caption("Smart Shelf Analytics System using YOLO + Auto Row Detection + Eye-Level Intelligence")


# import streamlit as st
# from ultralytics import YOLO
# from PIL import Image, ImageDraw
# import pandas as pd
# import io
# import plotly.express as px
# import numpy as np
# from sklearn.cluster import KMeans

# # -----------------------------
# # PAGE CONFIG
# # -----------------------------
# st.set_page_config(
#     page_title="Smart Shelf Analytics",
#     page_icon="🛒",
#     layout="wide"
# )

# # -----------------------------
# # TITLE
# # -----------------------------
# st.title("🛒 Smart Shelf Product Analytics")
# st.caption("AI-powered Retail Intelligence using YOLO + Shelf Analytics")

# # -----------------------------
# # LOAD MODEL
# # -----------------------------
# model = YOLO("runs/detect/train-17/weights/best.pt")

# # -----------------------------
# # FILE UPLOAD
# # -----------------------------
# uploaded_file = st.file_uploader(
#     "📤 Upload Shelf Image",
#     type=["jpg", "jpeg", "png"]
# )

# # -----------------------------
# # MAIN PIPELINE
# # -----------------------------
# if uploaded_file:

#     image = Image.open(uploaded_file).convert("RGB")
#     img_width, img_height = image.size

#     # -----------------------------
#     # SHELF SIZE (INCHES)
#     # -----------------------------
#     dpi = image.info.get("dpi", (96, 96))
#     dpi_x, dpi_y = dpi

#     shelf_width_in = round(img_width / dpi_x, 2)
#     shelf_height_in = round(img_height / dpi_y, 2)

#     # -----------------------------
#     # SHOW IMAGE
#     # -----------------------------
#     st.markdown("## 🖼 Original Shelf Image")
#     st.image(image, width=700)

#     # -----------------------------
#     # DETECTION
#     # -----------------------------
#     with st.spinner("🔍 Detecting Products..."):

#         results = model(image, conf=0.35, iou=0.5)

#         boxes_data = []
#         centers_y = []

#         for r in results:
#             for box in r.boxes:

#                 cls_id = int(box.cls[0])
#                 label = model.names[cls_id]

#                 x1, y1, x2, y2 = box.xyxy[0].tolist()

#                 cx = (x1 + x2) / 2
#                 cy = (y1 + y2) / 2

#                 centers_y.append([cy])

#                 boxes_data.append({
#                     "Label": label,
#                     "x1": x1, "y1": y1,
#                     "x2": x2, "y2": y2,
#                     "cx": cx,
#                     "cy": cy,
#                     "Confidence": float(box.conf[0])
#                 })

#         if len(boxes_data) == 0:
#             st.warning("⚠ No products detected")
#             st.stop()

#         # -----------------------------
#         # AUTO ROW DETECTION
#         # -----------------------------
#         k = min(5, len(boxes_data))

#         kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
#         labels = kmeans.fit_predict(np.array(centers_y))

#         for i in range(len(boxes_data)):
#             boxes_data[i]["Row"] = int(labels[i]) + 1

#         df = pd.DataFrame(boxes_data)

#         # -----------------------------
#         # SORT ROWS (TOP → BOTTOM)
#         # -----------------------------
#         row_order = (
#             df.groupby("Row")["cy"]
#             .mean()
#             .sort_values()
#             .index
#         )

#         row_map = {old: new + 1 for new, old in enumerate(row_order)}
#         df["Row"] = df["Row"].map(row_map)

#         # -----------------------------
#         # EYE-CATCHING LOGIC
#         # -----------------------------
#         df["Eye Catching"] = df["Row"].apply(
#             lambda x: "YES" if x <= 2 else "NO"
#         )

#         # -----------------------------
#         # OCCUPIED SPACE CALCULATION
#         # -----------------------------
#         occupied_width_px = df["x2"].max() - df["x1"].min()
#         occupied_height_px = df["y2"].max() - df["y1"].min()

#         occupied_width_in = round(occupied_width_px / dpi_x, 2)
#         occupied_height_in = round(occupied_height_px / dpi_y, 2)

#         width_util = round((occupied_width_in / shelf_width_in) * 100, 2)
#         height_util = round((occupied_height_in / shelf_height_in) * 100, 2)

#         # -----------------------------
#         # IMAGE WITH HIGHLIGHTS
#         # -----------------------------
#         annotated = results[0].plot()
#         annotated_image = Image.fromarray(annotated)
#         draw = ImageDraw.Draw(annotated_image)

#         for _, row in df.iterrows():

#             x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]

#             if row["Eye Catching"] == "YES":
#                 color = "green"
#                 label = "EYE LEVEL"
#             else:
#                 color = "red"
#                 label = "NOT EYE LEVEL"

#             draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
#             draw.text((x1, y1 - 10), label, fill=color)

#         st.markdown("## 🎯 Detection with Shelf Intelligence")
#         st.image(annotated_image, width=900)

#         # -----------------------------
#         # SHELF SIZE & UTILIZATION
#         # -----------------------------
#         st.markdown("## 📏 Shelf Capacity vs Usage")

#         col1, col2, col3 = st.columns(3)

#         with col1:
#             st.metric("Shelf Width (in)", shelf_width_in)
#             st.metric("Shelf Height (in)", shelf_height_in)

#         with col2:
#             st.metric("Occupied Width (in)", occupied_width_in)
#             st.metric("Occupied Height (in)", occupied_height_in)

#         with col3:
#             st.metric("Width Utilization", f"{width_util}%")
#             st.metric("Height Utilization", f"{height_util}%")

#         # -----------------------------
#         # INSIGHT
#         # -----------------------------
#         st.markdown("## 📊 Shelf Insight")

#         if width_util > 80:
#             st.error("⚠ Shelf width heavily crowded")
#         elif width_util > 50:
#             st.warning("Moderate width usage")
#         else:
#             st.success("Good width spacing")

#         if height_util > 80:
#             st.error("⚠ Shelf height over-utilized")
#         elif height_util > 50:
#             st.warning("Moderate height usage")
#         else:
#             st.success("Good vertical spacing")

#         # -----------------------------
#         # ANALYTICS
#         # -----------------------------
#         st.markdown("## 📊 Overview")

#         col1, col2 = st.columns(2)

#         with col1:
#             st.metric("Total Products", len(df))

#         with col2:
#             st.metric("Eye-Level Products", len(df[df["Eye Catching"] == "YES"]))

#         # -----------------------------
#         # ROW ANALYSIS
#         # -----------------------------
#         st.markdown("## 📦 Row Analysis")

#         row_summary = df.groupby("Row").agg(
#             Products=("Label", "count"),
#             Avg_Confidence=("Confidence", "mean")
#         ).reset_index()

#         st.dataframe(row_summary, use_container_width=True)

#         # -----------------------------
#         # EYE / NON EYE TABLES
#         # -----------------------------
#         st.markdown("## 👁 Eye-Catching Products")

#         eye_df = df[df["Eye Catching"] == "YES"]
#         not_eye_df = df[df["Eye Catching"] == "NO"]

#         if len(eye_df) > 0:
#             st.success(f"{len(eye_df)} eye-level products")
#             st.dataframe(eye_df, use_container_width=True)

#         st.markdown("## ⚠ Not Eye-Catching Products")

#         if len(not_eye_df) > 0:
#             st.error(f"{len(not_eye_df)} not in eye-level zone")
#             st.dataframe(not_eye_df, use_container_width=True)
#         else:
#             st.success("All products are eye-level 🎉")

#         # -----------------------------
#         # PRODUCT SUMMARY
#         # -----------------------------
#         st.markdown("## 📦 Product Summary")

#         summary = df.groupby("Label").agg(
#             Count=("Label", "count"),
#             Avg_Confidence=("Confidence", "mean")
#         ).reset_index()

#         summary["Avg_Confidence"] = (summary["Avg_Confidence"] * 100).round(1)

#         st.dataframe(summary, use_container_width=True)

#         # -----------------------------
#         # CHARTS
#         # -----------------------------
#         st.markdown("## 📊 Charts")

#         fig1 = px.pie(summary, names="Label", values="Count", hole=0.4)
#         st.plotly_chart(fig1, use_container_width=True)

#         fig2 = px.bar(summary, x="Label", y="Count", text="Count")
#         st.plotly_chart(fig2, use_container_width=True)

#         # -----------------------------
#         # DOWNLOAD IMAGE
#         # -----------------------------
#         st.markdown("## 📥 Download Result")

#         buf = io.BytesIO()
#         annotated_image.save(buf, format="JPEG")

#         st.download_button(
#             label="📸 Download Image",
#             data=buf.getvalue(),
#             file_name="smart_shelf_result.jpg",
#             mime="image/jpeg"
#         )

# # -----------------------------
# # FOOTER
# # -----------------------------
# st.markdown("---")
# st.caption("Smart Shelf Analytics | YOLO + Auto Row Detection + Shelf Utilization Intelligence")


 
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
model = YOLO("runs/detect/train-22/weights/best.pt")

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
        st.markdown("## 📊 Shelf Insight")

        if width_util > 80:
            st.error("⚠ Shelf width heavily crowded")
        elif width_util > 50:
            st.warning("Moderate width usage")
        else:
            st.success("Good width spacing")

        if height_util > 80:
            st.error("⚠ Shelf height over-utilized")
        elif height_util > 50:
            st.warning("Moderate height usage")
        else:
            st.success("Good vertical spacing")

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



 