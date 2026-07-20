import streamlit as st
from ultralytics import YOLO
from PIL import Image, ImageDraw
import pandas as pd
import io
import plotly.express as px
import numpy as np
from sklearn.cluster import KMeans
import mysql.connector
from mysql.connector import Error
import requests
from datetime import datetime, timedelta
import time
from io import BytesIO
import base64
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
# DATABASE CONNECTION
# -----------------------------
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host='192.168.1.199',
            database='yplmerchandizingappv3',
            user='yplmerchandizing',
            password='v4jaJKsKbYVDUYBJ',
            port='3306'
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# -----------------------------
# FETCH IMAGES FROM DATABASE
# -----------------------------
def fetch_images_from_db(start_date, end_date):
    connection = get_db_connection()
    if connection is None:
        return pd.DataFrame()
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT 
            s2.description,
            s2.address,
            s.photoname,
            s.phototype,
            ad.statusoption,
            s.photoCategory,
            s.latitude,
            s.longitude,
            s.created_at
        FROM shopphotos s 
        INNER JOIN activity_details ad 
            ON s.ActivityDetailId = ad.activitydetailsID 
        INNER JOIN shop s2 
            ON s.shopcode = s2.shopcode 
        WHERE s.created_at > %s 
            AND s.created_at < %s
            AND s.phototype = 'cat_post'
            AND s.photoCategory  = 'Mayonnaise'
        ORDER BY s.created_at DESC
        """
        cursor.execute(query, (start_date, end_date))
        results = cursor.fetchall()
        return pd.DataFrame(results)
    except Error as e:
        st.error(f"Error fetching data: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# -----------------------------
# LOAD MODEL
# -----------------------------
@st.cache_resource
def load_model():
    return YOLO("runs/detect/train-33/weights/best.pt")

model = load_model()

# -----------------------------
# PROCESS SINGLE IMAGE
# -----------------------------
def process_image(image_url, image_data):
    try:
        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        img_width, img_height = image.size
        
        # Get DPI info
        dpi = image.info.get("dpi", (96, 96))
        dpi_x, dpi_y = dpi
        
        shelf_width_in = round(img_width / dpi_x, 2)
        shelf_height_in = round(img_height / dpi_y, 2)
        
        # Detection
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
            return None, None, None, "No products detected"
        
        # Auto row detection
        k = min(5, len(boxes_data))
        kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
        labels = kmeans.fit_predict(np.array(centers_y))
        
        for i in range(len(boxes_data)):
            boxes_data[i]["Row"] = int(labels[i]) + 1
        
        df = pd.DataFrame(boxes_data)
        
        # Sort rows
        row_order = df.groupby("Row")["cy"].mean().sort_values().index
        row_map = {old: new + 1 for new, old in enumerate(row_order)}
        df["Row"] = df["Row"].map(row_map)
        
        # Eye-catching logic
        df["Eye Catching"] = df["Row"].apply(lambda x: "YES" if x <= 2 else "NO")
        
        # Occupied space calculation
        occupied_width_px = df["x2"].max() - df["x1"].min()
        occupied_height_px = df["y2"].max() - df["y1"].min()
        occupied_width_in = round(occupied_width_px / dpi_x, 2)
        occupied_height_in = round(occupied_height_px / dpi_y, 2)
        width_util = round((occupied_width_in / shelf_width_in) * 100, 2)
        height_util = round((occupied_height_in / shelf_height_in) * 100, 2)
        
        # Annotated image
        annotated = results[0].plot()
        annotated_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(annotated_image)
        
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            color = "green" if row["Eye Catching"] == "YES" else "red"
            label_text = "EYE LEVEL" if row["Eye Catching"] == "YES" else "NOT EYE LEVEL"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1, y1 - 10), label_text, fill=color)
        
        # Prepare results
        results_dict = {
            'df': df,
            'annotated_image': annotated_image,
            'shelf_width_in': shelf_width_in,
            'shelf_height_in': shelf_height_in,
            'occupied_width_in': occupied_width_in,
            'occupied_height_in': occupied_height_in,
            'width_util': width_util,
            'height_util': height_util,
            'total_products': len(df),
            'eye_level_products': len(df[df["Eye Catching"] == "YES"])
        }
        
        return results_dict, None, None, "Success"
        
    except Exception as e:
        return None, None, None, f"Error processing image: {str(e)}"

# -----------------------------
# CREATE IMAGE URL
# -----------------------------
def get_image_url(photoname):
    # base_url = "https://apps.youngsfood.com/yplrmapp/public/photouploads/"
    base_url = "http://192.168.1.199/yplrmapp/public/photouploads/"
    # Extract date from photoname if available
    # photoname format: 2026/06/061420261345060005700043post_category.jpg
    return f"{base_url}{photoname}"

# -----------------------------
# DISPLAY IMAGE WITH METADATA
# -----------------------------
def display_image_with_metadata(image_url, metadata, col):
    with col:
        st.image(image_url, use_column_width=True)
        st.caption(f"📸 {metadata.get('photoname', 'N/A')}")
        st.markdown(f"""
        **Shop:** {metadata.get('description', 'N/A')}  
        **Address:** {metadata.get('address', 'N/A')}  
        **Status:** {metadata.get('statusoption', 'N/A')}  
        **Category:** {metadata.get('photoCategory', 'N/A')}  
        **Date:** {metadata.get('created_at', 'N/A')}
        """)
        if metadata.get('latitude') and metadata.get('longitude'):
            st.caption(f"📍 {metadata['latitude']}, {metadata['longitude']}")

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🛒 Smart Shelf Product Analytics")
# st.caption("AI-powered Retail Intelligence using YOLO + Shelf Analytics")

# -----------------------------
# DATE SELECTION
# -----------------------------
st.sidebar.header("📅 Date Selection")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.strptime("2026-06-14", "%Y-%m-%d").date()
    )
with col2:
    end_date = st.sidebar.date_input(
        "End Date",
        value=datetime.strptime("2026-07-14", "%Y-%m-%d").date()
    )

start_datetime = datetime.combine(start_date, datetime.min.time())
end_datetime = datetime.combine(end_date, datetime.max.time())

# Format for query
start_str = start_datetime.strftime("%Y-%m-%d %H:%M:%S")
end_str = end_datetime.strftime("%Y-%m-%d %H:%M:%S")

# -----------------------------
# FETCH AND PROCESS BUTTONS
# -----------------------------
if st.sidebar.button("📥 Fetch Images from Database"):
    with st.spinner("Fetching images from database..."):
        df_images = fetch_images_from_db(start_str, end_str)
        
        if df_images.empty:
            st.warning("No images found for the selected date range.")
        else:
            st.session_state['df_images'] = df_images
            st.session_state['current_batch'] = 0
            st.session_state['batch_size'] = 20
            st.success(f"✅ Found {len(df_images)} images!")

# -----------------------------
# BATCH PROCESSING
# -----------------------------
if 'df_images' in st.session_state and not st.session_state['df_images'].empty:
    df_images = st.session_state['df_images']
    batch_size = st.session_state.get('batch_size', 20)
    current_batch = st.session_state.get('current_batch', 0)
    
    total_images = len(df_images)
    total_batches = (total_images + batch_size - 1) // batch_size
    
    # Batch navigation
    st.sidebar.markdown("---")
    st.sidebar.subheader("📑 Batch Navigation")
    st.sidebar.write(f"Batch {current_batch + 1} of {total_batches}")
    st.sidebar.write(f"Images {current_batch * batch_size + 1} - {min((current_batch + 1) * batch_size, total_images)} of {total_images}")
    
    col1, col2, col3 = st.sidebar.columns(3)
    with col1:
        if st.button("◀ Previous") and current_batch > 0:
            st.session_state['current_batch'] = current_batch - 1
            st.rerun()
    with col2:
        st.write("")
    with col3:
        if st.button("Next ▶") and current_batch < total_batches - 1:
            st.session_state['current_batch'] = current_batch + 1
            st.rerun()
    
    # Get current batch
    start_idx = current_batch * batch_size
    end_idx = min((current_batch + 1) * batch_size, total_images)
    batch_df = df_images.iloc[start_idx:end_idx]
    
    # Display batch info
    st.markdown(f"## 📸 Processing Batch {current_batch + 1} of {total_batches}")
    st.info(f"Processing images {start_idx + 1} to {end_idx} of {total_images}")
    
    # Process button for batch
    if st.button(f"🚀 Process Batch {current_batch + 1}", type="primary"):
        st.session_state['batch_results'] = {}
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            photo_name = row['photoname']
            image_url = get_image_url(photo_name)
            
            status_text.text(f"Processing image {idx + 1}/{len(batch_df)}: {photo_name}")
            
            # Process image
            results_dict, _, _, message = process_image(image_url, row.to_dict())
            
            if results_dict:
                st.session_state['batch_results'][photo_name] = {
                    'results': results_dict,
                    'metadata': row.to_dict(),
                    'success': True
                }
            else:
                st.session_state['batch_results'][photo_name] = {
                    'error': message,
                    'metadata': row.to_dict(),
                    'success': False
                }
            
            progress_bar.progress((idx + 1) / len(batch_df))
        
        status_text.text("✅ Batch processing complete!")
        st.rerun()
    
    # Display results
    if 'batch_results' in st.session_state:
        st.markdown("## 📊 Batch Results")
        
        # Create tabs for each image in batch
        tabs = st.tabs([f"Image {i+1}" for i in range(len(batch_df))])
        
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            photo_name = row['photoname']
            
            with tabs[idx]:
                if photo_name in st.session_state['batch_results']:
                    result = st.session_state['batch_results'][photo_name]
                    
                    if result['success']:
                        # Display metadata
                        metadata = result['metadata']
                        st.markdown(f"### 🏪 {metadata.get('description', 'N/A')}")
                        st.markdown(f"**Address:** {metadata.get('address', 'N/A')}")
                        st.markdown(f"**Status:** {metadata.get('statusoption', 'N/A')}")
                        st.markdown(f"**Category:** {metadata.get('photoCategory', 'N/A')}")
                        st.markdown(f"**Date:** {metadata.get('created_at', 'N/A')}")
                        
                        if metadata.get('latitude') and metadata.get('longitude'):
                            st.markdown(f"**Location:** {metadata['latitude']}, {metadata['longitude']}")
                        
                        st.markdown("---")
                        
                        # Display results
                        results_dict = result['results']
                        
                        # Show annotated image
                        st.image(results_dict['annotated_image'], use_column_width=True)
                        
                        # Metrics
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Products", results_dict['total_products'])
                        with col2:
                            st.metric("Eye-Level Products", results_dict['eye_level_products'])
                        with col3:
                            st.metric("Width Utilization", f"{results_dict['width_util']}%")
                        
                        # Show data
                        with st.expander("📊 View Detailed Data"):
                            st.dataframe(results_dict['df'], use_container_width=True)
                            
                            # Charts
                            chart_df = results_dict['df'].groupby("Label").agg(
                                Count=("Label", "count")
                            ).reset_index()
                            
                            if not chart_df.empty:
                                fig = px.pie(chart_df, names="Label", values="Count", hole=0.4)
                                st.plotly_chart(fig, use_container_width=True)
                        
                        # Download button
                        buf = io.BytesIO()
                        results_dict['annotated_image'].save(buf, format="JPEG")
                        st.download_button(
                            label=f"📥 Download Result - {photo_name}",
                            data=buf.getvalue(),
                            file_name=f"result_{photo_name.replace('/', '_')}.jpg",
                            mime="image/jpeg"
                        )
                        
                    else:
                        st.error(f"❌ Failed to process image: {result.get('error', 'Unknown error')}")
                        with st.expander("📋 Metadata"):
                            st.json(result['metadata'])
                else:
                    st.info("Click 'Process Batch' to analyze this image")

# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Smart Shelf Analytics | YOLO + Auto Row Detection")

st.markdown("---")
st.caption("Smart Shelf Analytics | YOLO + Auto Row Detection + Shelf Utilization Intelligence")