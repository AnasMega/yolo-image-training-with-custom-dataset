 

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
import hashlib

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Shelf Analytics",
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
def process_image(image_url, image_data, eye_level_min_ft=1.0, eye_level_max_ft=8.0):
    try:
        # Download image from URL
        response = requests.get(image_url, timeout=10)
        response.raise_for_status()
        image = Image.open(BytesIO(response.content)).convert("RGB")
        img_width, img_height = image.size
        
        # Get DPI info
        dpi = image.info.get("dpi", (96, 96))
        dpi_x, dpi_y = dpi
        
        # Calculate shelf dimensions in inches
        shelf_width_in = round(img_width / dpi_x, 2)
        shelf_height_in = round(img_height / dpi_y, 2)
        
        # Convert feet to inches for eye level (1 ft = 12 inches)
        eye_level_min_in = eye_level_min_ft * 12
        eye_level_max_in = eye_level_max_ft * 12
        
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
                
                # Calculate individual product width and height in inches
                product_width_px = x2 - x1
                product_height_px = y2 - y1
                product_width_in = round(product_width_px / dpi_x, 2)
                product_height_in = round(product_height_px / dpi_y, 2)
                
                # Calculate product position from bottom (in inches)
                height_per_pixel = shelf_height_in / img_height
                product_height_from_bottom_in = round((img_height - cy) * height_per_pixel, 2)
                product_bottom_from_bottom_in = round((img_height - y2) * height_per_pixel, 2)
                product_top_from_bottom_in = round((img_height - y1) * height_per_pixel, 2)
                
                boxes_data.append({
                    "Label": label,
                    "x1": x1, "y1": y1,
                    "x2": x2, "y2": y2,
                    "cx": cx,
                    "cy": cy,
                    "Confidence": float(box.conf[0]),
                    "Product Width (in)": product_width_in,
                    "Product Height (in)": product_height_in,
                    "Height from Bottom (in)": product_height_from_bottom_in,
                    "Bottom Edge (in)": product_bottom_from_bottom_in,
                    "Top Edge (in)": product_top_from_bottom_in
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
        
        # Eye-catching logic based on height in inches (converted from feet)
        df["Eye Catching"] = df["Height from Bottom (in)"].apply(
            lambda x: "YES" if eye_level_min_in <= x <= eye_level_max_in else "NO"
        )
        
        # Occupied space calculation
        occupied_width_px = df["x2"].max() - df["x1"].min()
        occupied_height_px = df["y2"].max() - df["y1"].min()
        occupied_width_in = round(occupied_width_px / dpi_x, 2)
        occupied_height_in = round(occupied_height_px / dpi_y, 2)
        width_util = round((occupied_width_in / shelf_width_in) * 100, 2)
        height_util = round((occupied_height_in / shelf_height_in) * 100, 2)
        
        # Calculate empty space
        empty_width_in = round(shelf_width_in - occupied_width_in, 2)
        empty_height_in = round(shelf_height_in - occupied_height_in, 2)
        empty_width_percent = round(100 - width_util, 2)
        empty_height_percent = round(100 - height_util, 2)
        
        # Row-wise occupancy analysis
        row_analysis = []
        for row_num in sorted(df["Row"].unique()):
            row_df = df[df["Row"] == row_num]
            row_min_x = row_df["x1"].min()
            row_max_x = row_df["x2"].max()
            row_width_px = row_max_x - row_min_x
            row_width_in = round(row_width_px / dpi_x, 2)
            row_products = len(row_df)
            
            # Check if row has any eye level products
            row_eye_level = "YES" if any(row_df["Eye Catching"] == "YES") else "NO"
            
            # Calculate average height of products in this row
            avg_height_in = round(row_df["Height from Bottom (in)"].mean(), 2)
            avg_height_ft = round(avg_height_in / 12, 2)
            
            row_analysis.append({
                "Row": row_num,
                "Products": row_products,
                "Width Occupied (in)": row_width_in,
                "Eye Level": row_eye_level,
                "Avg Height (in)": avg_height_in,
                "Avg Height (ft)": avg_height_ft
            })
        
        row_analysis_df = pd.DataFrame(row_analysis)
        
        # Annotated image with shelf boundaries
        annotated = results[0].plot()
        annotated_image = Image.fromarray(annotated)
        draw = ImageDraw.Draw(annotated_image)
        
        # Draw shelf boundary
        draw.rectangle([0, 0, img_width-1, img_height-1], outline="blue", width=3)
        
        # Draw occupied area boundary
        draw.rectangle(
            [df["x1"].min(), df["y1"].min(), df["x2"].max(), df["y2"].max()], 
            outline="orange", 
            width=3
        )
        
        # Add labels for each product
        for _, row in df.iterrows():
            x1, y1, x2, y2 = row["x1"], row["y1"], row["x2"], row["y2"]
            color = "green" if row["Eye Catching"] == "YES" else "red"
            label_text = "EYE LEVEL" if row["Eye Catching"] == "YES" else f"{row['Height from Bottom (in)']}in"
            draw.rectangle([x1, y1, x2, y2], outline=color, width=4)
            draw.text((x1, y1 - 10), label_text, fill=color)
            
            # Add product dimensions on the box
            dim_text = f"{row['Product Width (in)']}x{row['Product Height (in)']}in"
            draw.text((x1, y2 + 5), dim_text, fill="white")
        
        # Add shelf dimensions annotation in inches
        draw.text((10, 10), f"Shelf: {shelf_width_in} x {shelf_height_in} in", fill="blue")
        draw.text((10, 30), f"Occupied: {occupied_width_in} x {occupied_height_in} in", fill="orange")
        draw.text((10, 50), f"Empty: {empty_width_in} x {empty_height_in} in", fill="red")
        draw.text((10, 70), f"Eye Level: {eye_level_min_ft} - {eye_level_max_ft} ft ({eye_level_min_in:.0f} - {eye_level_max_in:.0f} in)", fill="green")
        
        # Prepare results
        results_dict = {
            'df': df,
            'annotated_image': annotated_image,
            'shelf_width_in': shelf_width_in,
            'shelf_height_in': shelf_height_in,
            'occupied_width_in': occupied_width_in,
            'occupied_height_in': occupied_height_in,
            'empty_width_in': empty_width_in,
            'empty_height_in': empty_height_in,
            'width_util': width_util,
            'height_util': height_util,
            'empty_width_percent': empty_width_percent,
            'empty_height_percent': empty_height_percent,
            'total_products': len(df),
            'eye_level_products': len(df[df["Eye Catching"] == "YES"]),
            'non_eye_level_products': len(df[df["Eye Catching"] == "NO"]),
            'row_analysis_df': row_analysis_df,
            'dpi_x': dpi_x,
            'dpi_y': dpi_y,
            'eye_level_min_ft': eye_level_min_ft,
            'eye_level_max_ft': eye_level_max_ft,
            'eye_level_min_in': eye_level_min_in,
            'eye_level_max_in': eye_level_max_in
        }
        
        return results_dict, None, None, "Success"
        
    except Exception as e:
        return None, None, None, f"Error processing image: {str(e)}"

# -----------------------------
# CREATE IMAGE URL
# -----------------------------
def get_image_url(photoname):
    base_url = "http://192.168.1.199/yplrmapp/public/photouploads/"
    return f"{base_url}{photoname}"

# -----------------------------
# MAIN UI
# -----------------------------
st.title("Shelf Product Utilization")

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
# Eye Level Configuration (Fixed range)
eye_level_min_ft = 1.0
eye_level_max_ft = 8.0

# Date Selection
st.sidebar.header("📅 Select Date")
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.strptime("2026-06-14", "%Y-%m-%d").date()
    )
with col2:
    end_date = st.date_input(
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
    # st.sidebar.subheader("📑 Batch Navigation")
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
        st.session_state['failed_images'] = []
        st.session_state['successful_images'] = []
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for idx, (_, row) in enumerate(batch_df.iterrows()):
            photo_name = row['photoname']
            image_url = get_image_url(photo_name)
            
            status_text.text(f"Processing image {idx + 1}/{len(batch_df)}: {photo_name}")
            
            # Process image with eye level range in feet
            results_dict, _, _, message = process_image(
                image_url, 
                row.to_dict(), 
                eye_level_min_ft, 
                eye_level_max_ft
            )
            
            if results_dict:
                st.session_state['batch_results'][photo_name] = {
                    'results': results_dict,
                    'metadata': row.to_dict(),
                    'success': True
                }
                st.session_state['successful_images'].append(photo_name)
            else:
                st.session_state['batch_results'][photo_name] = {
                    'error': message,
                    'metadata': row.to_dict(),
                    'success': False
                }
                st.session_state['failed_images'].append(photo_name)
            
            progress_bar.progress((idx + 1) / len(batch_df))
        
        # Show summary
        status_text.text("✅ Batch processing complete!")
        st.success(f"✅ Processed {len(st.session_state['successful_images'])} images successfully")
        if len(st.session_state['failed_images']) > 0:
            st.warning(f"⚠️ {len(st.session_state['failed_images'])} images failed (no products detected)")
        
        st.rerun()
    
    # Display results - Only show successful images
    if 'batch_results' in st.session_state and st.session_state['successful_images']:
        st.markdown("## 📊 Results")
        
        # Create tabs only for successful images
        successful_df = batch_df[batch_df['photoname'].isin(st.session_state['successful_images'])]
        
        if len(successful_df) > 0:
            tabs = st.tabs([f"Image {i+1}" for i in range(len(successful_df))])
            
            for idx, (_, row) in enumerate(successful_df.iterrows()):
                photo_name = row['photoname']
                # Create a unique ID for this image
                image_id = hashlib.md5(photo_name.encode()).hexdigest()[:8]
                
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
                            
                            st.markdown("---")
                            
                            # Display results
                            results_dict = result['results']
                            
                            # Show annotated image
                            st.image(results_dict['annotated_image'])
                            
                            # SHELF INSIGHTS SECTION - ALL IN INCHES
                            st.markdown("##  Shelf Space Insights")
                            
                            # Create a clean layout for shelf metrics
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.markdown("###  Shelf Dimensions")
                                st.metric("Total Shelf Width", f"{results_dict['shelf_width_in']} in")
                                st.metric("Total Shelf Height", f"{results_dict['shelf_height_in']} in")
                            
                            with col2:
                                st.markdown("###  Occupied Space")
                                st.metric("Occupied Width", f"{results_dict['occupied_width_in']} in")
                                st.metric("Occupied Height", f"{results_dict['occupied_height_in']} in")
                            
                            with col3:
                                st.markdown("###  Utilization")
                                st.metric(
                                    "Width Utilization", 
                                    f"{results_dict['width_util']}%",
                                    delta_color="normal"
                                )
                                st.metric(
                                    "Height Utilization", 
                                    f"{results_dict['height_util']}%",
                                    delta_color="normal"
                                )
                            
                            # Add visual progress bars for utilization
                            # st.markdown("### 📊 Space Utilization Visualization")
                            
                            util_col1, util_col2 = st.columns(2)
                            
                            
                            
                            summary_col1, summary_col2, summary_col3, summary_col4 = st.columns(4)
                            
                            
                            # Show data
                            with st.expander("📊 View Detailed Product Data"):
                                display_df = results_dict['df'][['Label', 'Row', 'Eye Catching', 
                                                                'Product Width (in)', 'Product Height (in)', 
                                                                'Height from Bottom (in)',
                                                                'Confidence']]
                                st.dataframe(display_df, use_container_width=True)
                                
                                # Charts with unique keys
                                chart_df = results_dict['df'].groupby("Label").agg(
                                    Count=("Label", "count"),
                                    Avg_Width=("Product Width (in)", "mean"),
                                    Avg_Height=("Product Height (in)", "mean"),
                                    Avg_Height_From_Bottom=("Height from Bottom (in)", "mean")
                                ).reset_index()
                                
                                if not chart_df.empty:
                                    # Product distribution pie chart
                                    fig_pie = px.pie(
                                        chart_df, 
                                        names="Label", 
                                        values="Count", 
                                        hole=0.4,
                                        title="Product Distribution"
                                    )
                                    st.plotly_chart(
                                        fig_pie, 
                                        use_container_width=True,
                                        key=f"pie_chart_{image_id}_{idx}"
                                    )
                                    
                                    # Eye level vs non-eye level chart
                                    eye_level_counts = results_dict['df']['Eye Catching'].value_counts().reset_index()
                                    eye_level_counts.columns = ['Eye Level', 'Count']
                                    
                                    fig_eye = px.bar(
                                        eye_level_counts,
                                        x='Eye Level',
                                        y='Count',
                                        color='Eye Level',
                                        color_discrete_map={'YES': 'green', 'NO': 'red'},
                                        title="Eye Level vs Non-Eye Level Products"
                                    )
                                    st.plotly_chart(
                                        fig_eye,
                                        use_container_width=True,
                                        key=f"eye_chart_{image_id}_{idx}"
                                    )
                                    
                                    # Height distribution chart
                                    fig_height = px.histogram(
                                        results_dict['df'],
                                        x="Height from Bottom (in)",
                                        color="Eye Catching",
                                        color_discrete_map={'YES': 'green', 'NO': 'red'},
                                        title="Product Height Distribution",
                                        labels={"Height from Bottom (in)": "Height from Bottom (inches)"},
                                        nbins=20
                                    )
                                    # Add vertical lines for eye level range
                                    fig_height.add_vline(x=results_dict['eye_level_min_in'], line_dash="dash", line_color="green", annotation_text="Min Eye Level")
                                    fig_height.add_vline(x=results_dict['eye_level_max_in'], line_dash="dash", line_color="green", annotation_text="Max Eye Level")
                                    st.plotly_chart(
                                        fig_height,
                                        use_container_width=True,
                                        key=f"height_chart_{image_id}_{idx}"
                                    )
                            
                            # Download button
                            buf = io.BytesIO()
                            results_dict['annotated_image'].save(buf, format="JPEG")
                            st.download_button(
                                label=f"📥 Download Result - {photo_name}",
                                data=buf.getvalue(),
                                file_name=f"result_{photo_name.replace('/', '_')}.jpg",
                                mime="image/jpeg",
                                key=f"download_{image_id}_{idx}"
                            )
        else:
            st.warning("No successful images to display in this batch.")
    
# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Shelf Analytics")

st.markdown("---")
st.caption("Shelf Analytics | Shelf Utilization")