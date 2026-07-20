
 
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
from dotenv import load_dotenv


# Load environment variables from .env file
load_dotenv()

db_host = os.environ.get("DB_HOST")
db_name = os.environ.get("DB_NAME")
db_user = os.environ.get("DB_USER")
db_password = os.environ.get("DB_PASSWORD")
db_port = os.environ.get("DB_PORT")

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(
    page_title="Shelf Analytics",
    page_icon="🛒",
    layout="wide"
)
# -----------------------------
# CUSTOM STYLES FOR BETTER DISPLAY
# -----------------------------
st.markdown("""
<style>
    /* Improve image container spacing */
    .stImage {
        margin: 0 !important;
        padding: 0 !important;
    }
    /* Make column spacing more compact */
    .stColumn {
        padding: 0 4px !important;
    }
    /* Ensure consistent card heights */
    .element-container {
        height: 100% !important;
    }
    /* Better scrollbar for long content */
    .stApp {
        scrollbar-width: thin;
    }
</style>
""", unsafe_allow_html=True)
# -----------------------------
# DATABASE CONNECTION
# -----------------------------
def get_db_connection():
    try:
        connection = mysql.connector.connect(
            host=db_host,
            database=db_name,
            user=db_user,
            password=db_password,
            port=db_port
        )
        return connection
    except Error as e:
        st.error(f"Error connecting to MySQL: {e}")
        return None

# -----------------------------
# FETCH SHOPS FROM DATABASE
# -----------------------------
def fetch_shops():
    connection = get_db_connection()
    if connection is None:
        return pd.DataFrame()
    
    try:
        cursor = connection.cursor(dictionary=True)
        query = """
        SELECT shopcode, description 
        FROM shop 
        ORDER BY description
        """
        cursor.execute(query)
        results = cursor.fetchall()
        return pd.DataFrame(results)
    except Error as e:
        st.error(f"Error fetching shops: {e}")
        return pd.DataFrame()
    finally:
        if connection.is_connected():
            cursor.close()
            connection.close()

# -----------------------------
# FETCH IMAGES FROM DATABASE
# -----------------------------
def fetch_images_from_db(start_date, end_date, selected_shops=None):
    connection = get_db_connection()
    if connection is None:
        return pd.DataFrame()
    
    try:
        cursor = connection.cursor(dictionary=True)
        
        # Base query
        query = """
        SELECT 
            s2.description,
            s2.address,
            s2.shopcode,
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
        """
        
        # Add shop filter if specific shops are selected
        params = [start_date, end_date]
        if selected_shops and len(selected_shops) > 0:
            # Create placeholders for shop codes
            placeholders = ','.join(['%s'] * len(selected_shops))
            query += f" AND s.shopcode IN ({placeholders})"
            params.extend(selected_shops)
        
        query += " ORDER BY s.created_at DESC"
        
        cursor.execute(query, params)
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
    return YOLO("runs/detect/train-37/weights/best.pt")

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
        
        # ✅ Check if at least 2 products are detected
        if len(boxes_data) < 2:
            return None, None, None, "Less than 2 products detected (skipping)"
        
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
        
        # ✅ Calculate product counts by label
        product_counts = df['Label'].value_counts().to_dict()
        total_products = len(df)
        
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
            'total_rows': len(df["Row"].unique()),
            'dpi_x': dpi_x,
            'dpi_y': dpi_y,
            'eye_level_min_ft': eye_level_min_ft,
            'eye_level_max_ft': eye_level_max_ft,
            'eye_level_min_in': eye_level_min_in,
            'eye_level_max_in': eye_level_max_in,
            'product_counts': product_counts  # ✅ NEW: Add product counts
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
# DISPLAY IMAGE WITH INSIGHTS - GRID VIEW
# -----------------------------
def display_image_grid(images_data, cols_per_row=3):
    """Display images in a grid with insights below each image"""
    
    if not images_data:
        st.warning("No images to display")
        return
    
    # Create rows
    rows = [images_data[i:i + cols_per_row] for i in range(0, len(images_data), cols_per_row)]
    
    for row_idx, row_images in enumerate(rows):
        # Create columns for this row with equal spacing
        cols = st.columns(cols_per_row, gap="large")
        
        for col_idx, (col, img_data) in enumerate(zip(cols, row_images)):
            with col:
                # Get image data
                photo_name = img_data['photo_name']
                results_dict = img_data['results']
                metadata = img_data['metadata']
                
                # Create a container with border for each image card
                with st.container():
                    # Display image with proper sizing
                    st.image(results_dict['annotated_image'], use_container_width=True)
                    
                    # Display insights below image with proper spacing
                    st.markdown("---")
                    
                    # Shop info with small text
                    st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b> {metadata.get('description', 'N/A')}</b></p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:11px; margin:2px 0;'> {metadata.get('address', 'N/A')}</p>", unsafe_allow_html=True)
                    st.markdown(f"<p style='font-size:11px; margin:2px 0;'> {metadata.get('created_at', 'N/A')}</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # ✅ NEW: Product Detection Summary
                    # st.markdown("<p style='font-size:13px; font-weight:bold; margin:4px 0;'>📦 Products Detected</p>", unsafe_allow_html=True)
                    
                    # Display product counts
                    # product_counts = results_dict.get('product_counts', {})
                    # if product_counts:
                    #     # Create a compact display of products
                    #     product_text = ""
                    #     for product_name, count in product_counts.items():
                    #         product_text += f"<span style='font-size:12px; background:#f0f0f0; padding:2px 8px; border-radius:10px; margin:2px; display:inline-block;'>{product_name}: {count}</span>"
                    #     st.markdown(f"<div style='margin:4px 0;'>{product_text}</div>", unsafe_allow_html=True)
                    # else:
                    #     st.markdown("<p style='font-size:11px; margin:2px 0;'>No products detected</p>", unsafe_allow_html=True)
                    
                    # # Total products count
                    # st.markdown(f"<p style='font-size:12px; margin:2px 0;'><b>Total Products:</b> {results_dict['total_products']}</p>", unsafe_allow_html=True)
                    
                    # st.markdown("---")
                    
                    # Shelf Dimensions Section
                    st.markdown("<p style='font-size:13px; font-weight:bold; margin:1px 0;'> Shelf Dimensions</p>", unsafe_allow_html=True)
                    
                    # Width metrics
                    col_w1, col_w2, col_w3 = st.columns(3)
                    with col_w1:
                        st.markdown(f"<p style='font-size:11px; margin:2px 0;'>Total Width</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:14px; font-weight:bold; margin:2px 0;'>{results_dict['shelf_width_in']}\"</p>", unsafe_allow_html=True)
                    with col_w2:
                        st.markdown(f"<p style='font-size:11px; margin:2px 0;'>Occupied</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:14px; font-weight:bold; margin:2px 0;'>{results_dict['occupied_width_in']}\"</p>", unsafe_allow_html=True)
                    with col_w3:
                        st.markdown(f"<p style='font-size:11px; margin:2px 0;'>Utilization</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:14px; font-weifght:bold; color:#0066cc; margin:2px 0;'>{results_dict['width_util']}%</p>", unsafe_allow_html=True)
                    
                    # Height metrics
                    col_h1, col_h2, col_h3 = st.columns(3)
                    with col_h1:
                        st.markdown(f"<p style='font-size:11px; margin:2px 0;'>Total Height</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:14px; font-weight:bold; margin:2px 0;'>{results_dict['shelf_height_in']}\"</p>", unsafe_allow_html=True)
                    with col_h2:
                        st.markdown(f"<p style='font-size:11px; margin:2px 0;'>Occupied</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:14px; font-weight:bold; margin:2px 0;'>{results_dict['occupied_height_in']}\"</p>", unsafe_allow_html=True)
                    with col_h3:
                        st.markdown(f"<p style='font-size:11px; margin:2px 0;'>Utilization</p>", unsafe_allow_html=True)
                        st.markdown(f"<p style='font-size:14px; font-weight:bold; color:#0066cc; margin:2px 0;'>{results_dict['height_util']}%</p>", unsafe_allow_html=True)
                    
                    st.markdown("---")

# -----------------------------
# PROCESS BATCH OF IMAGES
# -----------------------------
def process_batch_images(batch_df, eye_level_min_ft, eye_level_max_ft, batch_size=30):
    """Process a batch of images"""
    
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    batch_results = {}
    processed = []
    failed = []
    skipped = []  # ✅ New: Track skipped images (less than 2 products)
    
    total_images = len(batch_df)
    for idx, (_, row) in enumerate(batch_df.iterrows()):
        photo_name = row['photoname']
        image_url = get_image_url(photo_name)
        
        status_text.text(f"Processing {idx + 1}/{total_images}: {photo_name}")
        
        # Process image
        results_dict, _, _, message = process_image(
            image_url, 
            row.to_dict(), 
            eye_level_min_ft, 
            eye_level_max_ft
        )
        
        if results_dict:
            batch_results[photo_name] = {
                'results': results_dict,
                'metadata': row.to_dict(),
                'success': True
            }
            processed.append(photo_name)
        else:
            batch_results[photo_name] = {
                'error': message,
                'metadata': row.to_dict(),
                'success': False
            }
            # Check if it was skipped due to less than 2 products
            if "Less than 2 products detected" in message:
                skipped.append(photo_name)
            else:
                failed.append(photo_name)
        
        progress_bar.progress((idx + 1) / total_images)
    
    status_text.text("✅ Batch processing complete!")
    return batch_results, processed, failed, skipped

# -----------------------------
# MAIN UI
# -----------------------------
st.title("🛒 Shelf Product Utilization")

# Initialize session state
if 'current_batch' not in st.session_state:
    st.session_state['current_batch'] = 0
if 'all_results' not in st.session_state:
    st.session_state['all_results'] = {}
if 'processed_images' not in st.session_state:
    st.session_state['processed_images'] = []
if 'failed_images' not in st.session_state:
    st.session_state['failed_images'] = []
if 'skipped_images' not in st.session_state:
    st.session_state['skipped_images'] = []  # ✅ New: Track skipped images
if 'displayed_batches' not in st.session_state:
    st.session_state['displayed_batches'] = 1
if 'shops_df' not in st.session_state:
    st.session_state['shops_df'] = fetch_shops()

# -----------------------------
# SIDEBAR CONTROLS
# -----------------------------
# Eye Level Configuration (Fixed range)
eye_level_min_ft = 1.0
eye_level_max_ft = 8.0

# Shop Selection
st.sidebar.header(" Select Shops")
shops_df = st.session_state['shops_df']

if not shops_df.empty:
    # Create shop options with description
    shop_options = shops_df['description'].tolist()
    shop_codes = shops_df['shopcode'].tolist()
    
    # Add "Select All" option
    select_all = st.sidebar.checkbox("Select All Shops", value=True)
    
    if select_all:
        selected_shops = shop_codes  # All shops
        selected_descriptions = shop_options
    else:
        selected_descriptions = st.sidebar.multiselect(
            "Choose Shops",
            options=shop_options,
            default=[]
        )
        # Get shop codes for selected descriptions
        selected_shops = shops_df[shops_df['description'].isin(selected_descriptions)]['shopcode'].tolist()
    
    st.sidebar.write(f"Selected: {len(selected_shops)} shop(s)")
else:
    st.sidebar.warning("No shops found in database")
    selected_shops = []

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

# Batch size configuration
st.sidebar.markdown("---")
st.sidebar.header("⚙️ Configuration")
batch_size = st.sidebar.number_input(
    "Batch Size (images per batch)",
    min_value=10,
    max_value=300,
    value=30,
    step=5
)

# -----------------------------
# FETCH AND PROCESS BUTTONS
# -----------------------------
if st.sidebar.button("📥 Fetch Images from Database"):
    with st.spinner("Fetching images from database..."):
        df_images = fetch_images_from_db(start_str, end_str, selected_shops)
        
        if df_images.empty:
            st.warning("No images found for the selected date range and shops.")
        else:
            st.session_state['df_images'] = df_images
            st.session_state['all_results'] = {}
            st.session_state['processed_images'] = []
            st.session_state['failed_images'] = []
            st.session_state['skipped_images'] = []  # ✅ Reset skipped
            st.session_state['current_batch'] = 0
            st.session_state['displayed_batches'] = 1
            st.success(f"✅ Found {len(df_images)} images from {len(selected_shops)} shop(s)!")

# -----------------------------
# PROCESS AND DISPLAY ALL IMAGES IN BATCHES
# -----------------------------
if 'df_images' in st.session_state and not st.session_state['df_images'].empty:
    df_images = st.session_state['df_images']
    total_images = len(df_images)
    total_batches = (total_images + batch_size - 1) // batch_size
    
    # Show batch info
    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Batch Status")
    st.sidebar.write(f"Total Images: {total_images}")
    st.sidebar.write(f"Total Batches: {total_batches}")
    st.sidebar.write(f"Batch Size: {batch_size}")
    
    # Show selected shops info
    if selected_shops:
        st.sidebar.write(f"Selected Shops: {len(selected_shops)}")
    
    # Process next batch button
    current_batch = st.session_state['current_batch']
    
    if current_batch < total_batches:
        if st.button(f"🚀 Process Batch {current_batch + 1} of {total_batches}", type="primary"):
            # Get current batch
            start_idx = current_batch * batch_size
            end_idx = min((current_batch + 1) * batch_size, total_images)
            batch_df = df_images.iloc[start_idx:end_idx]
            
            with st.spinner(f"Processing batch {current_batch + 1}..."):
                batch_results, processed, failed, skipped = process_batch_images(
                    batch_df, 
                    eye_level_min_ft, 
                    eye_level_max_ft,
                    batch_size
                )
                
                # Update session state
                st.session_state['all_results'].update(batch_results)
                st.session_state['processed_images'].extend(processed)
                st.session_state['failed_images'].extend(failed)
                st.session_state['skipped_images'].extend(skipped)  # ✅ Track skipped
                st.session_state['current_batch'] += 1
                
                st.success(f"✅ Batch {current_batch + 1} processed! ({len(processed)} successful, {len(failed)} failed, {len(skipped)} skipped - less than 2 products)")
                st.rerun()
    
    # Check if all batches are processed
    if st.session_state['current_batch'] >= total_batches:
        st.success("🎉 All batches processed!")
    
    # Display results - Show all processed images (only those with >= 2 products)
    if st.session_state['processed_images']:
        total_processed = len(st.session_state['processed_images'])
        
        # Info about total processed
        # st.markdown("## 📊 Results Grid View")
        st.info(f"📸 Showing {total_processed} images with 2+ products detected (from {st.session_state['current_batch']} batch(es))")
        
        # Prepare data for grid display (all processed images)
        images_data = []
        for photo_name in st.session_state['processed_images']:
            if photo_name in st.session_state['all_results']:
                result = st.session_state['all_results'][photo_name]
                if result['success']:
                    images_data.append({
                        'photo_name': photo_name,
                        'results': result['results'],
                        'metadata': result['metadata']
                    })
        
        # Display in grid - 3 images per row with better spacing
        if images_data:
            display_image_grid(images_data, cols_per_row=3)
        else:
            st.warning("No successful images to display")
    
    # Show skipped images summary (less than 2 products)
    if st.session_state.get('skipped_images', []):
        with st.expander(f"⏭️ Skipped Images ({len(st.session_state['skipped_images'])}) - Less than 2 products detected"):
            for photo_name in st.session_state['skipped_images']:
                if photo_name in st.session_state['all_results']:
                    st.write(f"⏭️ {photo_name}: {st.session_state['all_results'][photo_name].get('error', 'Less than 2 products detected')}")
    
    # Show failed images summary (actual errors)
    if st.session_state['failed_images']:
        with st.expander(f"❌ Failed Images ({len(st.session_state['failed_images'])})"):
            for photo_name in st.session_state['failed_images']:
                if photo_name in st.session_state['all_results']:
                    st.write(f"❌ {photo_name}: {st.session_state['all_results'][photo_name].get('error', 'Unknown error')}")
    
    # Progress summary
    if st.session_state['current_batch'] > 0:
        total_processed = len(st.session_state['processed_images'])
        total_failed = len(st.session_state['failed_images'])
        total_skipped = len(st.session_state.get('skipped_images', []))
        total_processed_all = total_processed + total_failed + total_skipped
        
        st.sidebar.markdown("---")
        st.sidebar.subheader("📈 Progress")
        st.sidebar.write(f"✅ Successful (2+ products): {total_processed}")
        st.sidebar.write(f"⏭️ Skipped (<2 products): {total_skipped}")
        st.sidebar.write(f"❌ Failed: {total_failed}")
        st.sidebar.write(f"📊 Total Processed: {total_processed_all}")
        
        if total_images > 0:
            progress_percent = (total_processed_all / total_images) * 100
            st.sidebar.progress(progress_percent / 100)
            st.sidebar.write(f"Progress: {progress_percent:.1f}%")
    
# -----------------------------
# FOOTER
# -----------------------------
st.sidebar.markdown("---")
st.sidebar.caption("Shelf Analytics - Grid View (2+ products minimum)")

st.markdown("---")
st.caption("Shelf Analytics | Shelf Utilization")