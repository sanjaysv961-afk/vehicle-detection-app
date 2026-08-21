import streamlit as st
import cv2
import numpy as np
from PIL import Image
from ultralytics import YOLO

# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# Website Layout and UI
st.set_page_config(page_title="Traffic Vehicle Detection", page_icon="🚦")
st.title("🚦 Real-Time Vehicle Detection")
st.write("Upload a traffic image, and our Deep Learning model will detect and classify cars, trucks, buses, and motorcycles.")

# File Uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)
    
    # Process the image when the button is clicked
    if st.button("Detect Vehicles"):
        with st.spinner('Analyzing traffic...'):
            # Convert PIL image to OpenCV format (NumPy array)
            img_array = np.array(image)
            
            # Run YOLOv8 inference (Filtering for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck)
            results = model.predict(img_array, classes=[2, 3, 5, 7], conf=0.4)
            
            # Plot the bounding boxes on the image
            res_plotted = results[0].plot()
            
            # Convert back to PIL format (RGB) for Streamlit to display properly
            result_image = Image.fromarray(res_plotted[..., ::-1])
            
            st.success("Detection Complete!")
            st.image(result_image, caption="Detected Vehicles", use_container_width=True)
            
            # Display a count of detected vehicles
            vehicle_count = len(results[0].boxes)
            st.info(f"Total Vehicles Detected: {vehicle_count}")
