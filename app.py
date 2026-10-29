import streamlit as st
import cv2
import numpy as np
import pandas as pd
import random
from PIL import Image
from ultralytics import YOLO

# --- ENTERPRISE LOGIC ENGINE ---
class TrafficSystem:
    def __init__(self, speed_limit=80):
        self.limit = speed_limit
        self.log = [] # Stores all vehicle data for the Pandas DataFrame

    def process_vehicle(self, vehicle_id):
        # Simulate radar speed detection
        speed = random.randint(40, 120) 
        
        if speed > self.limit:
            self.generate_ticket(vehicle_id, speed)
            self.log.append({"Vehicle ID": vehicle_id, "Speed (km/h)": speed, "Status": "VIOLATION"})
        else:
            self.log.append({"Vehicle ID": vehicle_id, "Speed (km/h)": speed, "Status": "Legal"})

    def generate_ticket(self, vehicle_id, speed):
        # Crash-Proof File Writing (try...except)
        try:
            with open(f"ticket_{vehicle_id}.txt", "w") as file:
                file.write(f"OFFICIAL TICKET\nVehicle: {vehicle_id}\nSpeed: {speed} km/h (Limit: {self.limit})\nFine: 1500 INR\n")
        except IOError:
            st.error(f"System Error: Could not write ticket to disk for {vehicle_id}.")

# --- MACHINE LEARNING CACHE ---
# Cache the model so it doesn't reload on every interaction
@st.cache_resource
def load_model():
    return YOLO('yolov8n.pt')

model = load_model()

# --- WEBSITE UI & LAYOUT ---
st.set_page_config(page_title="Enterprise Traffic System", page_icon="🚦")
st.title("🚦 Smart Traffic & Violation Dashboard")
st.write("Upload a traffic image. The system will detect vehicles, monitor speed, and auto-generate tickets for violations.")

# File Uploader
uploaded_file = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display the uploaded image
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Feed", use_container_width=True)
    
    # Process the image when the button is clicked
    if st.button("Process Traffic Feed"):
        with st.spinner('Analyzing traffic and logging data...'):
            # Convert PIL image to OpenCV format (NumPy array)
            img_array = np.array(image)
            
            # Run YOLOv8 inference (Filtering for vehicles: 2=car, 3=motorcycle, 5=bus, 7=truck)
            results = model.predict(img_array, classes=[2, 3, 5, 7], conf=0.4)
            
            # Plot the bounding boxes on the image
            res_plotted = results[0].plot()
            
            # Convert back to PIL format (RGB) for Streamlit to display properly
            result_image = Image.fromarray(res_plotted[..., ::-1])
            
            st.success("Analysis Complete!")
            st.image(result_image, caption="Processed Feed", use_container_width=True)
            
            # --- ENTERPRISE DATA PROCESSING ---
            system = TrafficSystem(speed_limit=80)
            boxes = results[0].boxes
            
            # Loop through every single box YOLO found and process it
            for i, box in enumerate(boxes):
                vehicle_id = f"RIT-V{i+1000}" 
                system.process_vehicle(vehicle_id)
            
            # Display the Enterprise Data Table using Pandas
            st.subheader("Live Traffic Analytics Log")
            df = pd.DataFrame(system.log)
            
            if not df.empty:
                st.dataframe(df, use_container_width=True)
                
                # Show ticket alerts
                tickets = df[df["Status"] == "VIOLATION"]
                if not tickets.empty:
                    st.warning(f"⚠️ Generated {len(tickets)} automated speeding tickets. Text files saved locally.")
                else:
                    st.info("No speeding violations detected in this frame.")
            else:
                st.info("No vehicles detected.")
