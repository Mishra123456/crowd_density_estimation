# Real-Time Crowd Density Estimation and Heatmap Visualization System

## Project Overview
This project is an advanced real-time computer vision system that estimates crowd density and visualizes it using a dynamic heatmap. Built for scale, it utilizes the powerful and lightweight **YOLOv4-tiny** model paired with customized input tensor scaling to detect people in video streams with extreme accuracy. It categorizes live foot traffic crowd density into LOW, MEDIUM, and HIGH tiers.

## Advanced Features
- **Accurate Person Detection:** Uses YOLOv4-tiny deployed via OpenCV's `dnn` module for robust accuracy without demanding high-end GPU hardware.
- **Dynamic Heatmap Generation:** Creates a highly aesthetic, real-time thermal heatmap of crowd concentration using scaled Gaussian blur algorithms and color mapping.
- **Unique Object Tracking:** A purely mathematical NumPy Centroid Tracker actively assigns a unique ID to every single individual detected, allowing the system to log the total unique traffic flow over time instead of just basic snapshot counts.
- **Interactive Region of Interest (ROI):** Utilize the `--roi` argument to selectively draw a physical rectangular boundary constraint on the street/room, forcing the AI to exclusively monitor within your defined area!
- **Data Analytics Generation:** Silently logs detection matrices in the background and dynamically uses `matplotlib` to render and save a comprehensive analytical line graph charting density history the second you close the video stream.
- **Density Classification & Alert Handling:** 
  - 🟢 LOW (0–2 people)
  - 🟡 MEDIUM (3–5 people)
  - 🔴 HIGH (6+ people)
  - Automatically saves a snapshot photographic frame into the `outputs/` folder when a dangerous overcrowding event triggers!

## Project Structure
```text
crowd_density_estimation/
│── main.py            # Main execution script with stream loop and graphing
│── detector.py        # YOLOv4-tiny high-res object detection class
│── tracker.py         # Advanced Centroid Tracker module
│── heatmap.py         # Temporal heatmap renderer class
│── utils.py           # Density metrics and UI overlay parameters
│── models/
│   └── download_models.py # Extractor script to securely fetch Darknet YOLO weights
│── outputs/           # Automatically stores photographic alerts and `.png` graph charts
│── README.md
│── project_report.md
│── viva_questions.md
```

## Installation Steps
1. **Navigate to the Project Directory**
2. **Install Required Python Libraries:** Ensure Python 3 is installed.
   ```bash
   pip install opencv-python numpy matplotlib scipy
   ```
3. **Download the Model Weights:**
   Run the model downloader script to reliably fetch the required YOLOv4-tiny neural network weights directly from the central Darknet repositories.
   ```bash
   python models/download_models.py
   ```

## How to Run

To run the system using your default connected webcam:
```bash
python main.py
```

To run the system analyzing a specifically provided video file:
```bash
python main.py --source "path/to/your/video.mp4"
```

To run the system while utilizing the **Custom Interactive ROI Mapping Tool**:
```bash
python main.py --source "path/to/your/video.mp4" --roi
```
*(You will be requested to draw your bounding area constraint with a mouse and press ENTER before the analytics initiate).*

*Press `q` on your keyboard at any time while the target window is focused to safely exit the application stream and view your automatically Generated Analytical Chart.*
