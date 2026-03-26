# Real-Time Crowd Density Estimation and Heatmap Visualization System

## Project Overview
This project is a real-time computer vision system that estimates crowd density and visualizes it using a dynamic heatmap. It utilizes the powerful and lightweight **YOLOv4-tiny** model to detect people in video streams with high accuracy and classifies the crowd density into LOW, MEDIUM, and HIGH categories.

## Features
- **Accurate Person Detection:** Uses YOLOv4-tiny for robust accuracy without demanding high-end GPU hardware.
- **Dynamic Heatmap Generation:** Creates a visual heatmap of the crowd concentration using Gaussian blur overlays.
- **Density Classification:** 
  - 🟢 LOW (0–2 people)
  - 🟡 MEDIUM (3–5 people)
  - 🔴 HIGH (6+ people)
- **Event Handling:** Automatically saves a snapshot frame when overcrowding (HIGH density) is detected.
- **Performance Metrics:** Displays real-time FPS on the screen.

## Project Structure
```text
crowd_density_estimation/
│── main.py            # Main execution script
│── detector.py        # YOLOv4-tiny object detection class
│── heatmap.py         # Heatmap rendering class
│── utils.py           # Density logic and UI overlays
│── models/
│   └── download_models.py # Script to securely download the weights
│── outputs/           # Automatically stores saved overcrowding frames
│── README.md
│── project_report.md
│── viva_questions.md
```

## Installation Steps
1. **Navigate to the Project Directory**
2. **Install Required Libraries:** Ensure Python 3 is installed.
   ```bash
   pip install opencv-python numpy
   ```
3. **Download the Model Weights:**
   Run the model downloader script to reliably fetch the required YOLOv4-tiny files directly from the Darknet repos.
   ```bash
   python models/download_models.py
   ```

## How to Run

To run the system using your default laptop webcam:
```bash
python main.py
```

To run the system on a provided video file:
```bash
python main.py --source "path/to/your/video.mp4"
```

*Press `q` on your keyboard at any time while the window is focused to safely exit the application.*

## Sample Output Description
When the program runs, you will see your video stream overlaid with:
1. Green bounding boxes around detected people, along with precise confidence percentages.
2. A semi-transparent heatmap layered perfectly over the frame, with intense colors (red) centered geographically around dense groups of people.
3. A sleek information panel displaying the current total people count, the categorized density level, and real-time processing FPS.
4. If the density escalates to HIGH, a red alert message appears on screen and a high-resolution screenshot is immediately saved in the `outputs/` folder.
