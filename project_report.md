# Project Report: Real-Time Crowd Density Estimation and Heatmap Visualization System

## 1. Problem Statement
Monitoring large crowds in public spaces manually is a challenging, resource-intensive task. Ensuring public safety requires immediate alerts when a specific area exceeds its safe capacity.

## 2. Objective
To design and develop a modular, real-time computer vision system capable of counting people, categorizing the specific crowd density into tiers (LOW, MEDIUM, HIGH), and visually plotting concentration metrics on a dynamic heatmap overlay.

## 3. Methodology
The pipeline consists of the following sequential modules running per frame:
1. **Input Stream:** Capturing frames from a webcam or a video file using OpenCV.
2. **Detection:** Forwarding the pre-processed frame through the highly accurate YOLOv4-tiny Deep Neural Network (`cv2.dnn`).
3. **Filtering:** Retaining bounding boxes categorized specifically as the `person` class with an adequate confidence score. Applying Non-Maximum Suppression to remove overlapping duplicates.
4. **Counting and Density logic:** Evaluating the total count against defined thresholds.
5. **Heatmap Generation:** Mapping Gaussian blurred circles to bounding-box centers and applying a JET color map.
6. **Alert/Event Handling:** Saving a screenshot to disk automatically utilizing time-throttling when density is flagged as HIGH.

## 4. Tools & Technologies
- **Language:** Python 3
- **Libraries:** OpenCV (`cv2`), NumPy, standard `os` and `time`
- **Model:** YOLOv4-tiny architecture optimized for high accuracy on CPUs.

## 5. Features
- No GPU dependency while maintaining robust accuracy.
- Transparent, dynamically shifting heatmap.
- Modular and expandable Python architecture.
- Automatic snapshotting logs for security event auditing.

## 6. Challenges Faced
- Generating an aesthetically pleasing heatmap required tuning Gaussian Blur kernel sizes.
- Handling overlapping bounding boxes in a dense crowd, which requires Non-Maximum Suppression (handled perfectly by Darknet architecture integration).
- Designing the text overlay so it remains readable regardless of the background color stream.

## 7. Learning Outcomes
- Deeper understanding of integrating YOLO Deep Learning models via OpenCV `dnn`.
- Applying matrix manipulation and normalization algorithms using NumPy for generating thermal map representations.
- Principles of handling asynchronous events like throttling file writing.

## 8. Real-world Applications
- **Retail Analytics:** Queue monitoring and store layout optimization.
- **Smart Cities:** Traffic and metro station crowd monitoring.
- **Safety and Security:** Detecting rapid stampede buildups in stadiums or concerts.
