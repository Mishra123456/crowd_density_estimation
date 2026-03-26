# Viva Preparation: Crowd Density Estimation

**Q1: What is the main objective of this project?**
**A:** To detect people in real-time, count them, classify the crowd density, and visualize their positions through a dynamically updated heatmap.

**Q2: Which machine learning model is utilized for detection and why?**
**A:** YOLOv4-tiny. It provides a vastly superior balance of high accuracy and processing speed compared to older models like MobileNet. It is lightweight enough to run well on laptop CPUs without requiring a dedicated GPU.

**Q3: How does the system isolate only humans?**
**A:** YOLOv4-tiny is trained on the COCO dataset capable of detecting 80 classes. Our code explicitly checks the class ID array and filters the output matrix to only acknowledge detections belonging to the 'person' class index (Index 0).

**Q4: How did you implement the Heatmap?**
**A:** I created a blank, black image matrix, then drew filled white circles at the exact center of every person's bounding box. I applied a heavy Gaussian Blur to smooth these circles into a gradient, scaled the intensities, and finally applied the OpenCV `COLORMAP_JET` before blending it with the original frame.

**Q5: What is the role of OpenCV's `dnn` module here?**
**A:** It acts as the backend engine to load the pre-trained Darknet YOLO weights (`.weights`) and architecture (`.cfg`) and execute the forward pass (inference) on our video frames.

**Q6: How is the density dynamically classified?**
**A:** A custom threshold logic runs on the total number of detected bounding boxes every frame. 0-2 people is LOW, 3-5 is MEDIUM, and 6+ triggers HIGH density.

**Q7: How does the alert system function without spamming the hard drive?**
**A:** The event handler saves a frame when density hits HIGH, but checks a timestamp (`last_save_time`). It imposes a 5-second "cooldown" before allowing another image to be saved.

**Q8: Can this project detect people far away in the background?**
**A:** Yes, YOLOv4-tiny is much better at detecting smaller and distant figures compared to basic SSD models, though it is still ultimately limited by the input resolution (scaled to 416x416).

**Q9: What happens if two people overlap?**
**A:** We use a technique called Non-Maximum Suppression (NMS). When the model draws multiple overlapping boxes around the same person, NMS filters them out and keeps only the single box with the highest confidence score.

**Q10: Why did you use Gaussian Blur for the heatmap?**
**A:** Gaussian Blur softens the hard edges of the plotted center-points, generating a gradual fall-off effect. When points are close together, their blurred regions overlap and accumulate higher intensity, mimicking thermal heat.

**Q11: What is 'Confidence Score'?**
**A:** It is the mathematical probability returned by the neural network indicating how certain it is that the detected object is genuinely a person. We discard detections below 40%.

**Q12: If we wanted to run this on a Raspberry Pi, would it work?**
**A:** Yes! The "tiny" variants of YOLO are explicitly designed for edge devices. It would run efficiently on a Pi.

**Q13: What are the main limitations of this system?**
**A:** It relies purely on bounding boxes rather than specialized crowd counting density maps (like CSRNet), meaning it can still struggle in absolute extreme densities (like a packed concert pit) where individuals are completely hidden behind others.

**Q14: How could you improve this system in the future?**
**A:**
- Upgrading to YOLOv8 Nano for even faster processing.
- Connecting to a cloud database to log density events.
- Creating a graphical web dashboard to view the camera feeds remotely.

**Q15: What makes this a "modular" project?**
**A:** The code is purposefully split into independent files (`detector.py`, `heatmap.py`, `utils.py`, and `main.py`). This allows a developer to easily swap the YOLO detector for something else in the future without breaking the heatmap logic.
