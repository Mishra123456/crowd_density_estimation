import cv2
import time
import os
import argparse
from detector import PersonDetector
from heatmap import HeatmapGenerator
from utils import get_density_class, draw_info, draw_boxes
from tracker import CentroidTracker
import datetime
import matplotlib.pyplot as plt

def main():
    parser = argparse.ArgumentParser(description="Real-Time Crowd Density Estimation")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save high density frames")
    parser.add_argument("--roi", action="store_true", help="Draw a custom Region of Interest to monitor only a specific area")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    
    current_dir = os.path.dirname(os.path.abspath(__file__))
    cfg_path = os.path.join(current_dir, "models", "yolov4-tiny.cfg")
    weights_path = os.path.join(current_dir, "models", "yolov4-tiny.weights")
    names_path = os.path.join(current_dir, "models", "coco.names")
    
    print("[INFO] Loading YOLOv4-tiny model for high accuracy detection...")
    try:
        # Lowered confidence to 0.3 to catch partially occluded people in dense scenarios
        detector = PersonDetector(cfg_path, weights_path, names_path, confidence_threshold=0.3)
    except FileNotFoundError as e:
        print(f"\n[ERROR] {e}\n")
        print("Please run: python models/download_models.py to download the YOLO weights.")
        return
        
    heatmap_gen = HeatmapGenerator(blur_kernel=(81, 81))
    
    # Initialize the robust Centroid Tracker (Improvement 1)
    tracker = CentroidTracker(maxDisappeared=30, maxDistance=100)
    
    source = int(args.source) if args.source.isdigit() else args.source
    cap = cv2.VideoCapture(source)
    
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source {args.source}")
        return

    selected_roi = None
    if args.roi:
        print("\n[INFO] Please select the Region of Interest (ROI) using your mouse.")
        print("[INFO] Draw a rectangle and press ENTER or SPACE to confirm.")
        ret, setup_frame = cap.read()
        if ret:
            setup_frame = cv2.resize(setup_frame, (800, 600))
            selected_roi = cv2.selectROI("Select ROI (Press ENTER when done)", setup_frame, fromCenter=False, showCrosshair=True)
            cv2.destroyWindow("Select ROI (Press ENTER when done)")
        
    print(f"[INFO] Starting video stream on source {args.source}...")
    print("[INFO] Press 'q' to quit.")
    
    fps_start_time = time.time()
    fps_frames = 0
    fps = 0
    last_save_time = 0
    save_cooldown = 5.0
    
    # Analytics history logs for ending graph
    start_datetime = datetime.datetime.now()
    history_time = []
    history_count = []
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
            
        frame = cv2.resize(frame, (800, 600))
        
        raw_boxes, raw_confidences = detector.detect(frame)
        
        boxes = []
        confidences = []
        
        # Filter bounding boxes if user drew a custom ROI
        if selected_roi is not None and selected_roi[2] > 0 and selected_roi[3] > 0:
            (rx, ry, rw, rh) = selected_roi
            cv2.rectangle(frame, (rx, ry), (rx + rw, ry + rh), (255, 0, 0), 2)
            cv2.putText(frame, "Active ROI", (rx, ry - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
            
            for (box, conf) in zip(raw_boxes, raw_confidences):
                cX = int((box[0] + box[2]) / 2.0)
                cY = int((box[1] + box[3]) / 2.0)
                if rx < cX < rx + rw and ry < cY < ry + rh:
                    boxes.append(box)
                    confidences.append(conf)
        else:
            boxes = raw_boxes
            confidences = raw_confidences
            
        person_count = len(boxes)
        
        # Update tracker
        objects = tracker.update(boxes)
        # Calculate total unique people passed within monitored bounds
        total_unique = tracker.nextObjectID - 1
        
        density_label, density_color = get_density_class(person_count)
        
        if person_count > 0:
            frame_display = heatmap_gen.generate(frame, boxes)
        else:
            frame_display = frame.copy()
            
        draw_boxes(frame_display, boxes, confidences, objects=objects, color=density_color)
        
        fps_frames += 1
        elapsed_time = time.time() - fps_start_time
        if elapsed_time > 1.0:
            fps = fps_frames / elapsed_time
            fps_start_time = time.time()
            fps_frames = 0
            
            # Log metrics every 1 second for the analytics graph
            history_time.append((datetime.datetime.now() - start_datetime).total_seconds())
            history_count.append(person_count)
            
        draw_info(frame_display, person_count, density_label, density_color, fps, total_unique)
        
        if density_label == "HIGH":
            current_time = time.time()
            if current_time - last_save_time > save_cooldown:
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                save_path = os.path.join(args.output_dir, f"overcrowded_{timestamp}.jpg")
                cv2.imwrite(save_path, frame_display)
                print(f"[ALERT] Overcrowding detected! Frame saved to {save_path}")
                last_save_time = current_time
                
        cv2.imshow("Crowd Density Estimation & Heatmap", frame_display)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Quitting stream...")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Successfully exited video stream.")
    
    # 7. Generate Data Analytics Graph upon exiting
    if len(history_count) > 0:
        print("[INFO] Generating Crowd Density Analytics Graph...")
        plt.figure(figsize=(10, 5))
        plt.plot(history_time, history_count, label="Live Crowd Count", color="blue", linewidth=2.5)
        plt.fill_between(history_time, history_count, color="blue", alpha=0.2)
        
        # Draw analytical density boundary thresholds
        plt.axhline(y=2.5, color='orange', linestyle='--', label="MEDIUM Density")
        plt.axhline(y=5.5, color='red', linestyle='--', label="HIGH Density")
        
        plt.title('Real-Time Crowd Density Analytics')
        plt.xlabel('Time (Seconds)')
        plt.ylabel('Total People Detected')
        plt.legend(loc="upper left")
        plt.grid(True)
        
        graph_path = os.path.join(args.output_dir, "crowd_density_graph.png")
        plt.savefig(graph_path)
        print(f"[INFO] Analytical Graph successfully saved to {graph_path}")
        
        # Display the graph automatically
        plt.show()

if __name__ == "__main__":
    main()
