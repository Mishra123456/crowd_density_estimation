import cv2
import time
import os
import argparse
from detector import PersonDetector
from heatmap import HeatmapGenerator
from utils import get_density_class, draw_info, draw_boxes
from tracker import CentroidTracker

def main():
    parser = argparse.ArgumentParser(description="Real-Time Crowd Density Estimation")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory to save high density frames")
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
        
    print(f"[INFO] Starting video stream on source {args.source}...")
    print("[INFO] Press 'q' to quit.")
    
    fps_start_time = time.time()
    fps_frames = 0
    fps = 0
    last_save_time = 0
    save_cooldown = 5.0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[INFO] End of video stream.")
            break
            
        frame = cv2.resize(frame, (800, 600))
        
        boxes, confidences = detector.detect(frame)
        person_count = len(boxes)
        
        # Update tracker
        objects = tracker.update(boxes)
        # Calculate total unique people passed
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
    print("[INFO] Successfully exited.")

if __name__ == "__main__":
    main()
