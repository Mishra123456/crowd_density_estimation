import cv2
import numpy as np
import os

class PersonDetector:
    def __init__(self, cfg_path, weights_path, names_path, confidence_threshold=0.4, nms_threshold=0.3):
        """
        Initializes the YOLOv4-tiny detector for extremely accurate 'person' class detection.
        """
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        
        # Validate that all files exist
        if not os.path.exists(cfg_path) or not os.path.exists(weights_path) or not os.path.exists(names_path):
            raise FileNotFoundError(
                "YOLOv4-tiny model files not found. Please run: python models/download_models.py"
            )
            
        # Load COCO names
        with open(names_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
            
        # Load the Darknet YOLOv4-tiny model
        self.net = cv2.dnn.readNetFromDarknet(cfg_path, weights_path)
        
        # Use OpenCV standard CPU backend.
        self.net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
        self.net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)
        
        # Extract the YOLO output layer names
        layer_names = self.net.getLayerNames()
        self.output_layers = [layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
    
    def detect(self, frame):
        """
        Runs YOLOv4-tiny detection on a given frame and returns bounding boxes for 'person' classes.
        """
        h, w = frame.shape[:2]
        
        # Preprocessing: Convert image to blob format required by YOLO
        # Upgraded to 608x608 (from 416x416) to drastically improve accuracy on smaller/distant people
        blob = cv2.dnn.blobFromImage(frame, 1/255.0, (608, 608), swapRB=True, crop=False)
        self.net.setInput(blob)
        
        # Forward pass through the network
        outputs = self.net.forward(self.output_layers)
        
        boxes = []
        confidences = []
        
        # Loop over the outputs
        for output in outputs:
            for detection in output:
                # Extract the scores for all classes
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]
                
                # We only care about the 'person' class (COCO index 0)
                if confidence > self.confidence_threshold and class_id == 0:
                    # Scale the bounding box coordinates back to the size of the image
                    box = detection[0:4] * np.array([w, h, w, h])
                    (centerX, centerY, width, height) = box.astype("int")
                    
                    # Convert center coordinates to top-left corner coordinates
                    x = int(centerX - (width / 2))
                    y = int(centerY - (height / 2))
                    
                    boxes.append([x, y, int(width), int(height)])
                    confidences.append(float(confidence))
                    
        # Apply Non-Maximum Suppression to filter out overlapping multiple boxes for the same person
        indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
        
        final_boxes = []
        final_confidences = []
        
        # Format the surviving boxes
        if len(indices) > 0:
            for i in indices.flatten():
                x, y, w_box, h_box = boxes[i]
                final_boxes.append((x, y, x + w_box, y + h_box))
                final_confidences.append(confidences[i])
                
        return final_boxes, final_confidences
