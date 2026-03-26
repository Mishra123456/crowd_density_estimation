import cv2

def get_density_class(count):
    """
    Classifies crowd density based on the total count of people.
    LOW (0-2), MEDIUM (3-5), HIGH (6+)
    Returns: (Label, BGR_Color)
    """
    if count <= 2:
        return "LOW", (0, 255, 0)      # Green
    elif count <= 5:
        return "MEDIUM", (0, 255, 255) # Yellow
    else:
        return "HIGH", (0, 0, 255)     # Red

def draw_info(frame, count, density_label, color, fps):
    """
    Overlays count, density, and FPS info cleanly onto the frame using a transparent dark background.
    """
    # Create semi-transparent overlay for text background
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 160), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    # Fonts
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    # Texts
    cv2.putText(frame, f"People Count: {count}", (25, 45), font, 1, (255, 255, 255), 2)
    cv2.putText(frame, f"Density: {density_label}", (25, 95), font, 1.1, color, 3)
    cv2.putText(frame, f"FPS: {fps:.2f}", (25, 140), font, 0.9, (255, 180, 0), 2)
    
    # Alert text if HIGH density detected
    if density_label == "HIGH":
        alert_y = 210
        cv2.putText(frame, "ALERT: OVERCROWDING DETECTED!", (25, alert_y), font, 1.2, (0, 0, 255), 3)

def draw_boxes(frame, boxes, confidences, color=(0, 255, 0)):
    """
    Draws bounding boxes around detected people with confidence scores.
    """
    for i, (startX, startY, endX, endY) in enumerate(boxes):
        # Calculate box area and draw
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        
        # Display confidence score above bounding box
        text = f"{confidences[i]*100:.1f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
