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

def draw_info(frame, count, density_label, color, fps, total_unique=0):
    """
    Overlays count, tracker stats, density, and FPS info cleanly.
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (380, 200), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    
    cv2.putText(frame, f"Live Count: {count}", (25, 45), font, 0.9, (255, 255, 255), 2)
    cv2.putText(frame, f"Total Unique Seen: {total_unique}", (25, 85), font, 0.8, (200, 200, 255), 2)
    cv2.putText(frame, f"Density: {density_label}", (25, 130), font, 1.0, color, 3)
    cv2.putText(frame, f"FPS: {fps:.2f}", (25, 175), font, 0.8, (255, 180, 0), 2)
    
    if density_label == "HIGH":
        alert_y = 240
        cv2.putText(frame, "ALERT: OVERCROWDING DETECTED!", (25, alert_y), font, 1.1, (0, 0, 255), 3)

def draw_boxes(frame, boxes, confidences, objects=None, color=(0, 255, 0)):
    """
    Draws bounding boxes and highly visible Object Tracking IDs.
    """
    # Draw tracker centroids and IDs
    if objects is not None:
        for (objectID, centroid) in objects.items():
            text = f"ID: {objectID}"
            cv2.putText(frame, text, (centroid[0] - 15, centroid[1] - 15), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            cv2.circle(frame, (centroid[0], centroid[1]), 5, (0, 255, 255), -1)

    for i, (startX, startY, endX, endY) in enumerate(boxes):
        cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
        text = f"{confidences[i]*100:.1f}%"
        y = startY - 10 if startY - 10 > 10 else startY + 10
        cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
