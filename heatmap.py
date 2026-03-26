import cv2
import numpy as np

class HeatmapGenerator:
    def __init__(self, blur_kernel=(71, 71)):
        """
        Initializes the heatmap generator.
        """
        self.blur_kernel = blur_kernel
        
    def generate(self, frame, boxes):
        """
        Generates a transparent heatmap on top of the given frame,
        based on the center points of the provided bounding boxes.
        """
        h, w = frame.shape[:2]
        
        # Create a blank single channel image for the heatmap intensities
        heatmap = np.zeros((h, w), dtype=np.float32)
        
        for (startX, startY, endX, endY) in boxes:
            # Calculate the horizontal and vertical center of the bounding box
            centerX = int((startX + endX) / 2)
            centerY = int((startY + endY) / 2)
            
            # Add intensity at the person's center node (white circle)
            cv2.circle(heatmap, (centerX, centerY), 45, (255,), -1)
            
        # Apply intense Gaussian blur to create the smooth gradual heatmap effect
        heatmap = cv2.GaussianBlur(heatmap, self.blur_kernel, 0)
        
        # Normalize the heatmap intensity to scale properly (0-255)
        if np.max(heatmap) > 0:
            heatmap = heatmap / np.max(heatmap) * 255
            
        heatmap = heatmap.astype(np.uint8)
        
        # Apply the JET color map (blue -> green -> red)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        
        # Merge the heatmap with the original frame making the heatmap semi-transparent
        alpha = 0.5
        overlay = cv2.addWeighted(frame, 1 - alpha, colored_heatmap, alpha, 0)
        
        # Remove low-intensity color noise (like general blue tint) by masking them off
        gray_heatmap = cv2.cvtColor(colored_heatmap, cv2.COLOR_BGR2GRAY)
        mask = cv2.inRange(gray_heatmap, 50, 255) # ignore low pixel values
        
        # Better blending - Only apply heatmap where mask has high values
        final_result = frame.copy()
        np.copyto(final_result, overlay, where=(mask[:,:,None] > 0))
        
        return final_result
