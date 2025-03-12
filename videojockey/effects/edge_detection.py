"""
Edge detection effect that highlights edges with neon colors.
"""

import cv2
import numpy as np
import time

# Effect parameters
threshold1 = 50
threshold2 = 150
edge_color = [0, 255, 255]  # Yellow by default
last_color_change = 0
color_change_interval = 1.0  # seconds

def process_frame(frame, audio_info):
    """Apply edge detection with colorful edges to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with colorful edge detection
    """
    global threshold1, threshold2, edge_color, last_color_change, color_change_interval
    
    current_time = time.time()
    
    # Adjust edge detection sensitivity based on audio volume
    volume_boost = audio_info["volume"] * 5
    current_threshold1 = max(10, min(100, threshold1 - int(volume_boost * 40)))
    current_threshold2 = max(50, min(200, threshold2 - int(volume_boost * 40)))
    
    # Change edge color on beat
    if audio_info["beat"]:
        if current_time - last_color_change > color_change_interval:
            # Generate new neon color
            hue = np.random.randint(0, 180)  # Hue in HSV
            edge_color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
            last_color_change = current_time
    
    # Convert to grayscale for edge detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply Canny edge detection
    edges = cv2.Canny(blurred, current_threshold1, current_threshold2)
    
    # Create color mask for edges
    color_mask = np.zeros_like(frame)
    color_mask[edges > 0] = edge_color
    
    # Blend original with colored edges
    edge_strength = 0.7 + 0.3 * volume_boost  # Edge visibility increases with volume
    output = cv2.addWeighted(frame, 1.0 - edge_strength, color_mask, edge_strength, 0)
    
    return output