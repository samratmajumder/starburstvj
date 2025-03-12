"""
Halo effect that adds a glowing halo around humans.
"""

import cv2
import numpy as np
import time

from videojockey.core.human_segmentation import HumanSegmentation

# Effect parameters
segmentation = HumanSegmentation()
halo_size = 20
halo_color = [0, 200, 255]  # Orange by default
halo_alpha = 0.7
last_color_change = 0
color_change_interval = 1.0  # seconds
halo_pulse = 0.0  # Current pulse amount

def process_frame(frame, audio_info):
    """Apply halo effect to humans in the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with halo effect
    """
    global halo_size, halo_color, halo_alpha, last_color_change, halo_pulse
    
    current_time = time.time()
    
    # Change halo color on strong beats
    if audio_info["beat"] and audio_info["volume"] > 0.4:
        if current_time - last_color_change > color_change_interval:
            # Generate new color
            hue = np.random.randint(0, 180)  # Hue in HSV
            halo_color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
            last_color_change = current_time
    
    # Adjust halo size based on beat
    if audio_info["beat"]:
        halo_pulse = min(1.0, halo_pulse + 0.3)
    else:
        halo_pulse = max(0.0, halo_pulse - 0.05)
    
    # Current halo size with pulse effect
    current_halo_size = int(halo_size * (1.0 + halo_pulse * 0.5))
    
    # Adjust intensity based on audio volume
    current_alpha = min(0.9, halo_alpha * (1.0 + audio_info["volume"] * 0.5))
    
    # Segment humans in the frame
    mask, segmented = segmentation.segment_human(frame)
    
    # Convert mask to single channel
    mask_single = mask[:, :, 0].astype(np.uint8) * 255
    
    # Create a dilated mask for the halo
    kernel = np.ones((current_halo_size, current_halo_size), np.uint8)
    dilated_mask = cv2.dilate(mask_single, kernel, iterations=1)
    
    # Subtract the original mask to get only the halo region
    halo_region = cv2.subtract(dilated_mask, mask_single)
    
    # Create halo effect
    result = frame.copy()
    
    # Create a colored halo with current color
    halo_overlay = np.zeros_like(frame)
    
    # Create a pulsing color based on audio frequency bands
    bass = audio_info["frequency_bands"][0] * 5
    mid = audio_info["frequency_bands"][3] * 5
    high = audio_info["frequency_bands"][7] * 5
    
    # Modulate color with frequency bands
    color = [
        min(255, int(halo_color[0] * (1.0 + high * 0.5))),
        min(255, int(halo_color[1] * (1.0 + mid * 0.5))),
        min(255, int(halo_color[2] * (1.0 + bass * 0.5)))
    ]
    
    # Apply the color to the halo region
    halo_overlay[halo_region > 0] = color
    
    # Apply Gaussian blur to the halo for a glowing effect
    halo_overlay = cv2.GaussianBlur(halo_overlay, (15, 15), 0)
    
    # Blend the halo with the original frame
    cv2.addWeighted(result, 1.0, halo_overlay, current_alpha, 0, result)
    
    return result