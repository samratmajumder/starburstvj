"""
Neon glow effect that adds a colorful glow to the edges.
"""

import cv2
import numpy as np
import time

# Effect parameters
glow_amount = 10
glow_color = [0, 255, 255]  # Initial color (cyan)
color_phase = 0
last_beat_time = 0

def process_frame(frame, audio_info):
    """Apply neon glow effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with neon glow effect
    """
    global glow_amount, glow_color, color_phase, last_beat_time
    
    current_time = time.time()
    
    # Update color phase
    color_phase += 0.02
    if color_phase > 1.0:
        color_phase = 0.0
        
    # Beat response
    if audio_info["beat"]:
        last_beat_time = current_time
        glow_amount = min(25, glow_amount + 5)
    else:
        # Decay glow amount over time
        time_since_beat = current_time - last_beat_time
        if time_since_beat > 0.1:
            glow_amount = max(7, glow_amount - 0.5)
    
    # Volume response
    current_glow = int(glow_amount * (1.0 + audio_info["volume"] * 2))
    
    # Generate dynamic color based on phase and audio frequency bands
    # Use high frequency band to influence color
    high_freq = audio_info["frequency_bands"][-1] * 5
    
    # Create shifting hue based on color phase
    hue = int((color_phase * 180 + high_freq * 20) % 180)
    sat = 255
    val = min(255, 180 + int(audio_info["volume"] * 75))
    
    # Convert HSV to BGR
    color_hsv = np.uint8([[[hue, sat, val]]])
    glow_color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
    
    # Process the frame to create glow effect
    
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Dilate edges to create glow area
    kernel = np.ones((current_glow, current_glow), np.uint8)
    dilated_edges = cv2.dilate(edges, kernel, iterations=1)
    
    # Create colored glow mask
    glow_mask = np.zeros_like(frame)
    glow_mask[dilated_edges > 0] = glow_color
    
    # Blur the glow mask
    glow_mask = cv2.GaussianBlur(glow_mask, (21, 21), 0)
    
    # Blend with original image
    output = cv2.addWeighted(frame, 1.0, glow_mask, 0.8, 0)
    
    return output