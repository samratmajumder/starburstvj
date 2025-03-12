"""
Vortex effect that swirls the image around a central point.
"""

import cv2
import numpy as np
import math
import time
import random

# Effect parameters
vortex_strength = 10.0
rotation_speed = 0.5
center_x_offset = 0
center_y_offset = 0
radius_factor = 0.8
time_start = time.time()

def process_frame(frame, audio_info):
    """Apply vortex swirl effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with vortex effect
    """
    global vortex_strength, rotation_speed, center_x_offset, center_y_offset, radius_factor, time_start
    
    current_time = time.time()
    elapsed = current_time - time_start
    
    # Adjust effect parameters based on audio
    if audio_info["beat"]:
        # Increase vortex strength on beat
        vortex_strength = min(20.0, vortex_strength + 2.0)
        
        # Randomize center offset on strong beats
        if audio_info["volume"] > 0.6:
            center_x_offset = random.uniform(-0.1, 0.1)
            center_y_offset = random.uniform(-0.1, 0.1)
    else:
        # Gradually decrease strength
        vortex_strength = max(5.0, vortex_strength - 0.2)
    
    # Use volume to influence rotation speed
    current_rotation = rotation_speed * (1.0 + audio_info["volume"] * 2.0)
    
    # Use bass frequency to adjust radius
    if len(audio_info["frequency_bands"]) > 0:
        bass = audio_info["frequency_bands"][0]
        radius_factor = 0.5 + bass * 2.0
    
    # Create output frame
    height, width = frame.shape[:2]
    output = np.zeros_like(frame)
    
    # Calculate center point with offset
    center_x = width // 2 + int(center_x_offset * width)
    center_y = height // 2 + int(center_y_offset * height)
    
    # Maximum radius
    max_radius = min(width, height) * radius_factor // 2
    
    # Create meshgrid for vectorized operations
    y, x = np.mgrid[0:height, 0:width]
    
    # Calculate polar coordinates
    dx = x - center_x
    dy = y - center_y
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Apply vortex effect
    # Rotation amount decreases with distance from center
    rotation_amount = current_rotation + vortex_strength * (1.0 - np.clip(radius / max_radius, 0, 1))
    
    # Add time-based rotation
    rotation_amount += elapsed * rotation_speed
    
    # Update angles
    new_angle = angle + rotation_amount
    
    # Convert back to Cartesian coordinates
    new_x = center_x + radius * np.cos(new_angle)
    new_y = center_y + radius * np.sin(new_angle)
    
    # Clip coordinates to valid image bounds
    new_x = np.clip(new_x, 0, width-1).astype(np.int32)
    new_y = np.clip(new_y, 0, height-1).astype(np.int32)
    
    # Map pixels to new positions
    output[y, x] = frame[new_y, new_x]
    
    return output