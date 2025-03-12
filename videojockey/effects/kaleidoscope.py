"""
Optimized kaleidoscope effect using vectorized operations.
"""

import cv2
import numpy as np
import math

# Effect parameters
segments = 6
center_x_offset = 0
center_y_offset = 0
zoom_level = 1.0

def process_frame(frame, audio_info):
    """Apply a kaleidoscope effect to the frame using vectorized operations.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with kaleidoscope effect
    """
    global segments, center_x_offset, center_y_offset, zoom_level
    
    # Adjust parameters based on audio
    if audio_info["beat"]:
        # Add some randomization to segments on beat
        segments = max(3, min(8, segments + np.random.randint(-1, 2)))
        # Adjust zoom level on beat
        zoom_level = max(0.8, min(1.2, zoom_level + 0.1 * (audio_info["volume"] * 5 - 0.5)))
    
    # Apply subtle zoom changes based on volume
    current_zoom = zoom_level * (1.0 + 0.1 * audio_info["volume"])
    
    # Adjust center offsets based on bass frequency (first band)
    if len(audio_info["frequency_bands"]) > 0:
        bass_intensity = audio_info["frequency_bands"][0] * 20
        center_x_offset = int(frame.shape[1] * 0.1 * math.sin(bass_intensity))
        center_y_offset = int(frame.shape[0] * 0.1 * math.cos(bass_intensity))
    
    # Create kaleidoscope effect
    height, width = frame.shape[:2]
    center_x = width // 2 + center_x_offset
    center_y = height // 2 + center_y_offset
    
    # Downsample the frame for speed
    scale_factor = 0.5
    small_frame = cv2.resize(frame, None, fx=scale_factor, fy=scale_factor, 
                            interpolation=cv2.INTER_LINEAR)
    small_height, small_width = small_frame.shape[:2]
    small_center_x = small_width // 2 + int(center_x_offset * scale_factor)
    small_center_y = small_height // 2 + int(center_y_offset * scale_factor)
    
    # Create coordinate meshgrid
    y_coords, x_coords = np.mgrid[0:small_height, 0:small_width]
    
    # Calculate polar coordinates
    dx = x_coords - small_center_x
    dy = y_coords - small_center_y
    
    # Apply zoom
    dx = dx / current_zoom
    dy = dy / current_zoom
    
    # Calculate radius and angle
    radius = np.sqrt(dx**2 + dy**2)
    angle = np.arctan2(dy, dx)
    
    # Calculate segment angle
    segment_angle = 2 * np.pi / segments
    
    # Wrap angle to first segment
    segment_num = np.floor(angle / segment_angle)
    angle_segment = angle % segment_angle
    flip_mask = (segment_num.astype(int) % 2) == 1
    angle_segment[flip_mask] = segment_angle - angle_segment[flip_mask]
    
    # Convert back to Cartesian coordinates
    source_x = small_center_x + radius * np.cos(angle_segment)
    source_y = small_center_y + radius * np.sin(angle_segment)
    
    # Clip coordinates to image bounds
    source_x = np.clip(source_x, 0, small_width - 1).astype(np.int32)
    source_y = np.clip(source_y, 0, small_height - 1).astype(np.int32)
    
    # Create output image
    small_output = np.zeros_like(small_frame)
    small_output[y_coords, x_coords] = small_frame[source_y, source_x]
    
    # Resize back to original size
    output = cv2.resize(small_output, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return output