"""
Liquid effect that distorts the image like a flowing liquid.
"""

import cv2
import numpy as np
import time
import math

# Effect parameters
time_start = time.time()
wave_frequency = 20.0
wave_amplitude = 10.0
flow_speed = 2.0
horizontal_flow = True
vertical_flow = True

def process_frame(frame, audio_info):
    """Apply liquid distortion effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with liquid effect
    """
    global wave_frequency, wave_amplitude, flow_speed, horizontal_flow, vertical_flow
    
    current_time = time.time()
    elapsed = current_time - time_start
    
    # Adjust effect parameters based on audio
    if audio_info["beat"]:
        # Increase amplitude on beat
        wave_amplitude = min(20.0, wave_amplitude + 1.0)
        
        # Toggle flow direction occasionally on strong beats
        if audio_info["volume"] > 0.7:
            if np.random.random() < 0.3:
                horizontal_flow = not horizontal_flow
            if np.random.random() < 0.3:
                vertical_flow = not vertical_flow
    else:
        # Gradually decrease amplitude
        wave_amplitude = max(5.0, wave_amplitude - 0.2)
    
    # Adjust frequency based on audio energy in mid-range frequencies
    if len(audio_info["frequency_bands"]) > 3:
        mid_freq = audio_info["frequency_bands"][3]
        wave_frequency = 10.0 + mid_freq * 30.0
    
    # Create output frame
    height, width = frame.shape[:2]
    output = np.zeros_like(frame)
    
    # Create meshgrid for vectorized operations
    y_coords, x_coords = np.mgrid[0:height, 0:width]
    
    # Apply liquid distortion
    current_wave_amp = wave_amplitude * (1.0 + audio_info["volume"])
    
    # Time factor for flow
    time_factor = elapsed * flow_speed
    
    # Apply sinusoidal distortion
    if horizontal_flow:
        # Horizontal flowing waves
        x_offset = current_wave_amp * np.sin(
            (y_coords / height * wave_frequency) + time_factor
        )
        x_coords = x_coords + x_offset.astype(np.int32)
    
    if vertical_flow:
        # Vertical flowing waves
        y_offset = current_wave_amp * np.sin(
            (x_coords / width * wave_frequency) + time_factor
        )
        y_coords = y_coords + y_offset.astype(np.int32)
    
    # Add circular ripples from center on beat
    if audio_info["beat"]:
        center_x = width // 2
        center_y = height // 2
        
        # Distance from center
        dx = x_coords - center_x
        dy = y_coords - center_y
        distance = np.sqrt(dx**2 + dy**2)
        
        # Ripple effect
        ripple_offset = current_wave_amp * 0.5 * np.sin(distance * 0.1 - time_factor * 2)
        
        # Apply ripple outward from center
        angle = np.arctan2(dy, dx)
        x_coords = x_coords + (ripple_offset * np.cos(angle)).astype(np.int32)
        y_coords = y_coords + (ripple_offset * np.sin(angle)).astype(np.int32)
    
    # Ensure coordinates are within bounds
    x_coords = np.clip(x_coords, 0, width - 1)
    y_coords = np.clip(y_coords, 0, height - 1)
    
    # Map pixels to their new positions
    output[y_coords, x_coords] = frame[y_coords, x_coords]
    
    # If there are holes (black pixels), fill them using a simple interpolation
    mask = np.all(output == 0, axis=2)
    if np.any(mask):
        # Create a blurred version of the result to fill holes
        blurred = cv2.GaussianBlur(output, (5, 5), 0)
        output[mask] = blurred[mask]
    
    return output