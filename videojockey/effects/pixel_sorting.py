"""
Pixel sorting effect that sorts pixels in rows or columns for a glitchy look.
"""

import cv2
import numpy as np
import time

# Effect parameters
sort_vertical = False
sort_threshold = 30
last_beat_time = 0
beat_interval = 0.5  # seconds
sort_intensity = 0.7

def process_frame(frame, audio_info):
    """Apply pixel sorting effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with pixel sorting effect
    """
    global sort_vertical, sort_threshold, last_beat_time, sort_intensity
    
    current_time = time.time()
    
    # Change sort direction on beat
    if audio_info["beat"] and current_time - last_beat_time > beat_interval:
        sort_vertical = not sort_vertical
        last_beat_time = current_time
        
    # Adjust threshold based on volume
    current_threshold = int(sort_threshold * (1.0 + audio_info["volume"] * 2))
    
    # Create a copy of the frame
    result = frame.copy()
    
    # Apply pixel sorting
    if sort_vertical:
        # Sort pixels in columns
        for x in range(0, frame.shape[1]):
            for y in range(0, frame.shape[0] - 1, 10):  # Process in chunks for efficiency
                # Get column segment
                chunk_size = min(10, frame.shape[0] - y)
                col = result[y:y+chunk_size, x].reshape(chunk_size, 3)
                
                # Calculate luminance
                luminance = 0.299 * col[:, 2] + 0.587 * col[:, 1] + 0.114 * col[:, 0]
                
                # Only sort if average luminance exceeds threshold
                if np.mean(luminance) > current_threshold:
                    # Sort pixels by luminance
                    indices = np.argsort(luminance)
                    result[y:y+chunk_size, x] = col[indices]
    else:
        # Sort pixels in rows
        for y in range(0, frame.shape[0]):
            for x in range(0, frame.shape[1] - 1, 10):  # Process in chunks for efficiency
                # Get row segment
                chunk_size = min(10, frame.shape[1] - x)
                row = result[y, x:x+chunk_size].reshape(chunk_size, 3)
                
                # Calculate luminance
                luminance = 0.299 * row[:, 2] + 0.587 * row[:, 1] + 0.114 * row[:, 0]
                
                # Only sort if average luminance exceeds threshold
                if np.mean(luminance) > current_threshold:
                    # Sort pixels by luminance
                    indices = np.argsort(luminance)
                    result[y, x:x+chunk_size] = row[indices]
    
    # Adjust intensity based on volume
    current_intensity = min(1.0, sort_intensity + audio_info["volume"] * 0.3)
    
    # Blend with original
    output = cv2.addWeighted(frame, 1.0 - current_intensity, result, current_intensity, 0)
    
    return output