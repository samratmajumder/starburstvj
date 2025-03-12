"""
Color inversion effect that inverts colors on beat.
"""

import cv2
import numpy as np
import time

# Effect parameters
last_beat_time = 0
is_inverted = False
inversion_duration = 0.2  # seconds

def process_frame(frame, audio_info):
    """Apply color inversion effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with inverted colors
    """
    global last_beat_time, is_inverted, inversion_duration
    
    current_time = time.time()
    
    # Detect beat and trigger inversion
    if audio_info["beat"]:
        last_beat_time = current_time
        is_inverted = True
    
    # Apply inversion if active
    if is_inverted and current_time - last_beat_time < inversion_duration:
        # Calculate smooth transition for inversion
        progress = 1.0 - (current_time - last_beat_time) / inversion_duration
        
        # Create inverted frame
        inverted = cv2.bitwise_not(frame)
        
        # Blend between original and inverted based on progress
        output = cv2.addWeighted(frame, 1.0 - progress, inverted, progress, 0)
        return output
    else:
        is_inverted = False
        return frame