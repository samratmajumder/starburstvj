"""
Mirror split effect that creates symmetric reflections.
"""

import cv2
import numpy as np
import time

# Effect parameters
split_position = 0.5  # Position of the split (0.0 to 1.0)
mirror_mode = 0  # 0: left to right, 1: right to left, 2: top to bottom, 3: bottom to top
mode_change_time = 0
mode_change_interval = 5.0  # seconds
split_change_speed = 0.01

def process_frame(frame, audio_info):
    """Apply mirror split effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with mirror split effect
    """
    global split_position, mirror_mode, mode_change_time, split_change_speed
    
    current_time = time.time()
    
    # Change mirror mode periodically or on strong beats
    if (current_time - mode_change_time > mode_change_interval or 
        (audio_info["beat"] and audio_info["volume"] > 0.4)):
        mirror_mode = (mirror_mode + 1) % 4
        mode_change_time = current_time
    
    # Adjust split position based on audio volume
    volume_influence = (audio_info["volume"] - 0.5) * 0.05
    split_position += split_change_speed + volume_influence
    
    # Keep split_position in valid range
    if split_position <= 0.1 or split_position >= 0.9:
        split_change_speed = -split_change_speed
        split_position = max(0.1, min(0.9, split_position))
    
    # Create output frame
    output = np.zeros_like(frame)
    height, width = frame.shape[:2]
    
    # Apply different mirror modes
    if mirror_mode == 0:  # Left to right
        split_x = int(width * split_position)
        left_side = frame[:, :split_x]
        # Mirror the left side to create the right side
        right_side = cv2.flip(left_side, 1)
        # Handle size mismatch
        right_width = width - split_x
        if right_side.shape[1] > right_width:
            right_side = right_side[:, :right_width]
        # Combine the sides
        output[:, :split_x] = left_side
        output[:, split_x:split_x+right_side.shape[1]] = right_side
        
    elif mirror_mode == 1:  # Right to left
        split_x = int(width * split_position)
        right_side = frame[:, split_x:]
        # Mirror the right side to create the left side
        left_side = cv2.flip(right_side, 1)
        # Handle size mismatch
        if left_side.shape[1] > split_x:
            left_side = left_side[:, :split_x]
        # Combine the sides
        output[:, :left_side.shape[1]] = left_side
        output[:, split_x:] = right_side
        
    elif mirror_mode == 2:  # Top to bottom
        split_y = int(height * split_position)
        top_side = frame[:split_y, :]
        # Mirror the top side to create the bottom side
        bottom_side = cv2.flip(top_side, 0)
        # Handle size mismatch
        bottom_height = height - split_y
        if bottom_side.shape[0] > bottom_height:
            bottom_side = bottom_side[:bottom_height, :]
        # Combine the sides
        output[:split_y, :] = top_side
        output[split_y:split_y+bottom_side.shape[0], :] = bottom_side
        
    else:  # Bottom to top
        split_y = int(height * split_position)
        bottom_side = frame[split_y:, :]
        # Mirror the bottom side to create the top side
        top_side = cv2.flip(bottom_side, 0)
        # Handle size mismatch
        if top_side.shape[0] > split_y:
            top_side = top_side[:split_y, :]
        # Combine the sides
        output[:top_side.shape[0], :] = top_side
        output[split_y:, :] = bottom_side
    
    # Add beat responsiveness - pulse the image slightly on beat
    if audio_info["beat"]:
        # Create a slightly zoomed version of the output
        zoom_factor = 1.05
        zoomed = cv2.resize(output, None, fx=zoom_factor, fy=zoom_factor)
        
        # Crop to original size from center
        start_x = (zoomed.shape[1] - width) // 2
        start_y = (zoomed.shape[0] - height) // 2
        zoomed = zoomed[start_y:start_y+height, start_x:start_x+width]
        
        # Blend with original based on beat intensity
        beat_strength = min(1.0, audio_info["volume"] * 2)
        output = cv2.addWeighted(output, 1.0 - beat_strength * 0.5, zoomed, beat_strength * 0.5, 0)
    
    return output