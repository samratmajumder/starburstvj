"""
Glitch effect that creates digital distortion effects on beat.
"""

import cv2
import numpy as np
import random
import time

# Effect parameters
last_glitch_time = 0
glitch_duration = 0.2  # seconds
is_glitching = False
channel_shift = 5
block_size = 30
intensity = 0.5  # Base intensity

def process_frame(frame, audio_info):
    """Apply glitch effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with glitch effect
    """
    global last_glitch_time, is_glitching, channel_shift, block_size, intensity
    
    current_time = time.time()
    
    # Trigger glitch on beat
    if audio_info["beat"]:
        last_glitch_time = current_time
        is_glitching = True
        
        # Adjust glitch parameters based on audio
        channel_shift = int(5 + audio_info["volume"] * 15)
        block_size = int(20 + audio_info["volume"] * 30)
        intensity = 0.5 + audio_info["volume"] * 0.5
    
    # Check if we should still be glitching
    time_since_glitch = current_time - last_glitch_time
    if time_since_glitch > glitch_duration:
        is_glitching = False
    
    # If not glitching, return original frame
    if not is_glitching:
        return frame
    
    # Calculate glitch intensity
    current_intensity = intensity * (1.0 - time_since_glitch / glitch_duration)
    
    # Create a copy of the frame
    result = frame.copy()
    height, width = result.shape[:2]
    
    # Apply random glitch effects based on intensity
    
    # 1. Color channel shifting
    if random.random() < current_intensity * 0.8:
        # Separate the color channels
        b, g, r = cv2.split(result)
        
        # Shift the red channel horizontally
        shift_amount = int(random.uniform(-channel_shift, channel_shift))
        if shift_amount > 0:
            r = np.hstack((r[:, shift_amount:], r[:, :shift_amount]))
        elif shift_amount < 0:
            shift_amount = abs(shift_amount)
            r = np.hstack((r[:, -shift_amount:], r[:, :-shift_amount]))
        
        # Shift the blue channel horizontally in the opposite direction
        shift_amount = int(random.uniform(-channel_shift, channel_shift))
        if shift_amount > 0:
            b = np.hstack((b[:, shift_amount:], b[:, :shift_amount]))
        elif shift_amount < 0:
            shift_amount = abs(shift_amount)
            b = np.hstack((b[:, -shift_amount:], b[:, :-shift_amount]))
        
        # Merge the channels back together
        result = cv2.merge((b, g, r))
    
    # 2. Random block displacement
    if random.random() < current_intensity * 0.7:
        num_blocks = int(current_intensity * 5) + 1
        for _ in range(num_blocks):
            # Select random block
            block_x = random.randint(0, width - block_size)
            block_y = random.randint(0, height - block_size)
            
            # Select random destination
            dest_x = int(block_x + random.uniform(-30, 30))
            dest_y = int(block_y + random.uniform(-10, 10))
            
            # Make sure destination is within bounds
            dest_x = max(0, min(width - block_size, dest_x))
            dest_y = max(0, min(height - block_size, dest_y))
            
            # Copy block to new destination
            block = result[block_y:block_y+block_size, block_x:block_x+block_size].copy()
            result[dest_y:dest_y+block_size, dest_x:dest_x+block_size] = block
    
    # 3. Horizontal scan lines
    if random.random() < current_intensity * 0.6:
        num_lines = int(10 * current_intensity) + 1
        line_height = int(3 * current_intensity) + 1
        
        for _ in range(num_lines):
            y = random.randint(0, height - line_height)
            # Shift the scanline horizontally
            shift = int(random.uniform(-20, 20))
            if shift != 0:
                line = result[y:y+line_height, :].copy()
                if shift > 0:
                    result[y:y+line_height, shift:] = line[:, :-shift]
                    result[y:y+line_height, :shift] = line[:, -shift:]
                else:
                    shift = abs(shift)
                    result[y:y+line_height, :-shift] = line[:, shift:]
                    result[y:y+line_height, -shift:] = line[:, :shift]
    
    # 4. Noise
    if random.random() < current_intensity * 0.5:
        noise = np.zeros_like(result)
        cv2.randu(noise, 0, 255)
        noise_mask = np.random.random(result.shape[:2])
        noise_mask = (noise_mask < current_intensity * 0.1).reshape(height, width, 1)
        result = np.where(noise_mask, noise, result)
    
    return result