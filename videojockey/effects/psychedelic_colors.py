"""
Psychedelic color effect with posterization and color shifting.
"""

import cv2
import numpy as np
import time
import random

# Effect parameters
color_shift = [0, 0, 0]  # BGR color shift
shift_speed = 0.05
posterize_levels = 4  # Number of color levels per channel
hue_rotation = 0.0
saturation_boost = 1.2

def process_frame(frame, audio_info):
    """Apply psychedelic color effects to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with psychedelic color effects
    """
    global color_shift, posterize_levels, hue_rotation, saturation_boost
    
    # Update color shift randomly
    for i in range(3):
        color_shift[i] += random.uniform(-shift_speed, shift_speed) * 5
        color_shift[i] = color_shift[i] % 255
    
    # Adjust parameters on beat
    if audio_info["beat"]:
        # Change posterize levels on beat
        posterize_levels = random.choice([3, 4, 5, 6, 8])
        
        # Change hue rotation more dramatically on beat
        hue_rotation = (hue_rotation + random.uniform(30, 90)) % 180
        
        # Increase saturation on beat
        saturation_boost = 1.5 + audio_info["volume"]
    else:
        # Gradually rotate hue
        hue_rotation = (hue_rotation + 0.5) % 180
        
        # Adjust saturation boost based on volume
        saturation_boost = 1.2 + audio_info["volume"] * 0.5
    
    # Convert to HSV for easier color manipulation
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Rotate hue
    hsv[:, :, 0] = (hsv[:, :, 0] + hue_rotation) % 180
    
    # Boost saturation
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation_boost, 0, 255).astype(np.uint8)
    
    # Convert back to BGR
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Apply posterization (color quantization)
    # Divide by levels, round, and multiply back to get fewer color levels
    levels = 256 // posterize_levels
    result = ((result // levels) * levels).astype(np.uint8)
    
    # Apply color channel shifting
    b, g, r = cv2.split(result)
    
    # Shift each channel
    b = np.roll(b, int(color_shift[0]) % 100 - 50, axis=1)  # Horizontal shift for blue
    g = np.roll(g, int(color_shift[1]) % 100 - 50, axis=0)  # Vertical shift for green
    r = np.roll(r, -int(color_shift[2]) % 50 - 25, axis=1)  # Opposite horizontal shift for red
    
    # Merge channels back
    result = cv2.merge([b, g, r])
    
    # Add noise based on bass frequency
    if len(audio_info["frequency_bands"]) > 0:
        bass_intensity = audio_info["frequency_bands"][0] * 8
        if bass_intensity > 0.5:
            noise = np.random.randint(0, int(50 * bass_intensity), result.shape[:2], dtype=np.uint8)
            noise_mask = np.random.random(result.shape[:2]) < (bass_intensity * 0.1)
            noise_mask = noise_mask[:, :, np.newaxis].repeat(3, axis=2)
            result = np.where(noise_mask, result + noise[:, :, np.newaxis], result)
    
    return result