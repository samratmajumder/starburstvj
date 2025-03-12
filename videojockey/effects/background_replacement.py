"""
Background replacement effect that uses human segmentation.
"""

import cv2
import numpy as np
import time
import os
import random

from videojockey.core import config
from videojockey.core.human_segmentation import HumanSegmentation

# Effect parameters
bg_index = 0
last_bg_change = 0
bg_change_interval = 5.0  # seconds
background_images = []
segmentation = HumanSegmentation()

# Initialize dynamic background parameters
bg_scroll_x = 0
bg_scroll_y = 0
bg_zoom = 1.0

def load_backgrounds():
    """Load background images."""
    global background_images
    
    # Clear current backgrounds
    background_images = []
    
    # Add some generated backgrounds
    # 1. Colorful gradient
    gradient = np.zeros((720, 1280, 3), dtype=np.uint8)
    for i in range(1280):
        color = ((i * 255) // 1280, ((1280 - i) * 255) // 1280, 128)
        gradient[:, i] = color
    background_images.append(gradient)
    
    # 2. Starfield
    starfield = np.zeros((720, 1280, 3), dtype=np.uint8)
    for _ in range(1000):
        x = random.randint(0, 1279)
        y = random.randint(0, 719)
        brightness = random.randint(100, 255)
        starfield[y, x] = (brightness, brightness, brightness)
    background_images.append(cv2.GaussianBlur(starfield, (3, 3), 0))
    
    # Load from files if available
    for bg_path in config.BACKGROUND_IMAGES:
        if os.path.exists(bg_path):
            try:
                img = cv2.imread(bg_path)
                if img is not None:
                    # Resize to match video dimensions
                    img = cv2.resize(img, (1280, 720))
                    background_images.append(img)
            except Exception as e:
                if config.DEBUG:
                    print(f"Failed to load background image {bg_path}: {e}")
    
    # If no backgrounds, add a default one
    if not background_images:
        # Create a rainbow background
        rainbow = np.zeros((720, 1280, 3), dtype=np.uint8)
        for y in range(720):
            hue = int((y * 180) / 720)
            color_hsv = np.uint8([[[hue, 255, 255]]])
            color_bgr = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0]
            rainbow[y, :] = color_bgr
        background_images.append(rainbow)

def process_frame(frame, audio_info):
    """Apply background replacement effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with replaced background
    """
    global bg_index, last_bg_change, bg_scroll_x, bg_scroll_y, bg_zoom
    
    # Load backgrounds if not loaded
    if not background_images:
        load_backgrounds()
    
    current_time = time.time()
    
    # Change background on beat or after interval
    if (audio_info["beat"] and current_time - last_bg_change > 1.0) or (current_time - last_bg_change > bg_change_interval):
        bg_index = (bg_index + 1) % len(background_images)
        last_bg_change = current_time
    
    # Get current background
    background = background_images[bg_index].copy()
    
    # Update dynamic background parameters
    if audio_info["beat"]:
        # Add random movement on beat
        bg_scroll_x += random.uniform(-10, 10)
        bg_scroll_y += random.uniform(-10, 10)
        # Pulse zoom on beat
        bg_zoom = 1.0 + audio_info["volume"] * 0.3
    else:
        # Slowly return to normal
        bg_zoom = max(1.0, bg_zoom - 0.01)
    
    # Apply background transformations
    # 1. Create larger background for scrolling and zooming
    bg_height, bg_width = background.shape[:2]
    enlarged_bg = cv2.resize(background, None, fx=bg_zoom, fy=bg_zoom)
    
    # 2. Calculate scroll offsets - ensure we don't divide by zero
    if enlarged_bg.shape[1] <= bg_width:
        scroll_x = 0
    else:
        scroll_x = int(bg_scroll_x) % (enlarged_bg.shape[1] - bg_width)
        
    if enlarged_bg.shape[0] <= bg_height:
        scroll_y = 0
    else:
        scroll_y = int(bg_scroll_y) % (enlarged_bg.shape[0] - bg_height)
    
    # 3. Crop to get the visible portion
    visible_bg = enlarged_bg[
        scroll_y:scroll_y + bg_height,
        scroll_x:scroll_x + bg_width
    ]
    
    # Ensure the background is the right size
    if visible_bg.shape[:2] != frame.shape[:2]:
        visible_bg = cv2.resize(visible_bg, (frame.shape[1], frame.shape[0]))
    
    # Get human segmentation
    mask, _ = segmentation.segment_human(frame)
    
    # Apply background replacement
    result = frame.copy()
    result[~mask] = visible_bg[~mask]
    
    # Add some special effects on beat
    if audio_info["beat"]:
        # Add a flash border around the person
        outline = np.zeros_like(frame)
        
        # Get the outline by dilating the mask and subtracting the original mask
        mask_single = mask[:, :, 0].astype(np.uint8) * 255
        kernel = np.ones((5, 5), np.uint8)
        dilated_mask = cv2.dilate(mask_single, kernel, iterations=2)
        outline_mask = dilated_mask - mask_single
        
        # Create a colored outline that pulses with the beat
        outline_color = [255, 255, 255]  # White outline
        outline[outline_mask > 0] = outline_color
        
        # Add the outline to the result
        result = cv2.addWeighted(result, 0.7, outline, 0.3, 0)
    
    return result