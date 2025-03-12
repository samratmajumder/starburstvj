"""
Laser beams effect that shoots lasers from humans in the scene.
"""

import cv2
import numpy as np
import time
import random
import math

from videojockey.core.human_segmentation import HumanSegmentation

# Effect parameters
segmentation = HumanSegmentation()
last_beat_time = 0
lasers = []  # List of active laser beams
max_lasers = 10
laser_lifetime = 0.8  # seconds
last_laser_time = 0
min_laser_interval = 0.1  # seconds

# Laser colors in BGR format
LASER_COLORS = [
    (0, 0, 255),    # Red
    (0, 255, 255),  # Yellow
    (0, 255, 0),    # Green
    (255, 0, 0),    # Blue
    (255, 0, 255),  # Magenta
]

class Laser:
    """Represents a laser beam."""
    
    def __init__(self, start_x, start_y, angle, color, speed=30, thickness=2):
        self.start_x = start_x
        self.start_y = start_y
        self.angle = angle  # Radians
        self.length = 20  # Initial length
        self.max_length = random.randint(100, 500)
        self.color = color
        self.speed = speed
        self.thickness = thickness
        self.create_time = time.time()
        self.alive = True
        
    def update(self):
        """Update the laser beam."""
        current_time = time.time()
        time_alive = current_time - self.create_time
        
        # Grow the laser up to max_length
        if self.length < self.max_length:
            self.length += self.speed
        
        # Kill the laser after lifetime
        if time_alive > laser_lifetime:
            self.alive = False
            
    def draw(self, frame):
        """Draw the laser beam on the frame."""
        end_x = int(self.start_x + self.length * math.cos(self.angle))
        end_y = int(self.start_y + self.length * math.sin(self.angle))
        
        # Create fading effect based on lifetime
        current_time = time.time()
        time_alive = current_time - self.create_time
        fade_factor = max(0, 1.0 - (time_alive / laser_lifetime))
        
        # Draw the main laser beam
        cv2.line(frame, (self.start_x, self.start_y), (end_x, end_y), self.color, self.thickness)
        
        # Add glow effect
        for i in range(1, 4):
            alpha = fade_factor * (0.7 - i * 0.2)
            if alpha > 0:
                cv2.line(frame, (self.start_x, self.start_y), (end_x, end_y), self.color, self.thickness + i*2, cv2.LINE_AA)
        
        # Add a bright core
        cv2.line(frame, (self.start_x, self.start_y), (end_x, end_y), (255, 255, 255), 1, cv2.LINE_AA)
        
        return frame

def process_frame(frame, audio_info):
    """Apply laser beams effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with laser beams
    """
    global lasers, last_beat_time, last_laser_time
    
    current_time = time.time()
    
    # Segment humans in the frame
    mask, segmented = segmentation.segment_human(frame)
    
    # Generate new lasers on beat or based on volume
    should_create_laser = False
    
    if audio_info["beat"]:
        last_beat_time = current_time
        should_create_laser = True
    elif audio_info["volume"] > 0.4 and current_time - last_laser_time > min_laser_interval:
        # Create occasional lasers based on volume
        should_create_laser = random.random() < audio_info["volume"] * 0.3
    
    if should_create_laser and current_time - last_laser_time > min_laser_interval:
        last_laser_time = current_time
        
        # Find the contours of humans
        mask_uint8 = (mask[:,:,0] * 255).astype(np.uint8)
        contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Get the largest contour (main human)
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            
            # Get random point on the contour
            if len(largest_contour) > 10:  # Ensure enough points
                # Pick a random point from the contour
                idx = random.randint(0, len(largest_contour) - 1)
                start_x, start_y = largest_contour[idx][0]
                
                # Random angle away from human
                # Calculate center of contour
                M = cv2.moments(largest_contour)
                if M["m00"] != 0:
                    center_x = int(M["m10"] / M["m00"])
                    center_y = int(M["m01"] / M["m00"])
                    
                    # Angle from center to edge point
                    angle = math.atan2(start_y - center_y, start_x - center_x)
                else:
                    angle = random.uniform(0, 2 * math.pi)
                    
                # Random laser color
                color = random.choice(LASER_COLORS)
                
                # Create laser with thickness based on audio volume
                thickness = int(2 + audio_info["volume"] * 4)
                
                # Create and add the laser
                laser = Laser(start_x, start_y, angle, color, thickness=thickness)
                lasers.append(laser)
                
                # Limit the number of active lasers
                if len(lasers) > max_lasers:
                    lasers.pop(0)
    
    # Update and draw existing lasers
    result = frame.copy()
    for laser in lasers[:]:
        laser.update()
        if laser.alive:
            result = laser.draw(result)
        else:
            lasers.remove(laser)
    
    return result