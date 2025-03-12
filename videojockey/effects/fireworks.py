"""
Fireworks effect that creates particle explosions on beat.
"""

import cv2
import numpy as np
import random
import time
import math

from videojockey.core.human_segmentation import HumanSegmentation

# Effect parameters
segmentation = HumanSegmentation()
fireworks = []  # List of active fireworks
last_firework_time = 0
min_firework_interval = 0.2  # seconds
particle_lifetime = 1.0  # seconds

class Particle:
    """Represents a single particle in a firework explosion."""
    
    def __init__(self, x, y, color, velocity_x, velocity_y):
        self.x = x
        self.y = y
        self.color = color
        self.velocity_x = velocity_x
        self.velocity_y = velocity_y
        self.gravity = 0.2
        self.alpha = 1.0  # Full opacity
        self.create_time = time.time()
        self.alive = True
        
    def update(self):
        """Update the particle position."""
        # Apply velocity
        self.x += self.velocity_x
        self.y += self.velocity_y
        
        # Apply gravity
        self.velocity_y += self.gravity
        
        # Reduce alpha based on lifetime
        current_time = time.time()
        elapsed = current_time - self.create_time
        self.alpha = max(0, 1.0 - elapsed / particle_lifetime)
        
        # Kill particle if alpha reaches 0
        if self.alpha <= 0:
            self.alive = False

class Firework:
    """Represents a firework explosion with multiple particles."""
    
    def __init__(self, x, y, color=None, intensity=1.0):
        self.x = x
        self.y = y
        
        # Use random color if none provided
        if color is None:
            # Create a bright, saturated color
            hue = random.randint(0, 180)  # Hue in HSV
            color_hsv = np.uint8([[[hue, 255, 255]]])
            self.color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
        else:
            self.color = color
            
        # Create particles
        self.particles = []
        num_particles = int(20 + 30 * intensity)
        
        for _ in range(num_particles):
            # Random velocity in all directions
            angle = random.uniform(0, 2 * math.pi)
            speed = random.uniform(2, 5 + 3 * intensity)
            velocity_x = speed * math.cos(angle)
            velocity_y = speed * math.sin(angle)
            
            # Slightly vary the color for each particle
            color_variation = [
                min(255, max(0, self.color[0] + random.randint(-20, 20))),
                min(255, max(0, self.color[1] + random.randint(-20, 20))),
                min(255, max(0, self.color[2] + random.randint(-20, 20)))
            ]
            
            # Create particle
            particle = Particle(x, y, color_variation, velocity_x, velocity_y)
            self.particles.append(particle)
            
        self.alive = True
        
    def update(self):
        """Update all particles in the firework."""
        for particle in self.particles[:]:
            particle.update()
            if not particle.alive:
                self.particles.remove(particle)
                
        # Kill firework if no particles left
        if not self.particles:
            self.alive = False
            
    def draw(self, frame):
        """Draw all particles on the frame."""
        for particle in self.particles:
            # Calculate alpha-adjusted color
            color = [
                int(particle.color[0] * particle.alpha),
                int(particle.color[1] * particle.alpha),
                int(particle.color[2] * particle.alpha)
            ]
            
            # Draw the particle as a small circle
            cv2.circle(
                frame, 
                (int(particle.x), int(particle.y)), 
                int(2 + particle.alpha * 2),  # Size fades with alpha
                color, 
                -1,  # Filled circle
                cv2.LINE_AA
            )
            
        return frame

def process_frame(frame, audio_info):
    """Apply fireworks effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with fireworks effect
    """
    global fireworks, last_firework_time
    
    current_time = time.time()
    
    # Generate new fireworks on beat
    if audio_info["beat"] and current_time - last_firework_time > min_firework_interval:
        last_firework_time = current_time
        
        # Look for humans to launch fireworks from
        mask, _ = segmentation.segment_human(frame)
        
        # Find human head position (assume top of human silhouette)
        mask_uint8 = (mask[:,:,0] * 255).astype(np.uint8)
        if np.any(mask_uint8):
            # Find contours
            contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if contours:
                # Get the largest contour (main human)
                contour = max(contours, key=cv2.contourArea)
                
                # Find the top point (head approximation)
                top_point = tuple(contour[contour[:, :, 1].argmin()][0])
                
                # Launch firework from top of head
                intensity = 0.5 + audio_info["volume"] * 2  # Intensity based on volume
                firework = Firework(top_point[0], top_point[1], intensity=intensity)
                fireworks.append(firework)
        else:
            # If no human detected, create random fireworks
            x = random.randint(50, frame.shape[1] - 50)
            y = random.randint(50, frame.shape[0] - 50)
            intensity = 0.5 + audio_info["volume"] * 2
            firework = Firework(x, y, intensity=intensity)
            fireworks.append(firework)
    
    # Update and draw existing fireworks
    result = frame.copy()
    for firework in fireworks[:]:
        firework.update()
        if firework.alive:
            result = firework.draw(result)
        else:
            fireworks.remove(firework)
    
    return result