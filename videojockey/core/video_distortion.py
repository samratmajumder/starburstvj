"""
Video distortion processor to apply various distortion effects.
"""

import cv2
import numpy as np
import random
import time

class VideoDistortion:
    def __init__(self):
        self.distortion_level = 0  # 0-100
        self.last_distortion_change = time.time()
        self.current_distortion_type = 0
        self.distortion_types = [
            self._apply_posterize,
            self._apply_wash_out,
            self._apply_color_shift,
            self._apply_noise,
            self._apply_glitch
        ]
    
    def set_distortion_level(self, level):
        """Set the distortion level (0-100)."""
        self.distortion_level = max(0, min(100, level))
    
    def get_distortion_level(self):
        """Get the current distortion level."""
        return self.distortion_level
    
    def process_frame(self, frame, audio_info=None):
        """Apply distortion to the frame based on the current distortion level."""
        if self.distortion_level <= 0:
            return frame
            
        # Randomly change distortion type every few seconds
        current_time = time.time()
        if current_time - self.last_distortion_change > 3.0:
            self.current_distortion_type = random.randint(0, len(self.distortion_types) - 1)
            self.last_distortion_change = current_time
        
        # Apply selected distortion
        distortion_func = self.distortion_types[self.current_distortion_type]
        return distortion_func(frame)
    
    def _apply_posterize(self, frame):
        """Apply posterization effect (reduce color depth)."""
        intensity = self.distortion_level / 100.0
        
        # Calculate number of color levels (fewer = more posterization)
        levels = max(2, int(256 * (1 - intensity * 0.95)))
        
        # Quantize colors
        frame_float = frame.astype(np.float32) / 255.0
        frame_posterized = np.floor(frame_float * levels) / levels
        result = (frame_posterized * 255).astype(np.uint8)
        
        # Blend with original based on intensity
        return cv2.addWeighted(frame, 1 - intensity, result, intensity, 0)
    
    def _apply_wash_out(self, frame):
        """Apply wash-out effect (reduce contrast and shift colors)."""
        intensity = self.distortion_level / 100.0
        
        # Reduce contrast and add brightness
        washed = cv2.convertScaleAbs(frame, alpha=1-intensity*0.7, beta=intensity*50)
        
        # Add slight blur for a dreamy effect
        if intensity > 0.5:
            blur_amount = int(intensity * 10) * 2 + 1
            washed = cv2.GaussianBlur(washed, (blur_amount, blur_amount), 0)
        
        return cv2.addWeighted(frame, 1 - intensity, washed, intensity, 0)
    
    def _apply_color_shift(self, frame):
        """Apply color channel shifting."""
        intensity = self.distortion_level / 100.0
        
        # Split into channels
        b, g, r = cv2.split(frame)
        
        # Calculate shift amount
        shift_x = int(intensity * 20)
        shift_y = int(intensity * 15)
        
        # Apply shift to random channels
        if shift_x > 0 or shift_y > 0:
            channels = [b, g, r]
            shift_channel = random.randint(0, 2)
            
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            channels[shift_channel] = cv2.warpAffine(channels[shift_channel], 
                                                    M, 
                                                    (frame.shape[1], frame.shape[0]))
            
            result = cv2.merge(channels)
            return cv2.addWeighted(frame, 1 - intensity, result, intensity, 0)
        
        return frame
    
    def _apply_noise(self, frame):
        """Apply random noise to the frame."""
        intensity = self.distortion_level / 100.0
        
        # Create noise
        noise = np.zeros_like(frame, dtype=np.uint8)
        cv2.randn(noise, 128, 30)  # mean=128, stddev=30
        
        # Blend with original based on intensity
        return cv2.addWeighted(frame, 1 - intensity*0.7, noise, intensity*0.7, 0)
        
    def _apply_glitch(self, frame):
        """Apply digital glitch effect with block displacement."""
        intensity = self.distortion_level / 100.0
        
        # Create copy for result
        result = frame.copy()
        
        # Number of glitch blocks based on intensity
        num_glitches = int(intensity * 15)
        
        height, width = frame.shape[:2]
        
        for _ in range(num_glitches):
            # Random block size and position
            block_height = random.randint(5, max(5, int(height * 0.1)))
            block_width = random.randint(10, max(10, int(width * 0.3)))
            
            y = random.randint(0, height - block_height - 1)
            x = random.randint(0, width - block_width - 1)
            
            # Copy block to a new position or duplicate it
            y_offset = random.randint(-30, 30)
            x_offset = random.randint(-30, 30)
            
            y_dest = max(0, min(height - block_height - 1, y + y_offset))
            x_dest = max(0, min(width - block_width - 1, x + x_offset))
            
            block = frame[y:y+block_height, x:x+block_width].copy()
            result[y_dest:y_dest+block_height, x_dest:x_dest+block_width] = block
        
        return cv2.addWeighted(frame, 1 - intensity, result, intensity, 0)