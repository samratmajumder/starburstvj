"""
Pixelate effect that reduces image resolution in a dynamic pattern.
"""

import cv2
import numpy as np
import time

# Effect parameters
min_block_size = 5
max_block_size = 40
current_block_size = 15
block_size_change_speed = 0.5
pixelate_pattern = "uniform"  # uniform, radial, or horizontal

def process_frame(frame, audio_info):
    """Apply pixelate effect to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with pixelate effect
    """
    global current_block_size, pixelate_pattern
    
    # Adjust block size based on beat and audio volume
    if audio_info["beat"]:
        # Increase pixelation on beat
        target_block_size = min_block_size + (max_block_size - min_block_size) * audio_info["volume"]
        current_block_size = current_block_size * 0.7 + target_block_size * 0.3
        
        # Occasionally change pixelation pattern on strong beats
        if audio_info["volume"] > 0.7 and np.random.random() < 0.3:
            patterns = ["uniform", "radial", "horizontal"]
            pixelate_pattern = np.random.choice(patterns)
    else:
        # Gradually decrease pixelation when no beat
        current_block_size = max(
            min_block_size,
            current_block_size - block_size_change_speed
        )
    
    # Ensure block size is at least 1 and an integer
    block_size = max(1, int(current_block_size))
    
    # Create output frame
    height, width = frame.shape[:2]
    result = frame.copy()
    
    if pixelate_pattern == "uniform":
        # Uniform pixelation - reduce resolution and resize back
        small = cv2.resize(
            frame, 
            (width // block_size, height // block_size),
            interpolation=cv2.INTER_LINEAR
        )
        result = cv2.resize(
            small, 
            (width, height),
            interpolation=cv2.INTER_NEAREST  # Use nearest neighbor for pixelated look
        )
        
    elif pixelate_pattern == "radial":
        # Radial pixelation - blocks get larger as distance from center increases
        center_x = width // 2
        center_y = height // 2
        max_distance = np.sqrt(center_x**2 + center_y**2)
        
        # Process the image in blocks
        for y in range(0, height, block_size):
            for x in range(0, width, block_size):
                # Calculate distance from center
                dx = x + block_size // 2 - center_x
                dy = y + block_size // 2 - center_y
                distance = np.sqrt(dx**2 + dy**2)
                
                # Calculate block size based on distance
                distance_factor = distance / max_distance
                local_block_size = int(
                    min_block_size + distance_factor * (block_size * 2 - min_block_size)
                )
                local_block_size = max(1, local_block_size)
                
                # Define block region
                x_end = min(x + local_block_size, width)
                y_end = min(y + local_block_size, height)
                
                # Average color in the block
                block = frame[y:y_end, x:x_end]
                avg_color = np.mean(block, axis=(0, 1)).astype(np.uint8)
                
                # Fill the block with the average color
                result[y:y_end, x:x_end] = avg_color
                
    elif pixelate_pattern == "horizontal":
        # Horizontal bands with different pixelation levels
        num_bands = 8
        band_height = height // num_bands
        
        for i in range(num_bands):
            y_start = i * band_height
            y_end = min((i + 1) * band_height, height)
            
            # Alternate between small and large blocks
            band_block_size = block_size if i % 2 == 0 else block_size // 2
            band_block_size = max(1, band_block_size)
            
            # Pixelate this band
            band = frame[y_start:y_end, :]
            
            # Reduce resolution and resize back
            small_band = cv2.resize(
                band, 
                (width // band_block_size, (y_end - y_start) // band_block_size),
                interpolation=cv2.INTER_LINEAR
            )
            result[y_start:y_end, :] = cv2.resize(
                small_band, 
                (width, y_end - y_start),
                interpolation=cv2.INTER_NEAREST
            )
    
    return result