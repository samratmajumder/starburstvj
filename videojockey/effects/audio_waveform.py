"""
Audio waveform visualization effect.
"""

import cv2
import numpy as np
import time

# Effect parameters
waveform_height = 0.3  # Height of waveform as percentage of screen height
waveform_color = [0, 255, 0]  # Green by default
last_color_change = 0
color_change_interval = 2.0  # seconds
position = "bottom"  # bottom, top, or center

def process_frame(frame, audio_info):
    """Apply audio waveform visualization to the frame.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame with audio waveform
    """
    global waveform_color, last_color_change, position
    
    current_time = time.time()
    
    # Change color on beat
    if audio_info["beat"]:
        if current_time - last_color_change > color_change_interval:
            # Generate new color
            hue = np.random.randint(0, 180)  # Hue in HSV
            waveform_color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0].tolist()
            last_color_change = current_time
            
            # Also change position occasionally
            if np.random.random() < 0.3:
                position = np.random.choice(["bottom", "top", "center"])
    
    # Get raw audio data
    audio_data = audio_info["raw_audio"]
    
    # Create a copy of the frame
    result = frame.copy()
    height, width = result.shape[:2]
    
    # Prepare waveform
    wave_height = int(height * waveform_height)
    
    # Adjust color based on volume
    volume_boost = audio_info["volume"] * 3
    current_color = [
        min(255, int(waveform_color[0] * (1.0 + volume_boost))),
        min(255, int(waveform_color[1] * (1.0 + volume_boost))),
        min(255, int(waveform_color[2] * (1.0 + volume_boost)))
    ]
    
    # Create transparent overlay for waveform
    overlay = np.zeros_like(result)
    
    # Resample audio data to match the width of the frame
    if len(audio_data) > 0:
        # Normalize audio data to -1 to 1 range
        normalized_data = audio_data / np.max(np.abs(audio_data)) if np.max(np.abs(audio_data)) > 0 else audio_data
        
        # Resample to match the width
        resampled = np.interp(
            np.linspace(0, len(normalized_data), width),
            np.arange(len(normalized_data)),
            normalized_data
        )
        
        # Scale to fit the waveform height
        scaled_data = resampled * wave_height // 2
        
        # Draw the waveform
        if position == "bottom":
            base_y = height - wave_height // 2
        elif position == "top":
            base_y = wave_height // 2
        else:  # center
            base_y = height // 2
        
        for x in range(width - 1):
            y1 = int(base_y + scaled_data[x])
            y2 = int(base_y + scaled_data[x + 1])
            cv2.line(overlay, (x, y1), (x + 1, y2), current_color, 2, cv2.LINE_AA)
        
        # Add visualizations for frequency bands
        band_width = width // 8
        for i in range(8):
            band_height = int(audio_info["frequency_bands"][i] * wave_height * 2)
            band_x = i * band_width
            
            # Draw frequency band visualization
            if position == "bottom":
                cv2.rectangle(
                    overlay,
                    (band_x, height - band_height),
                    (band_x + band_width - 2, height),
                    current_color,
                    -1
                )
            elif position == "top":
                cv2.rectangle(
                    overlay,
                    (band_x, 0),
                    (band_x + band_width - 2, band_height),
                    current_color,
                    -1
                )
            else:  # center
                half_height = band_height // 2
                cv2.rectangle(
                    overlay,
                    (band_x, base_y - half_height),
                    (band_x + band_width - 2, base_y + half_height),
                    current_color,
                    -1
                )
    
    # Apply the overlay with alpha blending
    alpha = 0.6 + audio_info["volume"] * 0.3  # Increase opacity with volume
    cv2.addWeighted(result, 1.0, overlay, alpha, 0, result)
    
    # Add beat indicator
    if audio_info["beat"]:
        # Draw a pulse circle that fades out
        center_x = width // 2
        center_y = height // 2
        radius = int(min(width, height) * 0.1 * (1.0 + audio_info["volume"]))
        cv2.circle(result, (center_x, center_y), radius, current_color, 2, cv2.LINE_AA)
    
    return result