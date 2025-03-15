"""
Message manager for displaying random text messages on screen.
"""

import os
import time
import random
import cv2
import numpy as np

from videojockey.core import config

class MessageManager:
    def __init__(self):
        self.messages = []
        self.current_message = None
        self.current_message_start = 0
        self.last_message_time = 0
        self.font = config.MESSAGE_FONTS[0]
        self.font_size = 1.0
        self.font_color = (255, 255, 255)  # White by default
        self.position = (50, 50)  # Initial position
        self.animation_type = "static"  # static, scroll, pulse, wave
        self.glow_enabled = False
        
        # Load messages from file
        self.load_messages()
    
    def load_messages(self):
        """Load messages from the messages file."""
        if os.path.exists(config.MESSAGES_FILE):
            try:
                with open(config.MESSAGES_FILE, 'r', encoding='utf-8') as f:
                    self.messages = [line.strip() for line in f if line.strip()]
                if config.DEBUG:
                    print(f"Loaded {len(self.messages)} messages")
            except Exception as e:
                if config.DEBUG:
                    print(f"Failed to load messages: {e}")
        else:
            # Create default messages file
            os.makedirs(os.path.dirname(config.MESSAGES_FILE), exist_ok=True)
            default_messages = [
                "Welcome to the party!",
                "Feel the beat",
                "Dance!",
                "Let the music take control",
                "Vibe with the music",
                "The night is young",
                "Lost in the rhythm",
                "Feel the energy",
                "Music is life"
            ]
            try:
                with open(config.MESSAGES_FILE, 'w', encoding='utf-8') as f:
                    f.write('\n'.join(default_messages))
                self.messages = default_messages
                if config.DEBUG:
                    print(f"Created default messages file with {len(self.messages)} messages")
            except Exception as e:
                if config.DEBUG:
                    print(f"Failed to create default messages file: {e}")
                self.messages = default_messages
    
    def select_random_message(self):
        """Select a random message and formatting."""
        current_time = time.time()
        
        # Check if it's time to display a new message
        if self.current_message is None or current_time - self.last_message_time > config.MESSAGE_DISPLAY_INTERVAL:
            # Choose a random message
            if self.messages:
                self.current_message = random.choice(self.messages)
                self.current_message_start = current_time
                self.last_message_time = current_time
                
                # Select random font and size
                self.font = random.choice(config.MESSAGE_FONTS)
                self.font_size = random.uniform(1.0, 3.0)
                
                # Select random color (bright and vibrant)
                hue = random.randint(0, 179)
                sat = random.randint(200, 255)
                val = random.randint(200, 255)
                color_hsv = np.uint8([[[hue, sat, val]]])
                self.font_color = cv2.cvtColor(color_hsv, cv2.COLOR_HSV2BGR)[0][0].tolist()
                
                # Select random position
                def get_random_position(frame_width, frame_height, text_width, text_height):
                    pos_options = [
                        # Center
                        ((frame_width - text_width) // 2, (frame_height - text_height) // 2 + text_height),
                        # Top
                        ((frame_width - text_width) // 2, text_height + 30),
                        # Bottom
                        ((frame_width - text_width) // 2, frame_height - 30),
                        # Left
                        (30, (frame_height - text_height) // 2 + text_height),
                        # Right
                        (frame_width - text_width - 30, (frame_height - text_height) // 2 + text_height),
                        # Random
                        (random.randint(30, frame_width - text_width - 30), 
                         random.randint(text_height + 30, frame_height - 30))
                    ]
                    return random.choice(pos_options)
                
                # We'll set actual position when rendering since we need to know the frame dimensions
                self.position = get_random_position
                
                # Select random animation
                self.animation_type = random.choice(["static", "scroll", "pulse", "wave"])
                
                # Random glow effect
                self.glow_enabled = random.random() > 0.5
    
    def get_message_duration(self):
        """Get the duration of the current message display."""
        if self.current_message is None:
            return 0
        return time.time() - self.current_message_start
    
    def render_message(self, frame, audio_info):
        """Render the current message on the frame.
        
        Args:
            frame (numpy.ndarray): Video frame to render on
            audio_info (dict): Audio information including beat detection
            
        Returns:
            numpy.ndarray: Frame with message rendered
        """
        # Check if there's a message to display
        if self.current_message is None:
            self.select_random_message()
            return frame
        
        # Check if message duration has expired
        if self.get_message_duration() > config.MESSAGE_DURATION:
            self.current_message = None
            return frame
        
        # Create a copy of the frame to draw on
        result = frame.copy()
        height, width = result.shape[:2]
        
        # Calculate text size
        (text_width, text_height), baseline = cv2.getTextSize(
            self.current_message, self.font, self.font_size, 2)
        
        # Get text position
        if callable(self.position):
            try: 
                pos_x, pos_y = self.position(width, height, text_width, text_height)
            except:
                pos_x, pos_y = 0,0
        else:
            pos_x, pos_y = self.position
        
        # Apply animations
        current_time = time.time()
        time_in_animation = current_time - self.current_message_start
        
        # Modify position based on animation type
        if self.animation_type == "scroll":
            # Scroll from right to left
            progress = time_in_animation / config.MESSAGE_DURATION
            pos_x = int(width - (width + text_width) * progress)
        elif self.animation_type == "pulse":
            # Pulse size with beat
            if audio_info["beat"]:
                scale = 1.2 + audio_info["volume"] * 0.5
            else:
                scale = 1.0 + 0.1 * np.sin(time_in_animation * 5)
                
            # Adjust font size
            current_font_size = self.font_size * scale
            
            # Recalculate text size
            (text_width, text_height), baseline = cv2.getTextSize(
                self.current_message, self.font, current_font_size, 2)
                
            # Recenter text
            pos_x = (width - text_width) // 2
            pos_y = (height + text_height) // 2
        elif self.animation_type == "wave":
            # Wave up and down
            wave_height = 20 + audio_info["volume"] * 30
            pos_y += int(np.sin(time_in_animation * 4) * wave_height)
        
        # Get current color, possibly modified by audio
        current_color = self.font_color
        if audio_info["beat"]:
            # Brighten color on beat
            current_color = [min(255, c + 50) for c in self.font_color]
        
        # Add glow effect if enabled
        if self.glow_enabled:
            # Create a slightly larger, blurred text for the glow
            glow_size = self.font_size * 1.05
            glow_color = (0, 0, 0) if sum(current_color) > 380 else (255, 255, 255)  # Contrast with text color
            
            # Draw glow
            cv2.putText(result, self.current_message, (pos_x, pos_y), 
                        self.font, glow_size, glow_color, 5, cv2.LINE_AA)
        
        # Draw text
        cv2.putText(result, self.current_message, (pos_x, pos_y), 
                    self.font, self.font_size, current_color, 2, cv2.LINE_AA)
        
        return result
