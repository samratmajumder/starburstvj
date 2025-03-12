"""
Configuration settings for the VideoJockey application.
"""

import os
import cv2

# Video settings
VIDEO_WIDTH = 1280
VIDEO_HEIGHT = 720
FPS = 30
USE_RTSP = False
RTSP_URL = "rtsp://your_rtsp_stream_url"  # Replace with your RTSP stream URL

# Audio settings
AUDIO_SAMPLE_RATE = 44100
AUDIO_CHUNK_SIZE = 1024
AUDIO_CHANNELS = 1
AUDIO_FORMAT = 16  # bit depth

# Beat detection settings
BEAT_SENSITIVITY = 1.5
BEAT_MIN_INTERVAL = 0.1  # seconds

# Effects settings
DEFAULT_EFFECT = "kaleidoscope"
EFFECT_TRANSITION_TIME = 2.0  # seconds
AUTO_SWITCH_EFFECTS = True
AUTO_SWITCH_INTERVAL = 20  # seconds

# Resource paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESOURCES_DIR = os.path.join(ROOT_DIR, "resources")
EFFECTS_DIR = os.path.join(ROOT_DIR, "effects")

# Message display settings
MESSAGES_FILE = os.path.join(RESOURCES_DIR, "messages.txt")
MESSAGE_DISPLAY_INTERVAL = 10.0  # seconds
MESSAGE_DURATION = 5.0  # seconds
MESSAGE_FONTS = [
    cv2.FONT_HERSHEY_SIMPLEX,
    cv2.FONT_HERSHEY_PLAIN,
    cv2.FONT_HERSHEY_DUPLEX,
    cv2.FONT_HERSHEY_COMPLEX,
    cv2.FONT_HERSHEY_TRIPLEX,
    cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
    cv2.FONT_HERSHEY_SCRIPT_COMPLEX
]

# Background image paths for replacement effects
BACKGROUND_IMAGES = [
    os.path.join(RESOURCES_DIR, "bg1.jpg"),
    os.path.join(RESOURCES_DIR, "bg2.jpg"),
    os.path.join(RESOURCES_DIR, "bg3.jpg"),
]

# Debug mode
DEBUG = True