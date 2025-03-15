"""
Video capture module for ingesting video streams from camera or RTSP.
Uses OpenCV with macOS VideoToolbox acceleration when available.
Includes distortion processor for applying effects to the raw video stream.
"""

import cv2
import threading
import time
import numpy as np

from videojockey.core import config
from videojockey.core.video_distortion import VideoDistortion

class VideoCapture:
    def __init__(self):
        self.cap = None
        self.running = False
        self.thread = None
        self.frame = None
        self.lock = threading.Lock()
        self.fps = config.FPS
        self.last_frame_time = 0
        self.distortion_processor = VideoDistortion()
        
    def start(self):
        """Start video capture in a separate thread."""
        if self.running:
            return
            
        # Try to use macOS VideoToolbox acceleration if available
        if config.USE_RTSP:
            # For RTSP, use hardware acceleration
            self.cap = cv2.VideoCapture(config.RTSP_URL, cv2.CAP_FFMPEG)
            
            # Try to enable hardware acceleration for decoding
            self.cap.set(cv2.CAP_PROP_HW_ACCELERATION, cv2.VIDEO_ACCELERATION_ANY)
        else:
            # For webcam
            self.cap = cv2.VideoCapture(0)  # Use default camera
            
        # Set resolution
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, config.VIDEO_WIDTH)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, config.VIDEO_HEIGHT)
        self.cap.set(cv2.CAP_PROP_FPS, config.FPS)
        
        # Check if opened successfully
        if not self.cap.isOpened():
            raise ValueError("Failed to open video source")
            
        # Start capture thread
        self.running = True
        self.thread = threading.Thread(target=self._capture_frames)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop video capture."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
            
        if self.cap:
            self.cap.release()
            self.cap = None
            
    def _capture_frames(self):
        """Capture frames in a loop."""
        while self.running:
            try:
                # Maintain a consistent frame rate
                current_time = time.time()
                elapsed = current_time - self.last_frame_time
                
                # Only capture a new frame if enough time has passed
                if elapsed >= 1.0 / self.fps:
                    ret, frame = self.cap.read()
                    
                    if ret:
                        # Store the frame thread-safely
                        with self.lock:
                            self.frame = frame
                            
                        self.last_frame_time = current_time
                    else:
                        # Failed to read frame
                        if config.DEBUG:
                            print("Failed to read frame")
                        time.sleep(0.01)
                else:
                    # Not time for a new frame yet
                    time.sleep(max(0, (1.0 / self.fps) - elapsed))
                    
            except Exception as e:
                if config.DEBUG:
                    print(f"Video capture error: {e}")
                time.sleep(0.01)
                
    def get_frame(self):
        """Get the latest frame with distortion applied if enabled.
        
        Returns:
            numpy.ndarray: Latest video frame or None if not available
        """
        with self.lock:
            if self.frame is not None:
                # Get a copy to avoid thread issues
                frame_copy = self.frame.copy()
                
                # Apply distortion if level > 0
                return self.distortion_processor.process_frame(frame_copy)
            return None
            
    def set_distortion_level(self, level):
        """Set the distortion level (0-100).
        
        Args:
            level (int): Distortion level from 0 (none) to 100 (maximum)
        """
        self.distortion_processor.set_distortion_level(level)
        
    def get_distortion_level(self):
        """Get the current distortion level.
        
        Returns:
            int: Current distortion level (0-100)
        """
        return self.distortion_processor.get_distortion_level()