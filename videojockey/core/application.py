"""
Main application class for VideoJockey.
"""

import os
import time
import cv2
import numpy as np
import threading
import pygame
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtCore import Qt

from videojockey.core import config
from videojockey.core.video_capture import VideoCapture
from videojockey.core.audio_processor import AudioProcessor
from videojockey.core.effect_manager import EffectManager
from videojockey.core.human_segmentation import HumanSegmentation
from videojockey.core.message_manager import MessageManager

class VideoJockeyApp(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        
        # Setup UI
        self.setWindowTitle("Video Jockey")
        self.setup_ui()
        
        # Initialize components
        self.video_capture = VideoCapture()
        self.audio_processor = AudioProcessor()
        self.effect_manager = EffectManager()
        self.human_segmentation = HumanSegmentation()
        self.message_manager = MessageManager()
        
        # Application state
        self.running = False
        self.current_frame = None
        self.fps_counter = 0
        self.fps_timer = time.time()
        self.current_fps = 0
        self.fullscreen_mode = False
        
        # Load effects
        self.effect_manager.load_effects()
        self.update_effect_list()
        
        # Load background images
        self.background_images = []
        for bg_path in config.BACKGROUND_IMAGES:
            if os.path.exists(bg_path):
                try:
                    img = cv2.imread(bg_path)
                    if img is not None:
                        self.background_images.append(img)
                except Exception as e:
                    if config.DEBUG:
                        print(f"Failed to load background image {bg_path}: {e}")
        
        # Start timer for UI updates
        self.timer = QtCore.QTimer()
        self.timer.timeout.connect(self.update_ui)
        self.timer.start(30)  # ~30 FPS for UI updates
        
    def setup_ui(self):
        """Setup the application UI."""
        # Main widget and layout
        self.central_widget = QtWidgets.QWidget()
        self.setCentralWidget(self.central_widget)
        self.main_layout = QtWidgets.QVBoxLayout(self.central_widget)
        
        # Video display
        self.video_label = QtWidgets.QLabel()
        self.video_label.setMinimumSize(640, 480)
        self.video_label.setAlignment(QtCore.Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: black;")
        self.video_label.mousePressEvent = self.toggle_fullscreen
        self.main_layout.addWidget(self.video_label)
        
        # Controls widget and layout (can be hidden in fullscreen mode)
        self.controls_widget = QtWidgets.QWidget()
        controls_layout = QtWidgets.QHBoxLayout(self.controls_widget)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        
        # Start/stop button
        self.start_button = QtWidgets.QPushButton("Start")
        self.start_button.clicked.connect(self.toggle_running)
        controls_layout.addWidget(self.start_button)
        
        # Effect selection
        self.effect_combo = QtWidgets.QComboBox()
        self.effect_combo.currentTextChanged.connect(self.change_effect)
        controls_layout.addWidget(self.effect_combo)
        
        # Random effect button
        self.random_button = QtWidgets.QPushButton("Random Effect")
        self.random_button.clicked.connect(self.random_effect)
        controls_layout.addWidget(self.random_button)
        
        # Fullscreen button
        self.fullscreen_button = QtWidgets.QPushButton("Fullscreen")
        self.fullscreen_button.clicked.connect(lambda: self.toggle_fullscreen(None))
        controls_layout.addWidget(self.fullscreen_button)
        
        # FPS counter
        self.fps_label = QtWidgets.QLabel("FPS: 0")
        controls_layout.addWidget(self.fps_label)
        
        # Add controls to main layout
        self.main_layout.addWidget(self.controls_widget)
        
        # Set window size and position
        self.resize(800, 600)
        self.center_window()
        
        # Setup keyboard shortcuts
        self.shortcut_escape = QtWidgets.QShortcut(QtGui.QKeySequence(Qt.Key_Escape), self)
        self.shortcut_escape.activated.connect(self.exit_fullscreen)
        
    def center_window(self):
        """Center the window on the screen."""
        frame_geometry = self.frameGeometry()
        screen_center = QtWidgets.QDesktopWidget().availableGeometry().center()
        frame_geometry.moveCenter(screen_center)
        self.move(frame_geometry.topLeft())
        
    def update_effect_list(self):
        """Update the effect list in the combo box."""
        self.effect_combo.clear()
        effect_names = self.effect_manager.get_effect_names()
        for name in effect_names:
            self.effect_combo.addItem(name)
        
    def change_effect(self, effect_name):
        """Change to a different effect."""
        self.effect_manager.set_effect(effect_name)
        
    def random_effect(self):
        """Switch to a random effect."""
        self.effect_manager.switch_to_random_effect()
        
    def toggle_running(self):
        """Start or stop the application."""
        if self.running:
            self.stop()
        else:
            self.start()
            
    def start(self):
        """Start video and audio processing."""
        if self.running:
            return
            
        try:
            # Start video capture
            self.video_capture.start()
            
            # Start audio processing
            self.audio_processor.start()
            
            # Update UI
            self.running = True
            self.start_button.setText("Stop")
            
        except Exception as e:
            if config.DEBUG:
                print(f"Failed to start: {e}")
            QtWidgets.QMessageBox.critical(self, "Error", f"Failed to start: {str(e)}")
            self.stop()
            
    def stop(self):
        """Stop video and audio processing."""
        # Stop video capture
        if hasattr(self, 'video_capture'):
            self.video_capture.stop()
            
        # Stop audio processing
        if hasattr(self, 'audio_processor'):
            self.audio_processor.stop()
            
        # Update UI
        self.running = False
        self.start_button.setText("Start")
        
    def update_ui(self):
        """Update the UI with the latest frame."""
        if not self.running:
            return
            
        # Get latest frame
        frame = self.video_capture.get_frame()
        
        if frame is not None:
            # Get audio info
            audio_info = self.audio_processor.get_beat_info()
            
            # Process frame with current effect
            processed_frame = self.effect_manager.process_frame(frame, audio_info)
            
            # Apply message overlay
            processed_frame = self.message_manager.render_message(processed_frame, audio_info)
            
            # Convert to QImage for display
            if processed_frame is not None:
                height, width, channel = processed_frame.shape
                bytes_per_line = 3 * width
                q_img = QtGui.QImage(
                    processed_frame.data, 
                    width, 
                    height, 
                    bytes_per_line, 
                    QtGui.QImage.Format_RGB888
                ).rgbSwapped()
                
                # Scale to fit label while maintaining aspect ratio
                scaled_pixmap = QtGui.QPixmap.fromImage(q_img).scaled(
                    self.video_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                
                # Update the video display
                self.video_label.setPixmap(scaled_pixmap)
                
                # Update FPS counter
                self.fps_counter += 1
                if time.time() - self.fps_timer >= 1.0:
                    self.current_fps = self.fps_counter
                    self.fps_counter = 0
                    self.fps_timer = time.time()
                    self.fps_label.setText(f"FPS: {self.current_fps}")
                    
    def toggle_fullscreen(self, event):
        """Toggle fullscreen mode."""
        if self.fullscreen_mode:
            self.exit_fullscreen()
        else:
            self.enter_fullscreen()
    
    def enter_fullscreen(self):
        """Enter fullscreen mode."""
        self.fullscreen_mode = True
        self.controls_widget.hide()
        self.showFullScreen()
        
    def exit_fullscreen(self):
        """Exit fullscreen mode."""
        if self.fullscreen_mode:
            self.fullscreen_mode = False
            self.controls_widget.show()
            self.showNormal()
    
    def closeEvent(self, event):
        """Handle window close event."""
        self.stop()
        super().closeEvent(event)


def run_application():
    """Run the VideoJockey application."""
    import sys
    
    # Initialize PyQt application
    app = QtWidgets.QApplication(sys.argv)
    
    # Create and show the application window
    vj_app = VideoJockeyApp()
    vj_app.show()
    
    # Run the application event loop
    sys.exit(app.exec_())