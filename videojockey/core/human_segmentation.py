"""
Human segmentation module using MediaPipe.
"""

import cv2
import numpy as np
import mediapipe as mp

class HumanSegmentation:
    def __init__(self):
        self.mp_selfie_segmentation = mp.solutions.selfie_segmentation
        self.segmentation = self.mp_selfie_segmentation.SelfieSegmentation(model_selection=1)  # 1 is the landscape model
        
    def segment_human(self, frame):
        """Segment humans in the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            
        Returns:
            tuple: (mask, segmented_image) where:
                - mask is a binary mask where 1 represents human and 0 represents background
                - segmented_image is the input image with only the human part visible
        """
        # Convert to RGB (MediaPipe requires RGB)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process the frame
        results = self.segmentation.process(frame_rgb)
        
        # Get the segmentation mask
        mask = results.segmentation_mask
        
        # Convert mask to binary (0 or 1)
        mask = np.stack((mask,) * 3, axis=-1) > 0.1
        
        # Create segmented image
        segmented_image = np.zeros_like(frame)
        segmented_image[mask] = frame[mask]
        
        return mask, segmented_image
        
    def replace_background(self, frame, background, mask=None):
        """Replace the background of the frame.
        
        Args:
            frame (numpy.ndarray): Input frame
            background (numpy.ndarray): Background image or color
            mask (numpy.ndarray, optional): Pre-computed segmentation mask
            
        Returns:
            numpy.ndarray: Frame with replaced background
        """
        if mask is None:
            mask, _ = self.segment_human(frame)
        
        # Ensure the background is the same size as the frame
        if isinstance(background, np.ndarray) and background.shape[:2] != frame.shape[:2]:
            background = cv2.resize(background, (frame.shape[1], frame.shape[0]))
            
        # Create result image
        result = np.zeros_like(frame)
        
        # Set background
        if isinstance(background, np.ndarray):
            # Use image as background
            result[~mask] = background[~mask]
        else:
            # Use color as background
            result[~mask] = background
            
        # Set foreground (human)
        result[mask] = frame[mask]
        
        return result