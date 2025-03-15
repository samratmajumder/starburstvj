"""
Effect manager for loading and applying visual effects.
"""

import os
import time
import importlib
import importlib.util
import random
import numpy as np
import cv2

from videojockey.core import config

class EffectManager:
    def __init__(self):
        self.effects = {}
        self.disabled_effects = set()
        self.current_effect_name = None
        self.next_effect_name = None
        self.transition_start_time = 0
        self.transition_progress = 0
        self.last_auto_switch_time = time.time()
        
    def load_effects(self):
        """Load all effects from the effects directory."""
        effect_path = config.EFFECTS_DIR
        
        # Clear current effects
        self.effects = {}
        
        # Find all Python files in the effects directory
        for filename in os.listdir(effect_path):
            if filename.endswith('.py') and not filename.startswith('__'):
                effect_name = os.path.splitext(filename)[0]
                file_path = os.path.join(effect_path, filename)
                
                try:
                    # Load the module dynamically
                    spec = importlib.util.spec_from_file_location(effect_name, file_path)
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    
                    # Check if the module has a process_frame function
                    if hasattr(module, 'process_frame'):
                        self.effects[effect_name] = module
                        if config.DEBUG:
                            print(f"Loaded effect: {effect_name}")
                except Exception as e:
                    if config.DEBUG:
                        print(f"Failed to load effect {effect_name}: {e}")
        
        # Set default effect if available
        if config.DEFAULT_EFFECT in self.effects:
            self.current_effect_name = config.DEFAULT_EFFECT
        elif self.effects:
            self.current_effect_name = list(self.effects.keys())[0]
            
    def get_effect_names(self):
        """Get names of all available effects.
        
        Returns:
            list: Names of available effects
        """
        return list(self.effects.keys())
        
    def get_enabled_effect_names(self):
        """Get names of all enabled effects.
        
        Returns:
            list: Names of available and enabled effects
        """
        return [name for name in self.effects.keys() if name not in self.disabled_effects]
        
    def disable_effect(self, effect_name):
        """Disable an effect without removing it from the system.
        
        Args:
            effect_name (str): Name of the effect to disable
        
        Returns:
            bool: True if the effect was disabled, False otherwise
        """
        if effect_name in self.effects and effect_name not in self.disabled_effects:
            if effect_name == self.current_effect_name:
                # Don't disable the current effect
                return False
                
            self.disabled_effects.add(effect_name)
            if config.DEBUG:
                print(f"Disabled effect: {effect_name}")
            return True
        return False
        
    def enable_effect(self, effect_name):
        """Re-enable a previously disabled effect.
        
        Args:
            effect_name (str): Name of the effect to enable
            
        Returns:
            bool: True if the effect was enabled, False otherwise
        """
        if effect_name in self.effects and effect_name in self.disabled_effects:
            self.disabled_effects.remove(effect_name)
            if config.DEBUG:
                print(f"Enabled effect: {effect_name}")
            return True
        return False
        
    def is_effect_enabled(self, effect_name):
        """Check if an effect is enabled.
        
        Args:
            effect_name (str): Name of the effect to check
            
        Returns:
            bool: True if the effect is enabled, False otherwise
        """
        return effect_name in self.effects and effect_name not in self.disabled_effects
        
    def set_effect(self, effect_name):
        """Set the current effect.
        
        Args:
            effect_name (str): Name of the effect to use
        """
        if effect_name in self.effects:
            # Start transition
            self.next_effect_name = effect_name
            self.transition_start_time = time.time()
            self.transition_progress = 0
            
    def switch_to_random_effect(self):
        """Switch to a random effect different from the current one."""
        available_effects = [name for name in self.effects.keys() 
                             if name != self.current_effect_name and name not in self.disabled_effects]
        if available_effects:
            random_effect = random.choice(available_effects)
            if config.DEBUG:
                print(f"Switching to random effect: {random_effect}")
            self.set_effect(random_effect)
            
    def process_frame(self, frame, audio_info):
        """Process the frame with the current effect.
        
        Args:
            frame (numpy.ndarray): Video frame to process
            audio_info (dict): Audio information including beat detection
            
        Returns:
            numpy.ndarray: Processed frame
        """
        if not self.effects or not self.current_effect_name:
            return frame
            
        # Check if we should auto-switch effects
        current_time = time.time()
        if (config.AUTO_SWITCH_EFFECTS and 
            current_time - self.last_auto_switch_time > config.AUTO_SWITCH_INTERVAL):
            if config.DEBUG:
                print(f"Auto-switching effect after {config.AUTO_SWITCH_INTERVAL} seconds")
            self.last_auto_switch_time = current_time
            self.switch_to_random_effect()
            
        # Handle effect transition
        if self.next_effect_name and self.next_effect_name != self.current_effect_name:
            # Calculate transition progress
            elapsed = current_time - self.transition_start_time
            self.transition_progress = min(1.0, elapsed / config.EFFECT_TRANSITION_TIME)
            
            # Get the processed frames from both effects
            current_processed = self.effects[self.current_effect_name].process_frame(frame.copy(), audio_info)
            next_processed = self.effects[self.next_effect_name].process_frame(frame.copy(), audio_info)
            
            # Blend the frames based on transition progress
            result = cv2.addWeighted(
                current_processed, 
                1.0 - self.transition_progress, 
                next_processed, 
                self.transition_progress, 
                0
            )
            
            # If transition is complete, switch to the next effect
            if self.transition_progress >= 1.0:
                self.current_effect_name = self.next_effect_name
                self.next_effect_name = None
                
            return result
        else:
            # Apply the current effect
            return self.effects[self.current_effect_name].process_frame(frame, audio_info)