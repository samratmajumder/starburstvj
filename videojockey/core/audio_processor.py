"""
Audio processor module for capturing and analyzing audio from the microphone.
Uses PyAudio for capture and librosa for audio analysis.
"""

import threading
import time
import numpy as np
import pyaudio
import librosa
from collections import deque

from videojockey.core import config

class AudioProcessor:
    def __init__(self):
        self.audio = pyaudio.PyAudio()
        self.stream = None
        self.running = False
        self.thread = None
        
        # Beat detection parameters
        self.beat_detected = False
        self.last_beat_time = 0
        self.beat_history = deque(maxlen=20)
        self.energy_history = deque(maxlen=30)  # Store energy history for beat detection
        self.beat_threshold = 1.3  # Energy threshold to detect beats
        
        # Audio analysis
        self.volume = 0
        self.frequency_bands = np.zeros(8)  # Different frequency bands
        self.audio_buffer = np.zeros(config.AUDIO_CHUNK_SIZE)
        
    def start(self):
        """Start audio processing in a separate thread."""
        if self.running:
            return
            
        self.stream = self.audio.open(
            format=self.audio.get_format_from_width(config.AUDIO_FORMAT // 8),
            channels=config.AUDIO_CHANNELS,
            rate=config.AUDIO_SAMPLE_RATE,
            input=True,
            frames_per_buffer=config.AUDIO_CHUNK_SIZE
        )
        
        self.running = True
        self.thread = threading.Thread(target=self._process_audio)
        self.thread.daemon = True
        self.thread.start()
        
    def stop(self):
        """Stop audio processing."""
        self.running = False
        if self.thread:
            self.thread.join(timeout=1.0)
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
            self.stream = None
            
    def _detect_beat(self, energy):
        """Simple beat detection using energy threshold.
        
        Args:
            energy (float): Current audio energy/volume level
            
        Returns:
            bool: True if beat detected, False otherwise
        """
        # Add current energy to history
        self.energy_history.append(energy)
        
        # Need some history before detecting beats
        if len(self.energy_history) < 10:
            return False
        
        # Calculate local average (excluding current sample)
        local_avg = np.mean(list(self.energy_history)[:-1])
        
        # Calculate variance for adaptive threshold
        local_variance = np.std(list(self.energy_history)[:-3])
        
        # Dynamic threshold with noise guard
        current_threshold = max(self.beat_threshold, 1.0 + local_variance * 0.5)
        
        # Beat detected if current energy is significantly higher than local average
        is_beat = energy > local_avg * current_threshold
        
        # Enforce minimum time between beats
        current_time = time.time()
        if is_beat and current_time - self.last_beat_time > config.BEAT_MIN_INTERVAL:
            self.last_beat_time = current_time
            return True
        
        return False
            
    def _process_audio(self):
        """Process audio data from the microphone."""
        while self.running:
            try:
                # Read audio chunk
                data = self.stream.read(config.AUDIO_CHUNK_SIZE, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.int16)
                
                # Normalize
                normalized_data = audio_data.astype(np.float32) / 32768.0
                self.audio_buffer = normalized_data
                
                # Calculate volume (RMS)
                self.volume = np.sqrt(np.mean(normalized_data**2))
                
                # Detect beats using energy level
                self.beat_detected = self._detect_beat(self.volume)
                
                if self.beat_detected:
                    self.beat_history.append(time.time())
                    
                # Basic frequency analysis using FFT
                if len(normalized_data) > 0:
                    fft_data = np.abs(np.fft.rfft(normalized_data))
                    # Divide the FFT into 8 frequency bands
                    bands = np.array_split(fft_data[:len(fft_data)//4], 8)  # Use first quarter for more bass/mid focus
                    self.frequency_bands = np.array([np.mean(band) for band in bands])
                    
            except Exception as e:
                if config.DEBUG:
                    print(f"Audio processing error: {e}")
                time.sleep(0.01)
                
    def get_beat_info(self):
        """Get current beat information.
        
        Returns:
            dict: Beat information including current beat status, volume and frequency bands
        """
        # Calculate BPM if we have enough beat history
        bpm = 0
        if len(self.beat_history) > 4:
            # Calculate average time between beats
            beat_times = list(self.beat_history)
            intervals = [beat_times[i+1] - beat_times[i] for i in range(len(beat_times)-1)]
            if intervals:
                avg_interval = sum(intervals) / len(intervals)
                if avg_interval > 0:
                    bpm = 60.0 / avg_interval
        
        return {
            "beat": self.beat_detected,
            "volume": self.volume,
            "bpm": bpm,
            "frequency_bands": self.frequency_bands,
            "raw_audio": self.audio_buffer
        }