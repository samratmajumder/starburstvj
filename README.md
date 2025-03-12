# VideoJockey

A Python application for creating trippy visuals for parties. The application uses your MacBook's camera and microphone to generate real-time visual effects that react to audio beats.

## Features

- Real-time video processing from MacBook camera or RTSP stream
- Audio beat detection using the built-in microphone
- 15+ trippy visual effects that react to music beats
- Human segmentation for special effects (background replacement, laser effects, etc.)
- Smooth transitions between effects
- Hardware acceleration using macOS-specific optimizations

## Requirements

- macOS
- Python 3.7+
- Webcam or RTSP video source
- Microphone

## Installation

1. Clone this repository:
```bash
git clone https://github.com/yourusername/videojockey.git
cd videojockey
```

2. Create and activate a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate
```

3. Install the package:
```bash
pip install -e .
```

## Usage

Simply run the application:

```bash
videojockey
```

Or you can run it as a module:

```bash
python -m videojockey
```

## Controls

- Start/Stop: Begins or stops video and audio processing
- Effect Selection: Choose from the available visual effects
- Random Effect: Switch to a random visual effect

## Creating Custom Effects

You can create your own visual effects by adding Python files to the `effects` directory. Each effect should define a `process_frame` function that takes a video frame and audio information as input and returns the processed frame.

Example effect structure:

```python
import cv2
import numpy as np

def process_frame(frame, audio_info):
    """
    Process the frame with your custom effect.
    
    Args:
        frame (numpy.ndarray): Input video frame
        audio_info (dict): Audio information including beat detection
        
    Returns:
        numpy.ndarray: Processed frame
    """
    # Your effect implementation here
    # Example: Invert colors on beat
    if audio_info["beat"]:
        frame = cv2.bitwise_not(frame)
    
    return frame
```

## License

MIT