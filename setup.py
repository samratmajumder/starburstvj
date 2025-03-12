from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="videojockey",
    version="0.1.0",
    author="Your Name",
    author_email="your.email@example.com",
    description="A Python application for creating trippy visuals for parties",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/videojockey",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: MacOS",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "opencv-python>=4.5.0",
        "opencv-contrib-python>=4.5.0",
        "pyaudio>=0.2.11",
        "librosa>=0.8.0",
        "mediapipe>=0.8.9",
        "pygame>=2.0.0",
        "PyQt5>=5.15.0",
    ],
    entry_points={
        "console_scripts": [
            "videojockey=videojockey:main",
        ],
    },
)