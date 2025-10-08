# Simple Fish Counter

This project is a **real-time fish counting application** that utilizes the **YOLO** object detection model to track and count fish crossing a defined line in a live video feed. The application provides **memory management**, **status reports**, and allows for **real-time fish tracking and counting**.

## Features

- **Real-time fish tracking and counting** using the YOLO model.
- **Memory management**: Automatically cleans up old tracks to prevent memory overload.
- **Status reporting**: Displays current fish count, FPS, and memory usage every 10 minutes.
- **Line configuration**: Draw a counting line in the video to count fish crossing in or out.
- **Customizable tracking**: Supports custom model paths, camera input, and runtime limits.

## Requirements

- Python 3.6+
- **OpenCV**: For handling video input/output and drawing operations.
- **Ultralytics YOLO**: For fish detection (use YOLO11n model).
- **psutil**: For system resource monitoring (memory usage, CPU).
- **numpy**: For handling arrays and numerical operations.

### Install Dependencies

You can install the required dependencies by running:

```bash
pip install opencv-python ultralytics psutil numpy
```

## Setup
-**Clone the repository:**
```bash
git clone https://github.com/yourusername/fish-counter.git
```

## Configuration
The program allows you to draw a counting line interactively in the video feed. The line is used to detect fish crossing in or out. Here’s what you need to do:

- Line Positioning: The program will guide you through drawing the line. This line should be placed where fish will most likely pass, ensuring no fish is missed and that the line is as far from the fish exit as possible.

- Saved Configurations: You will be prompted to load a previously saved line configuration, or you can choose to draw a new line if you're using the system for the first time.
## Running the Application
## 🚀 Run Instructions

First, configure the environment variable to set the Dashboard server address. This tells the program where to send the fish counting results.

```bash
export FISHDASH_SERVER="http://100.76.99.48:8000"
python3 enhanced_fish_counter_w9_v1.py
```
**After starting, open the Dashboard link in your browser (http://100.76.99.48:8000) to see the real-time statistics and updates as the fish are detected.**

The main entry point of the program is defined at the bottom of the script:
```bash
if __name__ == "__main__":
    try:
        run_simple_fish_counter(
            model_path="best.pt", 
            camera_index=1,
            save_output=True,
            max_runtime_hours=None  # unlimited runtime
        )

```
```bash
python enhanced_fish_counter_w9.py
```
- camera_index: Please change this value according to the camera device you are using.

  0 usually refers to the default laptop/webcam camera.

  1, 2, ... refer to external USB or other connected cameras.

- model_path: Path to the trained YOLO model (default "best.pt").

- save_output: If set to True, saves the counting results to a video file.

- max_runtime_hours: Allows you to limit the runtime in hours (set to None for unlimited runtime).
## Example of Output:
- Once the application starts, you will see the following status:
```bash
ENHANCED FISH COUNTER WITH MEMORY MANAGEMENT
--------------------------------------------------
Camera initialized: 640x480 @ 100fps
Initial memory usage: 311.2MB
======================================================
INTERACTIVE LINE POSITIONING FOR LIVE CAMERA
======================================================
Loaded saved line configuration.
Use saved line position? (y/n): n

Instructions:
- Click and drag to draw the counting line.
- Press 's' to save and continue.
- Press 'r' to reset the line.
- Press 'q' to quit.

Line set: vertical from (230, 93) to (147, 458)
Line configuration saved to: live_camera_line_config.json
Recording enabled: fish_output_20250924_235657.mp4 @ 100fps

Starting live counting...
Press 'q' to quit, 's' for immediate status report
--------------------------------------------------
```
## Example Status Output
```bash
STATUS REPORT - 2023-09-01 14:23:45
--------------------------------------------------
Runtime: 1:30:45
Total Frames: 2000
Current FPS: 25.3 | Average FPS: 24.1
Fish Count - IN: 25 | OUT: 15 | Net: 10
Active Tracks: 5 | Total Counted IDs: 100
Memory Usage: 120.5MB
--------------------------------------------------
```
