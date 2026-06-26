# IntelliVision-Monitor
An intelligent computer vision application that combines face recognition, object detection, and attentiveness analysis in real time. Built with Python, OpenCV, YOLO, and DeepFace for live video processing and AI-based behavioral analysis.


## Features

- Face Registration
- Face Recognition
- Object Detection
- Attentiveness Detection
- Live Webcam Support


## Technologies Used

- Python
- OpenCV
- DeepFace
- YOLO
- NumPy


## Project Workflow

### Step 1: Register a Face

Run the following command:

```bash
python face_capture.py
```

- Enter the person's name.
- Face images will be captured through the webcam.
- Images will be saved in the `known_faces` folder.


### Step 2: Start Detection

Run:

```bash
python main.py
```

The application will:

- Detect faces
- Recognize registered persons
- Detect objects
- Analyze attentiveness
- Display the person's name and attentiveness status


## Installation

### Clone the Repository

```bash
git clone https://github.com/Shiksha-22/IntelliVision-Monitor.git
```

### Move to the Project Folder

```bash
cd IntelliVision-Monitor
```

### Install Required Libraries

```bash
pip install -r requirements.txt
```


## Run the Project

### Register Face

```bash
python face_capture.py
```

### Start Monitoring

```bash
python main.py
```


## Project Structure

```
IntelliVision-Monitor/
│
├── known_faces/
├── face_capture.py
├── main.py
├── requirements.txt
└── README.md
```
