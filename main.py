import cv2
import numpy as np
from ultralytics import YOLO
from deepface import DeepFace
import os

# Load YOLO Model
yolo_model = YOLO("yolov8n.pt")

# Known Faces Folder
KNOWN_FACES_FOLDER = "known_faces"

# OpenCV Haar Cascades
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)
eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_eye.xml"
)

# Face Detection
def detect_faces(frame):
    gray  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (60, 60)
    )
    return faces

# Attentiveness Check
def check_attentiveness(frame, fx, fy, fw, fh):
   
    # face area
    face_region = frame[fy:fy+fh, fx:fx+fw]
    gray_face   = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)

    # search eye in face region
    eyes = eye_cascade.detectMultiScale(
        gray_face,
        scaleFactor  = 1.1,
        minNeighbors = 5,
        minSize      = (20, 20)
    )

    if len(eyes) >= 2:
        return "Attentive", "Eyes Open"
    elif len(eyes) == 1:
        return "Attentive", "Eyes Detected"
    else:
        return "Inattentive", "Eyes Not Detected"

# Face Recognition
def recognize_face(face_img):
    if not os.path.exists(KNOWN_FACES_FOLDER):
        return "Unknown"
    if not os.listdir(KNOWN_FACES_FOLDER):
        return "Unknown"

    try:
        results = DeepFace.find(
            img_path          = face_img,
            db_path           = KNOWN_FACES_FOLDER,
            model_name        = "VGG-Face",
            enforce_detection = False,
            silent            = True
        )
        if results and not results[0].empty:
            matched_path = results[0].iloc[0]["identity"]
            filename     = os.path.basename(matched_path)
            name         = filename.split("_")[0]
            return name
    except Exception:
        pass

    return "Unknown"

# Main
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam Not Found")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nSystem Started - Press Q to Quit\n")

frame_count = 0
face_names  = {}

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1

    # Object Detection
    yolo_results = yolo_model(frame, verbose=False)[0]

    for box in yolo_results.boxes:
        cls_id       = int(box.cls[0])
        conf         = float(box.conf[0])
        label        = yolo_model.names[cls_id]
        x1,y1,x2,y2 = map(int, box.xyxy[0])

        if conf < 0.40:
            continue

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 200, 255), 2)
        cv2.putText(frame, f"{label} {conf:.0%}",
                    (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                    (0, 200, 255), 1)

    # Face Detection
    faces = detect_faces(frame)

    for i, (fx, fy, fw, fh) in enumerate(faces):

        # Har 10 frame par recognize karo
        if frame_count % 10 == 0 or i not in face_names:
            face_crop    = frame[fy:fy+fh, fx:fx+fw]
            face_names[i] = recognize_face(face_crop)

        name = face_names.get(i, "Unknown")

        # Attentiveness
        status, reason = check_attentiveness(frame, fx, fy, fw, fh)

        # Face box color
        if name != "Unknown":
            box_color = (0, 255, 0)    # Green
        else:
            box_color = (0, 0, 255)    # Red

        # Attentiveness color
        if status == "Attentive":
            att_color = (0, 255, 0)    # Green
        else:
            att_color = (0, 0, 255)    # Red

        # Draw face box
        cv2.rectangle(frame, (fx, fy), (fx+fw, fy+fh), box_color, 2)

        # Name label
        cv2.putText(frame, name,
                    (fx, fy - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                    box_color, 2)

        # Attentiveness label
        cv2.putText(frame, status,
                    (fx, fy + fh + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.60,
                    att_color, 2)

        # Reason label
        cv2.putText(frame, reason,
                    (fx, fy + fh + 42),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.50,
                    (200, 200, 200), 1)

    cv2.imshow("EbroTec AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("\nSystem Stopped")
        break

cap.release()
cv2.destroyAllWindows()
