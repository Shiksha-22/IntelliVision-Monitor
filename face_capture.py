import cv2
import os

print("\n FACE CAPTURING \n")
name = input("Enter Your Name: ").strip()

if not name:
    print("No Name Was Given")
    exit()

os.makedirs("known_faces", exist_ok=True)

cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Webcam Not Found")
    exit()

cap.set(cv2.CAP_PROP_FRAME_WIDTH,  640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

instructions = [
    "Look at the Camera",
    "Turn Left",
    "Turn Right",
    "Look Up",
    "Look Down",
]

total = len(instructions)
count = 0
print(instructions)

while count < total:
    ret, frame = cap.read()
    if not ret:
        break
    

    h, w = frame.shape[:2]

   
    cv2.imshow("Face Capture " + name, frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord('p'):
        filename = os.path.join("known_faces", f"{name}_{count + 1}.jpg")
        cv2.imwrite(filename, frame)
        print(f"  Saved → {filename}")
        count += 1

        # Flash effect
        flash = frame.copy()
        cv2.rectangle(flash, (0, 0), (w, h), (0, 255, 0), 14)
        cv2.imshow("Face Capture " + name, flash)
        cv2.waitKey(300)

    elif key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

print(f"\n{count} Photos are Saved in the Folder")