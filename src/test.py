import cv2
import numpy as np
from recognizer import ImageEncodePipeline
from PIL import Image
from utils.helper import faceEmbedder, who_is_it

encoder = ImageEncodePipeline("models/YOLOV11_Face/yolov11m-face.pt", "models/Landmark_predictor/shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Try with DirectShow backend

db = {
    "jolie": encoder.encode(Image.open("test_data/jolie.jpg").convert("RGB")),
    "tk": encoder.encode(Image.open("test_data/tk.jpg").convert("RGB")),
}

if not cap.isOpened():
    print("Cannot open camera")
    exit()
else:
    print("Camera is opened")

while cap.isOpened():
    ret, frame = cap.read()

    if not ret or frame is None:
        print("Can't receive frame (stream end?). Exiting ...")
        break

    # Process the frame only if it's valid
    en_vec = encoder.encode(Image.fromarray(frame).convert("RGB"))
    if en_vec is not None:
        name = who_is_it(db, en_vec)
        if name:
            print(name)
            break

    # Flip and display the frame
    frame = cv2.flip(frame, 1)
    cv2.imshow('frame', frame)

    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
