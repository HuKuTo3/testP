import cv2
from ultralytics import YOLO
from deepface import DeepFace
import json
import numpy as np

def numpy_to_python(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj

model = YOLO("yolov8l-face.pt")
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open video.")
    exit()

frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    faces = model(frame)

    for face in faces[0].boxes:
        x1, y1, x2, y2 = map(int, face.xyxy[0])
        confidence = face.conf.item()
        face_roi = frame[y1:y2, x1:x2]
        result = DeepFace.analyze(face_roi, actions=['emotion'], enforce_detection=False)
        # emotion = result[0]['dominant_emotion']
        if result[0]:
            serializable_result = json.loads(json.dumps(result[0], default=numpy_to_python))
            with open('emotion_result.json', 'w') as json_file:
                json.dump(serializable_result, json_file, indent=4)

        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f"{confidence:.2f}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLO Face Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
