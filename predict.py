import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import pickle

# Load trained model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# For smoothing predictions
predictions = []

# Load MediaPipe model
model_path = "hand_landmarker.task"

BaseOptions = python.BaseOptions
HandLandmarker = vision.HandLandmarker
HandLandmarkerOptions = vision.HandLandmarkerOptions
VisionRunningMode = vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.VIDEO
)

landmarker = HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)
frame_id = 0

while True:
    ret, frame = cap.read()
    frame_id += 1

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

    result = landmarker.detect_for_video(mp_image, frame_id)

    if result.hand_landmarks:
        for hand in result.hand_landmarks:
            landmarks = []

            for lm in hand:
                landmarks.append(lm.x)
                landmarks.append(lm.y)

            # Raw prediction
            pred = model.predict([landmarks])[0]

            # Smooth predictions
            predictions.append(pred)
            if len(predictions) > 10:
                predictions.pop(0)

            prediction = max(set(predictions), key=predictions.count)

            # Normalize label (fix HELLO vs Hello)
            prediction = prediction.upper()

            # Display on screen
            cv2.putText(frame, prediction, (50, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 255, 0), 2)

    cv2.imshow("ISL Prediction", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()