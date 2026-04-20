import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import csv

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

label = input("Enter gesture label: ")

with open("dataset.csv", "a", newline="") as f:
    writer = csv.writer(f)

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

                key = cv2.waitKey(1)

                if key == ord('s'):
                    writer.writerow([label] + landmarks)
                    print("Saved sample")

        cv2.imshow("Collect Data", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()