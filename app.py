import cv2
import mediapipe as mp
import pyttsx3

# Initialize speech engine (offline)
engine = pyttsx3.init()

def speak(text):
    engine.say(text)
    engine.runAndWait()

# -------- MediaPipe Latest Tasks API Setup -------- #

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.IMAGE,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# -------- Open Webcam -------- #

cap = cv2.VideoCapture(0)
sentence = ""

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    result = landmarker.detect(mp_image)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        letter = "A"   # temporary fixed letter
        cv2.putText(frame, letter, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2)

        sentence += letter

    cv2.imshow("ISL Translator - Latest Version", frame)

    key = cv2.waitKey(1)

    if key == ord('s'):
        speak(sentence)
        sentence = ""

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()