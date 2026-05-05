
import cv2
import mediapipe as mp
import threading
import time
import os
from gtts import gTTS
from playsound import playsound

# -------- TEXT TO SPEECH (FIXED) -------- #
def speak(text):
    def run():
        try:
            filename = "voice.mp3"
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("Speech error:", e)

    threading.Thread(target=run).start()

# -------- MediaPipe Tasks API -------- #

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)

landmarker = HandLandmarker.create_from_options(options)

# -------- Webcam -------- #
cap = cv2.VideoCapture(0)

sentence = ""
last_letter = ""
last_time = 0
delay = 1.5   # seconds between adding letters

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp)

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            for lm in hand_landmarks:
                x = int(lm.x * frame.shape[1])
                y = int(lm.y * frame.shape[0])
                cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

        # -------- TEMP LETTER -------- #
        letter = "A"

        cv2.putText(frame, f"Letter: {letter}", (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        current_time = time.time()

        # -------- CONTROLLED ADD -------- #
        if letter != last_letter or (current_time - last_time > delay):
            sentence += letter
            last_letter = letter
            last_time = current_time

    # -------- DISPLAY WORD -------- #
    cv2.putText(frame, f"Word: {sentence}", (50, 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("ISL Translator (Fixed Speech)", frame)

    key = cv2.waitKey(1) & 0xFF

    # -------- SPEAK WORD -------- #
    if key == ord('s') and sentence != "":
        print("Speaking:", sentence)
        speak(sentence)
        sentence = ""
        last_letter = ""

    # -------- EXIT -------- #
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

