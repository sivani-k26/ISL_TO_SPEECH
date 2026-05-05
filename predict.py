
import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import pickle
import time
import threading
import os
import uuid
from collections import Counter
from gtts import gTTS
from playsound import playsound
import queue

# -------- SPEECH QUEUE -------- #
speech_queue = queue.Queue()

def speech_worker():
    while True:
        text = speech_queue.get()
        try:
            filename = f"voice_{uuid.uuid4().hex}.mp3"
            tts = gTTS(text=text.replace("_", " "), lang='en', slow=False)
            tts.save(filename)
            playsound(filename)
            os.remove(filename)
        except Exception as e:
            print("Speech error:", e)
        speech_queue.task_done()

threading.Thread(target=speech_worker, daemon=True).start()

def speak(text):
    speech_queue.put(text)

# -------- LOAD MODEL -------- #
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------- MEDIAPIPE -------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# -------- VIDEO -------- #
cap = cv2.VideoCapture(0)

# -------- SETTINGS -------- #
prediction_history = []
history_size = 6
confidence_threshold = 0.75

last_spoken = ""
stable_label = ""
last_speak_time = 0
cooldown = 2.5   # 🔥 adjust (2–3 sec recommended)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    data = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:

            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            for lm in hand_landmarks.landmark:
                data.append(lm.x)
                data.append(lm.y)

        # -------- PREDICT -------- #
        prediction = model.predict([data])
        predicted_label = str(prediction[0])

        # -------- SMOOTHING -------- #
        prediction_history.append(predicted_label)
        if len(prediction_history) > history_size:
            prediction_history.pop(0)

        counter = Counter(prediction_history)
        current_label, count = counter.most_common(1)[0]
        confidence = count / len(prediction_history)

        # -------- DISPLAY -------- #
        if confidence > confidence_threshold:
            display_text = f"{current_label} ({confidence:.2f})"
        else:
            display_text = "..."

        cv2.putText(frame, display_text, (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 3)

        current_time = time.time()

        # -------- ANTI-SPAM SPEAK -------- #
        if confidence > confidence_threshold:
            if current_label != stable_label:
                stable_label = current_label

                # speak only if new + cooldown passed
                if (stable_label != last_spoken and 
                    current_time - last_speak_time > cooldown):

                    speak(stable_label)
                    last_spoken = stable_label
                    last_speak_time = current_time

    else:
        # reset when hand disappears
        prediction_history.clear()
        stable_label = ""
        last_spoken = ""

    cv2.imshow("ISL to Speech (No Spam)", frame)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()

