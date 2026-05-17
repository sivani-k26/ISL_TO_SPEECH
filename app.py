import cv2
import mediapipe as mp
import time
import pickle
from gtts import gTTS
from playsound import playsound
import os
import threading
from collections import deque, Counter
import speech_recognition as sr
import sounddevice as sd
import numpy as np
import io
import wave

# -------- LOAD MODEL -------- #
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# -------- THEME COLORS (BGR) -------- #
PINK      = (180, 105, 255)
LAVENDER  = (230, 180, 255)
PURPLE    = (160,  50, 180)
MAGENTA   = (200,  50, 220)
WHITE     = (255, 255, 255)
DARK      = ( 20,   5,  25)
BAR_BG    = ( 50,  20,  60)
BAR_FILL  = (200,  80, 240)
GREEN     = ( 50, 200, 100)

# -------- TEXT TO SPEECH -------- #
def speak(text):
    def _speak():
        try:
            print("Speaking:", text)
            filename = f"voice_{int(time.time()*1000)}.mp3"
            tts = gTTS(text=text, lang='en', slow=False)
            tts.save(filename)
            playsound(filename)
            time.sleep(0.3)
            os.remove(filename)
        except Exception as e:
            print("Speech error:", e)
    threading.Thread(target=_speak, daemon=True).start()

# -------- SPEECH TO TEXT -------- #
recognizer      = sr.Recognizer()
heard_text      = ""
is_listening    = False

def listen():
    global heard_text, is_listening
    is_listening = True
    heard_text   = "Listening..."
    try:
        samplerate = 16000
        duration   = 8
        recording  = sd.rec(int(duration * samplerate),
                            samplerate=samplerate,
                            channels=1,
                            dtype='int16')
        sd.wait()

        # Convert to wav in memory
        buffer = io.BytesIO()
        with wave.open(buffer, 'wb') as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(samplerate)
            wf.writeframes(recording.tobytes())
        buffer.seek(0)

        # Recognise
        with sr.AudioFile(buffer) as source:
            audio = recognizer.record(source)
        heard_text = recognizer.recognize_google(audio)
        print("Heard:", heard_text)

    except sr.UnknownValueError:
        heard_text = "Could not understand"
    except Exception as e:
        heard_text = f"Error: {e}"
        print("Listen error:", e)
    is_listening = False
# -------- HELPERS -------- #
def draw_rounded_rect(img, x1, y1, x2, y2, color, alpha=0.55, radius=12):
    overlay = img.copy()
    cv2.rectangle(overlay, (x1 + radius, y1), (x2 - radius, y2), color, -1)
    cv2.rectangle(overlay, (x1, y1 + radius), (x2, y2 - radius), color, -1)
    for cx, cy in [(x1+radius, y1+radius), (x2-radius, y1+radius),
                   (x1+radius, y2-radius), (x2-radius, y2-radius)]:
        cv2.circle(overlay, (cx, cy), radius, color, -1)
    cv2.addWeighted(overlay, alpha, img, 1 - alpha, 0, img)

def draw_pill_bar(img, x, y, w, h, progress, bg_color, fill_color):
    r = h // 2
    cv2.rectangle(img, (x + r, y), (x + w - r, y + h), bg_color, -1)
    cv2.circle(img, (x + r, y + r), r, bg_color, -1)
    cv2.circle(img, (x + w - r, y + r), r, bg_color, -1)
    fill_w = max(int((w - 2*r) * progress), 0)
    if fill_w > 0:
        cv2.rectangle(img, (x + r, y), (x + r + fill_w, y + h), fill_color, -1)
        cv2.circle(img, (x + r, y + r), r, fill_color, -1)
        cv2.circle(img, (x + r + fill_w, y + r), r, fill_color, -1)

def put_text_shadow(img, text, pos, scale, color, thickness=2):
    cv2.putText(img, text, (pos[0]+2, pos[1]+2),
                cv2.FONT_HERSHEY_DUPLEX, scale, DARK, thickness + 1, cv2.LINE_AA)
    cv2.putText(img, text, pos,
                cv2.FONT_HERSHEY_DUPLEX, scale, color, thickness, cv2.LINE_AA)

# -------- MEDIAPIPE SETUP -------- #
BaseOptions           = mp.tasks.BaseOptions
HandLandmarker        = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
VisionRunningMode     = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(model_asset_path="hand_landmarker.task"),
    running_mode=VisionRunningMode.VIDEO,
    num_hands=1
)
landmarker = HandLandmarker.create_from_options(options)

HAND_CONNECTIONS = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17)
]

# -------- WEBCAM -------- #
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Camera index 0 failed, trying index 1...")
    cap = cv2.VideoCapture(1)
if not cap.isOpened():
    print("ERROR: Could not open any camera.")
    exit()

# -------- STATE -------- #
sentence          = ""
cleaned_sentence  = ""
last_spoken_word  = ""
speak_delay       = 2
hold_start_time   = None
current_candidate = ""
prediction_buffer = deque(maxlen=15)

while True:
    ret, frame = cap.read()
    if not ret:
        print("ERROR: Failed to read frame.")
        break

    frame = cv2.flip(frame, 1)
    H, W  = frame.shape[:2]

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image  = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    timestamp = int(time.time() * 1000)
    result    = landmarker.detect_for_video(mp_image, timestamp)
    current_time = time.time()

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:

            # ---- Landmark dots ----
            for lm in hand_landmarks:
                x = int(lm.x * W)
                y = int(lm.y * H)
                cv2.circle(frame, (x, y), 8, PURPLE,   -1)
                cv2.circle(frame, (x, y), 5, PINK,     -1)
                cv2.circle(frame, (x, y), 2, LAVENDER, -1)

            # ---- Connections ----
            for conn in HAND_CONNECTIONS:
                a, b = conn
                ax = int(hand_landmarks[a].x * W)
                ay = int(hand_landmarks[a].y * H)
                bx = int(hand_landmarks[b].x * W)
                by = int(hand_landmarks[b].y * H)
                cv2.line(frame, (ax, ay), (bx, by), MAGENTA, 1, cv2.LINE_AA)

            # ---- Extract + Normalize landmarks ----
            wrist_x = hand_landmarks[0].x
            wrist_y = hand_landmarks[0].y

            landmarks = []
            for lm in hand_landmarks:
                landmarks.extend([
                    lm.x - wrist_x,
                    lm.y - wrist_y,
                ])

            # ---- Majority voting + confidence threshold ----
            proba      = model.predict_proba([landmarks])[0]
            confidence = max(proba)
            raw_word   = model.classes_[proba.argmax()].upper()

            if confidence >= 0.7:
                prediction_buffer.append(raw_word)

            word = Counter(prediction_buffer).most_common(1)[0][0] if prediction_buffer else "..."

            # ---- Confidence indicator ----
            conf_color = PINK if confidence >= 0.7 else LAVENDER

            # ---- Hold detection ----
            if word == current_candidate and word != "...":
                held = current_time - hold_start_time
                prog = min(held / speak_delay, 1.0)

                draw_rounded_rect(frame, 30, 20, 280, 75, DARK, alpha=0.6)
                put_text_shadow(frame, word, (45, 57), 0.95, PINK, 2)
                put_text_shadow(frame, f"conf: {confidence:.0%}", (45, 72), 0.38, conf_color, 1)

                draw_rounded_rect(frame, 30, 85, 280, 120, DARK, alpha=0.55)
                draw_pill_bar(frame, 44, 97, 222, 14, prog, BAR_BG, BAR_FILL)
                put_text_shadow(frame, "hold...", (44, 113), 0.38, LAVENDER, 1)

                if held >= speak_delay and word != last_spoken_word:
                    speak(word)
                    sentence        += word + " "
                    cleaned_sentence = ""
                    last_spoken_word = word

            else:
                current_candidate = word
                hold_start_time   = current_time
                last_spoken_word  = ""

                if word != "...":
                    draw_rounded_rect(frame, 30, 20, 280, 75, DARK, alpha=0.6)
                    put_text_shadow(frame, word, (45, 57), 0.95, LAVENDER, 2)
                    put_text_shadow(frame, f"conf: {confidence:.0%}", (45, 72), 0.38, conf_color, 1)

    else:
        current_candidate = ""
        hold_start_time   = None
        last_spoken_word  = ""
        prediction_buffer.clear()

    # ---- Sentence panel (signed words) ----
    draw_rounded_rect(frame, 20, H - 120, W - 20, H - 70, DARK, alpha=0.65)
    put_text_shadow(frame, "YOU:", (36, H - 100), 0.45, LAVENDER, 1)
    disp  = sentence if sentence.strip() else "--- start signing ---"
    color = WHITE if sentence.strip() else LAVENDER
    if len(disp) > 52:
        disp = "..." + disp[-51:]
    put_text_shadow(frame, disp, (36, H - 80), 0.65, color, 1)

    # ---- Heard text panel (speech to text) ----
    draw_rounded_rect(frame, 20, H - 65, W - 20, H - 15, DARK, alpha=0.65)
    heard_color = GREEN if heard_text and heard_text not in ["Listening...", "No speech detected", "Could not understand"] else LAVENDER
    listen_label = "THEM:" if not is_listening else "LISTENING..."
    put_text_shadow(frame, listen_label, (36, H - 45), 0.45, GREEN if is_listening else LAVENDER, 1)
    heard_disp = heard_text if heard_text else "--- press R to listen ---"
    if len(heard_disp) > 52:
        heard_disp = "..." + heard_disp[-51:]
    put_text_shadow(frame, heard_disp, (36, H - 25), 0.65, heard_color, 1)

    # ---- Keybind strip ----
    draw_rounded_rect(frame, W - 270, 15, W - 15, 75, DARK, alpha=0.5)
    put_text_shadow(frame, "[S]speak [R]listen [C]clear [Q]quit", (W - 263, 38), 0.30, LAVENDER, 1)
    put_text_shadow(frame, "ISL Translator", (W - 263, 62), 0.4, PINK, 1)

    cv2.imshow("ISL Translator", frame)
    key = cv2.waitKey(1) & 0xFF

    # ---- S: speak signed sentence ----
    if key == ord('s') and sentence.strip():
        def speak_clean():
            global cleaned_sentence
            try:
                import google.generativeai as genai
                genai.configure(api_key="AIzaSyDR3OdS3sH8r7ylXFgj7KhijS3pzpd-ouc")
                model_llm = genai.GenerativeModel("gemini-2.0-flash")
                response  = model_llm.generate_content(
                    f"Convert this sign language word sequence to natural English. Return only the result, nothing else: {sentence}"
                )
                clean = response.text.strip()
                cleaned_sentence = clean
                print(f"Original: {sentence}")
                print(f"Cleaned:  {clean}")
                speak(clean)
            except Exception as e:
                print("LLM error:", e)
                speak(sentence)
        threading.Thread(target=speak_clean, daemon=True).start()

    # ---- R: listen for speech ----
    if key == ord('r') and not is_listening:
        threading.Thread(target=listen, daemon=True).start()

    # ---- C: clear everything ----
    if key == ord('c'):
        sentence          = ""
        cleaned_sentence  = ""
        heard_text        = ""
        last_spoken_word  = ""
        current_candidate = ""
        prediction_buffer.clear()

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()