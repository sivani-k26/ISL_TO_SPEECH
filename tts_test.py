from gtts import gTTS
import os

def speak(text):
    tts = gTTS(text=text, lang='en')
    tts.save("voice.mp3")
    os.system("start voice.mp3")  # Windows