import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import time

def say(words):
    tts = gTTS(text=words, lang='en-au', slow=False)
    tts.save("output.mp3")
    playsound("output.mp3")


def listen():
    r = sr.Recognizer()
    mic_list = sr.Microphone.list_microphone_names()
    print(mic_list)
    mic_index = 0
    mic = sr.Microphone(device_index=mic_index)
    with mic as source:
        r.adjust_for_ambient_noise(source)
        say('LISTENING')
        print('listening....')
        time.sleep(1)
        audio = r.listen(source, timeout=7.0)

    text = r.recognize_google(audio)
    return text