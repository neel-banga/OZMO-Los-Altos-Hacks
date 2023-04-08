import pyttsx3
import speech_recognition as sr

def say(words):
    engine = pyttsx3.init()
    engine.say(words)
    engine.runAndWait()


def listen():
    r = sr.Recognizer()
    mic_list = sr.Microphone.list_microphone_names()
    mic_index = 0
    mic = sr.Microphone(device_index=mic_index)
    with mic as source:
        r.adjust_for_ambient_noise(source)
        audio = r.listen(source)


    text = r.recognize_google(audio)
    return text