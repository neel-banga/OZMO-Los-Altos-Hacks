import speech_recognition as sr
from gtts import gTTS
from playsound import playsound

def say(words):
    tts = gTTS(text=words, lang='en')
    tts.save("output.mp3")
    playsound("output.mp3")

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