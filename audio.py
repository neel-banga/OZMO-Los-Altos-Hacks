import speech_recognition as sr
from gtts import gTTS
from playsound import playsound
import pyaudio
import wave
import whisper
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


'''def listen():
    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    RECORD_SECONDS = 10  # Set the desired recording duration in seconds
    FILE_NAME = "output.wav"  # Specify the desired name for the output WAV file

    p = pyaudio.PyAudio()

    # Open microphone stream
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNK)

    print("Recording started...")

    frames = []

    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Recording completed.")

    # Stop and close the microphone stream
    stream.stop_stream()
    stream.close()
    p.terminate()

    # Save the recorded audio to a WAV file
    wf = wave.open(FILE_NAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

    print(f"Audio saved to '{FILE_NAME}'")

    model = whisper.load_model("base")
    result = model.transcribe("output.wav")
    return result["text"]

model = whisper.load_model("base")
result = model.transcribe("output.mp3")'''
