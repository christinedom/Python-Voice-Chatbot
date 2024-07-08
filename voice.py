import sys
import time
import warnings
import openai
import speech_recognition as sr
from EdgeGPT.EdgeUtils import Query
import whisper
import pyttsx3
import ctypes
import os

libc_name = 'msvcrt.dll' if sys.platform == 'win32' else 'libc.so.6'
libc = ctypes.CDLL(libc_name)

tiny_model = whisper.load_model("tiny")

if sys.platform == 'win32':
    engine = pyttsx3.init()

BING_WAKE_WORD = "bing"
GPT_WAKE_WORD = "gpt"

client = openai.OpenAI(api_key="ENTER YOUR API KEY HERE")

r = sr.Recognizer()
source = sr.Microphone()
warnings.filterwarnings("ignore", category=UserWarning, module='whisper.transcribe', lineno=114)

def speak(text):
    if sys.platform == 'win32':
        engine.say(text)
        engine.runAndWait()
    else:
        print("Text-to-speech functionality is not available on this platform.")

def listen_and_transcribe():
    with source as s:
        r.adjust_for_ambient_noise(s, duration=1)
        audio = r.listen(s, timeout=10)
    with open("input.wav", "wb") as f:
        f.write(audio.get_wav_data())
    result = tiny_model.transcribe('input.wav')
    text_input = result['text']
    print(f"Transcription result: {text_input}")
    return text_input

def main():
    print('\nSay Ok Bing or Ok GPT to wake me up. \n')
    while True:
        print("Listening for wake word...")
        wake_word_text = listen_and_transcribe()

        if BING_WAKE_WORD in wake_word_text.lower().strip():
            print("Speak your prompt to Bing.")
            speak('Listening')
            prompt_text = listen_and_transcribe()
            if prompt_text.strip():
                print('User: ' + prompt_text)
                try:
                    output = Query(prompt_text)
                    print('Bing: ' + str(output))
                    speak(str(output))
                except Exception as e:
                    print("Error querying Bing: ", e)
            else:
                print("No prompt detected. Returning to wake word listening.")
                speak("No prompt detected. Returning to wake word listening.")

        elif GPT_WAKE_WORD in wake_word_text.lower().strip():
            print("Speak your prompt to GPT 3.5 Turbo.")
            speak('Listening')
            prompt_text = listen_and_transcribe()
            if prompt_text.strip():
                print('User: ' + prompt_text)
                try:
                    response = client.chat.completions.create(
                        model="gpt-3.5-turbo",
                        messages=[
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": prompt_text},
                        ],
                        temperature=0.5,
                        max_tokens=150,
                        top_p=1,
                        frequency_penalty=0,
                        presence_penalty=0,
                        n=1,
                    )
                    bot_response = response.choices[0].message.content
                    print("GPT 3.5 Turbo:", bot_response)
                    speak(bot_response)
                except Exception as e:
                    print("Error querying GPT-3.5 Turbo: ", e)
            else:
                print("No prompt detected. Returning to wake word listening.")
                speak("No prompt detected. Returning to wake word listening.")

        else:
            print("No valid wake word detected.")

if __name__ == '__main__':
    main()
