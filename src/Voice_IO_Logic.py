import speech_recognition as sr
import pyttsx3


engine = pyttsx3.init()

'''
def speak(text):
    engine = pyttsx3.init() 
    engine.say(text)
    engine.runAndWait()
    engine.stop()
    del engine
'''
    
def listen():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        audio = r.listen(source)
    try:
        query = r.recognize_google(audio)
        print("User (voice):", query)
        return query
    except Exception as e:
        print("Could not understand audio")
        return ""