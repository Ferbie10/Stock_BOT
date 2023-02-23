import speech_recognition as sr
import time
import os
import pyttsx3
import pytz
import subprocess
import datetime
import webbrowser

NMAP_STR = ['run a scan', 'scan this', 'find ports', 'whats running on']
NOTE_STRS = ["make a note", "write this down", "add this"]
TARGET_STRS = ["the target is", "are target is", "set target as"]

# will change speak when called 
def speak(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()
# will get input from the user and convert the speech to text 
def get_audio():
    r = sr.Recognizer()
    with sr.Microphone() as source:
        audio = r.listen(source)
        said = " "
        
        try:
            said = r.recognize_google(audio)
            print(said)
        except Exception as e:
            print('Exception: ' + str(e))
        return said.lower()

def get_target(text):
    target = text.lower()
    return target


def nmap_scan(target):
    if 

def note(text):
    date = datetime.datetime.now()
    file_name = str(date).replace(":", "-") + "-note.txt"
    with open(file_name, "w") as f:
        f.write(text)
    subprocess.Popen(["mousepad %F", file_name])


text = get_audio()


for phrase in NOTE_STRS:
    if phrase in text:
        speak("What would you like me to take note of?")
        note_text = get_audio().lower()
        note(note_text)
        speak("I have made a note of that.")
for phrase in TARGET_STRS:
    if phrase in text:
        set_target = get_audio().lower()
        get_target(set_target)
        speak("I have set the target as" + set_target)
