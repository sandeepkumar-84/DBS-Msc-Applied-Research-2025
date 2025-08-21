import subprocess
import sys

packages = [
    "sentence-transformers",
    "sentencepiece",
    "faiss-cpu",
    "transformers",
    "accelerate",
    "bitsandbytes",
    "nltk",
    "bert-score",
    "tf-keras",
    "wordcloud",
    "SpeechRecognition",
    "pyttsx3",
    "pyaudio",
    "requests",
    "beautifulsoup4"    
]

for p in packages:
    subprocess.check_call([sys.executable, "-m", "pip", "install", p])