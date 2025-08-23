import subprocess
import sys
import os
import requests

# set this variable if you want to create directories and upload files from github repo
# else if directories and files are already present, set this variable to False
# or if the files are manually uploaded, set this variable to False
create_dir_and_uploaf_files_flag = True

# 1. install required packages. If already installed, it will skip the installation.
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

# 2. Create necessary directories if they don't exist

if not create_dir_and_uploaf_files_flag:
    print("skipping directory creation and file download as the flag is set to False.")
    sys.exit(0)

directories = ["/content", "/content/Research-Chatbot", "/content/Research-Chatbot/DataCollection/Brochures", "/content/Research-Chatbot/DataCollection/Brochures_text"]
for directory in directories:
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")
    else:
        print(f"Directory already exists: {directory}")  

# 3. download data files from github repo

file_urls = {
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/Intent-Common.json": "/content/Research-Chatbot/Intent-Common.json",
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/LSTM_Training_DataSet.json": "/content/Research-Chatbot/LSTM_Training_DataSet.json",
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/LSTM_Testing_DataSet.json": "/content/Research-Chatbot/LSTM_Testing_DataSet.json",
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/Transformer_Training_DataSet-1.txt": "/content/Research-Chatbot/Transformer_Training_DataSet-1.txt",
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/Transformer_Training_DataSet-2.txt": "/content/Research-Chatbot/Transformer_Training_DataSet-2.txt",
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/Transformer_Test_DataSet.json": "/content/Research-Chatbot/Transformer_Test_DataSet.json",
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/main/data/training-testing-data-files/DBS-Logo-Chat.png": "/content/DBS-Logo-Chat.png"
}
for url, path in file_urls.items():
    response = requests.get(url)
    with open(path, 'wb') as f:
        f.write(response.content)
    print(f"Downloaded {url} to {path}")


