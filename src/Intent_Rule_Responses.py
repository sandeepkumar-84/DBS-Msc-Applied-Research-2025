import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import requests

intent_file_path = "/content/Research-Chatbot/Intent-Common.json"
intent_github_path = "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/Intent-Common.json"


try:
    with open(intent_file_path, "r") as f:
        intent_data = json.load(f)
        print(f"Loaded intent data from local file - {intent_file_path}")
except FileNotFoundError:
    try:
        print(f"local file not found. fetching intent data from GitHub - {intent_github_path}")         
        response = requests.get(intent_github_path)
        intent_data = response.json()
        print("loaded intent data sucessfully from GitHub")
    except Exception as e:
        print(f"rrror loading intent data: {e}")


print(f"Total intents loaded: {len(intent_data['intents'])}")

X_dbs_intent_texts = []
y_dbs_intent_labels = []
responses = {}


for item in intent_data["intents"]:
    tag = item["tag"]
    responses[tag] = item["responses"]
    for p in item["patterns"]:
        X_dbs_intent_texts.append(p)
        y_dbs_intent_labels.append(tag)

embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
X_emb = embedder.encode(X_dbs_intent_texts)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_emb, y_dbs_intent_labels)

def detect_intent_and_respond(question):
    emb = embedder.encode([question])
    predicted_tag = clf.predict(emb)[0]
    return random.choice(responses[predicted_tag]), predicted_tag

''' uncomment below to test this file independently ''' 
'''
greeting_ex_response = detect_intent_and_respond("Hello! How are you?")

print(f"Detected Intent: {greeting_ex_response[1]}")
print(f"Response: {greeting_ex_response[0]}")
'''
