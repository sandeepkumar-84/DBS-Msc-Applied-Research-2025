import json
import random
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression

intent_file_path = "/content/Research-Chatbot/Intent-Common.json"
with open(intent_file_path, "r") as f:
    intent_data = json.load(f)

X_texts = []
y_labels = []
responses = {}


for item in intent_data["intents"]:
    tag = item["tag"]
    responses[tag] = item["responses"]
    for p in item["patterns"]:
        X_texts.append(p)
        y_labels.append(tag)

# load small embedding model (fast)
embedder = SentenceTransformer("sentence-transformers/paraphrase-MiniLM-L3-v2")
X_emb = embedder.encode(X_texts)

clf = LogisticRegression(max_iter=1000)
clf.fit(X_emb, y_labels)

def detect_intent_and_respond(question):
    emb = embedder.encode([question])
    predicted_tag = clf.predict(emb)[0]
    # choose smart response
    return random.choice(responses[predicted_tag]), predicted_tag


'''
greeting_ex_response = detect_intent_and_respond("Hello! How are you?")

print(f"Detected Intent: {greeting_ex_response[1]}")
print(f"Response: {greeting_ex_response[0]}")
'''