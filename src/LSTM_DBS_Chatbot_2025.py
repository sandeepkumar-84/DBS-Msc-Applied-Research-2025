import os
import json
import time
import torch
import numpy as np
import torch.nn as nn
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from nltk.tokenize import word_tokenize
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
from bert_score import score as bert_score
import torch.nn.functional as F

# for visulizations
from wordcloud import WordCloud
import matplotlib.pyplot as plt_lstm
import matplotlib
from collections import Counter

import requests

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')

# local file path where training and testing data is present
local_research_path = "/content/Research-Chatbot"

# create the directory if it do not exist
os.makedirs(local_research_path, exist_ok=True)

# exact path defined
training_path_files = [   
    os.path.join(local_research_path, "LSTM_Training_DataSet.json")
]
testing_path_files = [   
    os.path.join(local_research_path, "LSTM_Testing_DataSet.json")
]

# github path defined
github_url_training  = "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/LSTM_Training_DataSet.json"

qa_pairs_combined_raw = []
load_flag = False

# function to load the training data from local files
def load_local_files(file_list):
    global load_flag
    for file_name in file_list:
        if file_name.endswith('.json'):
            try:
                with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
                    qa_list = json.load(f)
                    if isinstance(qa_list, list):
                        qa_pairs_combined_raw.extend(qa_list)
                        print(f"QA pair loaded {len(qa_list)} from file: {file_name}")
                        load_flag = True
                    else:
                        print(f"skipping {file_name}: not a list of QA pairs.")
            except json.JSONDecodeError:
                print(f"error decoding json in file: {file_name}")
            except Exception as e:
                print(f"error reading {file_name}: {e}")
        else:
            print(f"skipping non-JSON file: {file_name}")

# function to load the training data from github if local file is not found
def load_github_file(file_url):
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            qa_list = response.json()
            if isinstance(qa_list, list):
                qa_pairs_combined_raw.extend(qa_list)
                print(f"QA Pairs Loaded {len(qa_list)} from GitHub file: {file_url}")
                return True
            else:
                print(f"Skipping GitHub file {file_url}: not a list of QA pairs.")
        else:
            print(f"GitHub file not found: {file_url} (status code {response.status_code})")
    except Exception as e:
        print(f"Error reading GitHub file {file_url}: {e}")
    return False

# control statement to load data first from local files, if not found then from github
load_local_files(training_path_files)
if not load_flag:
    load_github_file(github_url_training)
  
qa_pairs = qa_pairs_combined_raw

print(f"total QA pairs loaded: {len(qa_pairs)}")

all_text = " ".join([q["query"].lower() for q in qa_pairs_combined_raw])


#below plot shows the wordcloud representation of the training data. The font size and the dark font color represent the higher frequency of that wordin the training data."""

wordcloud_alltext = WordCloud(width=800, height=400, background_color='white',
                      colormap='viridis', max_words=100).generate(all_text)


plt_lstm.figure(figsize=(10, 5))
plt_lstm.imshow(wordcloud_alltext, interpolation='bilinear')
plt_lstm.axis('off')
plt_lstm.title("Word Cloud of Training Questions", fontsize=16)
plt_lstm.show()

all_words = all_text.split()
word_freq = Counter(all_words)
top_words = word_freq.most_common(10)
words, counts = zip(*top_words)
plt_lstm.figure(figsize=(10, 6))
plt_lstm.bar(words, counts, color='lightgrey', edgecolor='black')
plt_lstm.title("Top 10 Most Frequent Words in Vocabulary", fontsize=16)
plt_lstm.xlabel("Words", fontsize=14)
plt_lstm.ylabel("Frequency", fontsize=14)
plt_lstm.xticks(rotation=45)
plt_lstm.tight_layout()
plt_lstm.show()


# defines the size of word embedding vectors
embedding_dim_lstm = 100
# number of hidden units in the hiden layers
hidden_dim_lstm = 128
#max length of the input sequence
max_len = 20

# dataset preperation where questions and anwers variables are populated using for in loop
questions = [q["query"].lower() for q in qa_pairs if "query" in q]
answers = [q["expected_answer"] for q in qa_pairs if "expected_answer" in q]

# next step is to build the vocabulary using the word_tokenize nltk function for all the words in the question.
all_words = [word for q in questions for word in word_tokenize(q)]

# word frequency is calculated
word_freq = Counter(all_words)

# unique id starting from 2 is assigned to each word while 0 and 1 are kept for padding and unknown words
vocab = {word: i + 2 for i, (word, _) in enumerate(word_freq.items())}
vocab["<pad>"] = 0
vocab["<unk>"] = 1
vocab_size = len(vocab)
print(f"Vocabulary size: {vocab_size}")

# function to return the padded tensors of the tokens.
def encode_question(q):
    tokens = word_tokenize(q.lower())
    idxs = [vocab.get(w, 1) for w in tokens]
    padded = idxs[:max_len] + [0] * (max_len - len(idxs))
    return torch.tensor(padded)

wordcloud_vocab = WordCloud(width=800, height=400, background_color='white',
                      colormap='viridis', max_words=100).generate(" ".join(vocab.keys()))

plt_lstm.figure(figsize=(10, 5))
plt_lstm.imshow(wordcloud_vocab, interpolation='bilinear')
plt_lstm.axis('off')
plt_lstm.title("Word Cloud of vocab", fontsize=16)
plt_lstm.show()


# torch library is utilized below.
class LSTMEncoderClass(nn.Module):
    def __init__(self, vocab_size, embedding_dim_lstm, hidden_dim_lstm):
        super(LSTMEncoderClass, self).__init__() # calling the base class constructor
        self.embedding = nn.Embedding(vocab_size, embedding_dim_lstm) # embedding layer to map word ids to vectors
        self.lstm = nn.LSTM(embedding_dim_lstm, hidden_dim_lstm, batch_first=True) # LSTM layer to catch semantic patterns
        self.fc = nn.Linear(hidden_dim_lstm, hidden_dim_lstm) # to transform hidden state fully connected layer is returned.

    def forward(self, input_ids):
        embedded = self.embedding(input_ids) # tokens are converted to embeddings
        _, (hidden, _) = self.lstm(embedded) # fetches final hidden state from LSTM
        return self.fc(hidden[-1])  # return the last hiddeb layer


encoded_questions = torch.stack([encode_question(q) for q in questions])

model = LSTMEncoderClass(vocab_size=vocab_size, embedding_dim_lstm=embedding_dim_lstm, hidden_dim_lstm=hidden_dim_lstm)
# Set the model to evaluation mode
model.eval()

# generate vector embeddings as float tensors for all encoded questions without tracking gradients
with torch.no_grad():
    question_vecs_tensor = model(encoded_questions).float()

def lstm_chatbot():
    print("DBS Specific LSTM Chatbot. Type 'exit' to quit.")
    while True:
        user_input = input("You: ").lower().strip() # read the user input and lowercae and trim them for any spaces for ideal comparison later
        if user_input in ['exit', 'quit']:
            print("DBS Chatbot: Goodbye!")
            break

        input_vec = encode_question(user_input).unsqueeze(0) # use above encode function to encode the user input
        with torch.no_grad():
            query_vec = model(input_vec).float() # generate embedding of the user query
            similarities = F.cosine_similarity(query_vec, question_vecs_tensor) #compare using cosine similarity
            best_idx = torch.argmax(similarities).item() # get the index of the most similar question
            best_score = similarities[best_idx].item() # get the smilarity score of the best match

        if best_score < 0.6: # if the score is less than standard 0.6 then user is prompted with appropriate message otherwise the generated response if returned
            print("DBS Chatbot: Sorry, I don't understand.")
        else:
            print("DBS Chatbot:", answers[best_idx])



# execute the function to start the sample chatbot
lstm_chatbot()

def evaluate_lstm_model(test_set, model, qa_pairs, vocab, max_len, question_vecs_tensor, threshold=0.5):
    results = []
    total_time = 0
    all_generated = []
    all_expected = []

    model.eval()
    smoothie = SmoothingFunction().method4 #smoothing function is used so that even if there is not a perfect match, it provides score for partial match or reasonably correct response

    # for all the QnA pair in the test set
    for item in test_set:
        query = item["query"]
        expected = item["expected_answer"]

        start_time_lstm = time.time() # to track the inference time
        input_vec = encode_question(query).unsqueeze(0) # input query is encode into vector
        with torch.no_grad():
            query_embedding = model(input_vec).float() # get the corresponding embedding
            similarities = F.cosine_similarity(query_embedding, question_vecs_tensor) # get the cosine similarity
            best_match_index = torch.argmax(similarities).item() # get the best matching question
            best_score = similarities[best_match_index].item() # get the best similarity score

        if best_score >= threshold:
            generated = qa_pairs[best_match_index]["expected_answer"]
        else:
            generated = "I'm sorry, I don't know the answer."

        response_time_lstm = time.time() - start_time_lstm
        total_time += response_time_lstm

        # Exact Match is calculated for the accuracy
        exact_match = int(expected.lower() in generated.lower()) if expected else 0

        # BLEU Score is calculated
        bleu = sentence_bleu([expected.split()], generated.split(), smoothing_function=smoothie) if expected else 0.0

        # variable result is appended with the metrics calculated above
        results.append({
            "Query": query,
            "Generated": generated,
            "Expected": expected,
            "ExactMatch": exact_match,
            "BLEU": bleu,
            "TimeTaken": response_time_lstm
        })

        all_generated.append(generated)
        all_expected.append(expected)

    # empty expected answers are filtered out
    filtered_generated = [g for g, e in zip(all_generated, all_expected) if e.strip()]
    filtered_expected = [e for e in all_expected if e.strip()]

  # berts score is calculated
    if filtered_expected:
        P, R, F1 = bert_score(filtered_generated, filtered_expected, lang="en", verbose=True)
        avg_bertscore_f1 = F1.mean().item()
    else:
        avg_bertscore_f1 = 0.0


    accuracy_lstm = sum(r["ExactMatch"] for r in results) / len(results)
    avg_bleu_lstm = sum(r["BLEU"] for r in results) / len(results)
    avg_time_lstm = total_time / len(results)

    print("\n--- Evaluation Summary ---")
    print(f"Accuracy (Exact Match): {accuracy_lstm:.2f}")
    print(f"Average BLEU Score: {avg_bleu_lstm:.2f}")
    print(f"Average BERTScore F1: {avg_bertscore_f1:.2f}")
    print(f"Average Response Time: {avg_time_lstm:.2f}s")

    return results


# Evaluation pipeline starts from here for LSTM model

github_file_path_testing = "https://raw.githubusercontent.com/sandeepkumar-84/DBS-Msc-Applied-Research-2025/refs/heads/main/data/training-testing-data-files/LSTM_Testing_DataSet.json"
test_set = []

def load_test_set_lstm(file_list):    
    all_data = []
    for path in file_list:
        if not os.path.exists(path):
            print(f"file {path} does not exist, skipping...")
            continue
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
                if isinstance(data, list):
                    all_data.extend(data)
                    print(f"loaded {len(data)} records from {path}")
                else:
                    print(f"not a list of QA pairs - {path}")
        except json.JSONDecodeError:
            print(f"error in decoding JSON in {path}")
        except Exception as e:
            print(f"error while reading {path}")
    return all_data

def load_github_file(file_url):
    try:
        response = requests.get(file_url)
        if response.status_code == 200:
            test_set = response.json()
            if isinstance(test_set, list):
                print(f"test set loaded from GitHub file: {file_url}")
                return test_set
            else:
                print(f"github does not contain a list of test items.")
        else:
            print(f"gitHub file not found: {file_url}")
    except Exception as e:
        print(f"error reading GitHub file {file_url}: {e}")
    return []

test_set = load_test_set_lstm(testing_path_files)

if not test_set:
    print(f"test set is empty or not loaded from local file. trying to load from GitHub...")
    test_set = load_github_file(github_file_path_testing)
    

print(f"total test set loaded: {len(test_set)}")

results = evaluate_lstm_model(
    test_set=test_set,
    model=model,
    qa_pairs=qa_pairs,
    vocab=vocab,
    max_len=max_len,
    question_vecs_tensor=question_vecs_tensor,
    threshold=0.5
)