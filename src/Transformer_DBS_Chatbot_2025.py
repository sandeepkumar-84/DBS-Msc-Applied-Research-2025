import sys
if 'transformers' not in sys.modules:
    import os
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    from transformers import T5ForConditionalGeneration, T5Tokenizer
    from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
    from huggingface_hub import hf_hub_download, snapshot_download, login, HfApi, HfFolder, Repository
    import nltk
    import shutil
    from transformers.modeling_utils import PreTrainedModel
    import time
    from sklearn.metrics import accuracy_score
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from bert_score import score as bert_score
    import json
    import requests
    import matplotlib.pyplot as plt
    print("all necessary imports have been successfully completed.")
else:
    print("sentence_transformers is already imported, skipping import statements.")

# if already imported then dont run the code again

try:
    nltk.data.find('tokenizers/punkt')    
except LookupError:
    print("punkt not found. Attempting its download....")
    nltk.download('punkt')
    print("punkt has been downloaded........")

print("all necessary imports have been successfully completed.")

# variable declaration section 
mode_to_run = 'local' # options are 'local' or 'hf' (hugging face)
save_to_hf_required = False # set to True if you want to save to hugging face
local_save_path = "/content/Saved_Model_Local" # path to save the model and index locally
local_hf_save_path = "/content/Saved_Model_HF" # path to save the model and index before pushing to hugging face
local_research_path = "/content/Research-Chatbot" # path to research folder where the training and testing data is stored
repo_id = "sandeepkumar84/dbs-chatbot-transformer-hf-v3" # hugging face repository id
local_repo_path = "./dbs-chatbot-transformer-hf-v3" # define a local path for the repository
evaluation_results_path = "/content/evaluation_results.txt" # path to save evalution results


print("all necessary variables are declared")

print(f"The program is running in the mode = {mode_to_run}")

if mode_to_run in ['local']:
    print("model and index will be loaded from LOCAL")
elif mode_to_run in ['hf']:
    print("model and index will be loaded from HUGGINGFACE")


print(f"create directories if they do not exist.")

os.makedirs(local_save_path, exist_ok=True)
os.makedirs(local_hf_save_path, exist_ok=True)
os.makedirs(local_research_path, exist_ok=True)

print(f"directories {local_save_path} and {local_hf_save_path} are created if they did not exist.")

print(f"put the following files in the {local_research_path} folder if you want to load data from local files:")
print("1. Transformer_Training_DataSet-1.txt")
print("2. Transformer_Training_DataSet-2.txt")
print("3. Transformer_Test_DataSet.json")
print("if these files are not found in the local folder, the program will attempt to fetch them from GitHub.")

print("innitial set is completed")

# github urls for training data
github_urls_training = [
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS/refs/heads/dbs_applied_research_project_v1/AppliedResearch/Working%20v1/Transformer%20Version/Transformer_Training_DataSet-1.txt",
    "https://raw.githubusercontent.com/sandeepkumar-84/DBS/refs/heads/dbs_applied_research_project_v1/AppliedResearch/Working%20v1/Transformer%20Version/Transformer_Training_DataSet-2.txt"
]

# local path for data files
training_path_files = [
    f"{local_research_path}/Transformer_Training_DataSet-1.txt", f"{local_research_path}/Transformer_Training_DataSet-2.txt",  f"{local_research_path}/Transformer_Test_DataSet.json",
]

# parameter to check if local is sucessfuly loaded or not. 
loaded = False
# corpus to hold the paragraphs
corpus_dbs = []
folder_path_training = local_research_path

# function to load the training data from local files
def load_local_files(training_path_files=None):
    success = False
    for file_name in training_path_files:
            try:
                with open(file_name, 'r', encoding='utf-8', errors='ignore') as f:
                    content = f.read()
                    paras = [para.strip() for para in content.split('\n') if len(para.strip()) > 50]
                    corpus_dbs.extend(paras)
                    success = True
                    print(f"loaded {len(paras)} paragraphs from local: {file_name}")
            except Exception as e:
                print(f"error reading {file_name}: {e}")
    return success


# function to load the training data from github urls
def load_from_github(urls):
    for url in urls:
        try:
            response = requests.get(url)
            if response.status_code == 200:
                content = response.text
                paras = [para.strip() for para in content.split('\n') if len(para.strip()) > 50]
                corpus_dbs.extend(paras)
                print(f"loaded {len(paras)} paragraphs from Github: {url}")
            else:
                print(f"Failed to fetch from github")
        except Exception as e:
            print(f"error while reading from github")

# call function to load from local 
loaded = load_local_files(training_path_files)

# check if not loaded from local, then attempt to load from github
if not loaded:
    print("getting data from github ...................")
    load_from_github(github_urls_training)

print(f"Total paragraphs loaded into corpus_dbs: {len(corpus_dbs)}")

# Wordcloud reprenstation of the corpus
from wordcloud import WordCloud
text_data = " ".join(corpus_dbs)
wordcloud = WordCloud(
    width=800,
    height=400,
    background_color='white',
    colormap='viridis'
).generate(text_data)


plt.figure(figsize=(10, 6))
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
plt.title("Word Cloud of Corpus", fontsize=16)
plt.show()

print("word cloud generated and displayed.")

# creating hugging face sentence transformer (which maps sentences and paragrapghs to 384 dimensional vector space.)  
# model and faiss index. It is for the similarity search. 
model_dbs_transformer = SentenceTransformer('all-MiniLM-L6-v2')
# encode the dbs specific corpus to get the embeddings
corpus_embeddings_dbs = model_dbs_transformer.encode(corpus_dbs, show_progress_bar=True, convert_to_numpy=True)

# fetching the dimensions of the embeddings
embedding_dim_transformer = corpus_embeddings_dbs.shape[1]
# creating a faiss index for similarity search. faiss is a facebook library for similarity search
index_dbs = faiss.IndexFlatL2(embedding_dim_transformer)
# add the corpus embeddings to the FAISS index
index_dbs.add(corpus_embeddings_dbs)

print(f"successfuly built FAISS index_dbs with {index_dbs.ntotal} vectors.")

# saving the model and index to local path
model_dbs_transformer.save(f"{local_save_path}/sentence_transformer")
faiss.write_index(index_dbs, f"{local_save_path}/faiss_index_dbs.index")
with open(f"{local_save_path}//corpus_dbs.json", "w") as f:
    json.dump(corpus_dbs, f)

print("saved SentenceTransformer model, FAISS index, and corpus to local save path.")

# function to deploy it on hugging face
def save_to_hugging_face_repo():
  api = HfApi()
  api.create_repo(repo_id, exist_ok=True)
  repo = Repository(local_dir=local_hf_save_path, clone_from=repo_id,use_auth_token=True)
  shutil.copytree(f"{local_save_path}/sentence_transformer",
                  f"{local_hf_save_path}/sentence_transformer",
                  dirs_exist_ok=True)

  shutil.copy(f"{local_save_path}/faiss_index_dbs.index", f"{local_hf_save_path}/faiss_index_dbs.index")
  shutil.copy(f"{local_save_path}/corpus_dbs.json", f"{local_hf_save_path}/corpus_dbs.json")

  repo.push_to_hub(commit_message="dbs specific chatbot for applied research project 2025")

# control statement to save to hugging face or not
if save_to_hf_required == True:
  print("saving to hugging face..............")
  save_to_hugging_face_repo()
else:
    print("skipping saving to hugging face as save_to_hf_required is False.")

# variable declaration for reload
reloaded_model_local = None
reloaded_index_local = None
reloaded_corpus_local = None
reloaded_model_hf = None
reloaded_index_hf = None
reloaded_corpus_hf = None

# function to reload from local
def reload_local(local_save_path):
    global reloaded_model_local, reloaded_index_local, reloaded_corpus_local
    reloaded_model_local = SentenceTransformer(f"{local_save_path}/sentence_transformer")
    reloaded_index_local = faiss.read_index(f"{local_save_path}/faiss_index_dbs.index")
    with open(f"{local_save_path}/corpus_dbs.json","r",encoding="utf-8") as f:
        reloaded_corpus_local = json.load(f)
    print("reloaded from LOCAL")

# function to reload from hugging face
def reload_hf(repo_id):
    global reloaded_model_hf, reloaded_index_hf, reloaded_corpus_hf
    local_repo_dir = snapshot_download(repo_id=repo_id)
    reloaded_model_hf  = SentenceTransformer(os.path.join(local_repo_dir,"sentence_transformer"))
    reloaded_index_hf  = faiss.read_index(os.path.join(local_repo_dir,"faiss_index_dbs.index"))
    with open(os.path.join(local_repo_dir,"corpus_dbs.json"),"r") as f:
        reloaded_corpus_hf = json.load(f)
    print("reloaded from HUGGINGFACE")

# based on mode_to_run variable reload the model and index
if mode_to_run=='local':
    reload_local(local_save_path)
elif mode_to_run=='hf':
    reload_hf(repo_id)

# function which  converts query into embedding, finds distances between query and passages,
# then retrieves the actual text passages from corpus. before that it loads the model and index based on 
# the mode_to_run variable
def retrieve_passages(query, top_k=3):
    if mode_to_run == 'local' and reloaded_model_local is not None:
        model_dbs_transformer = reloaded_model_local
        index_dbs = reloaded_index_local
        corpus_dbs = reloaded_corpus_local
    elif mode_to_run == 'hf' and reloaded_model_local is not None:
        model_dbs_transformer = reloaded_model_hf
        index_dbs = reloaded_index_hf
        corpus_dbs = reloaded_corpus_hf
    else:
        model_dbs_transformer = model_dbs_transformer
        index_dbs = index_dbs
        corpus_dbs = corpus_embeddings_dbs

    query_embedding = model_dbs_transformer.encode([query], convert_to_numpy=True)
    distances, indices = index_dbs.search(query_embedding, top_k)
    results = [corpus_dbs[idx] for idx in indices[0]]
    return results

# loading t5 model and tokenizer for generating the answer
t5_model_dbs = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer_dbs = T5Tokenizer.from_pretrained("t5-base")

# based on the passages retrieved above, this function generates the answer
def generate_answer(query, context_passages):
    context = " ".join(context_passages)
    prompt = f"question: {query} context: {context}"
    inputs = t5_tokenizer_dbs(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = t5_model_dbs.generate(**inputs, max_length=128)
    answer = t5_tokenizer_dbs.decode(outputs[0], skip_special_tokens=True)
    return answer

# example usage
query = "How many books does the DBS Library have?"
passages = retrieve_passages(query)
answer = generate_answer(query, passages)
print("Generated Answer:\n", answer)

# function that mimics a chatbot interface
def start_chatbot(top_k=3):
    print(f"DBS Chatbot is ready Type 'exit' to quit.\n Running in mode = {mode_to_run}")

    while True:
        query = input("You: ")
        if query.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break

        retrieved_passages = retrieve_passages(query)
        answer = generate_answer(query,retrieved_passages)

        print("DBS Chatbot:", answer)
        print("-" * 60)

# execute the chatbot function to start the conversation
start_chatbot()

# following code is purosefully commented out. It is to refine the answer generated by the t5 model using a 
# llama based model. It is time consuming and can be used for future enhancements
'''
model_id_tinyllm_dbs = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"
tokenizer_tinyllm_dbs = AutoTokenizer.from_pretrained(model_id_tinyllm_dbs)
model_tinyllm_dbs = AutoModelForCausalLM.from_pretrained(model_id_tinyllm_dbs, device_map="auto", torch_dtype="auto")

def refine_answer_llama(query, raw_answer):
    prompt = f"""<|system|>You are an intelligent chatbot. Your role is to convert the retreived chunks into humanized text..<|end|>
<|user|>Question: {query}
Answer: {raw_answer}
Convert the following text to a chatbot response. Add greetings and ask the user if they have any further questions. AT the end of the conversion, say Thank You.<|end|>
<|assistant|>"""

  # the prompt is tokenize
    inputs = tokenizer_tinyllm_dbs(prompt, return_tensors="pt").to(model_tinyllm_dbs.device)
  # generate the output
    outputs = model_tinyllm_dbs.generate(
        **inputs,
        max_new_tokens=500,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        temperature=0.8,
        pad_token_id=tokenizer_tinyllm_dbs.eos_token_id
    )

    # decode the generated tokens into text
    full_output = tokenizer_tinyllm_dbs.decode(outputs[0], skip_special_tokens=True)
    # extract only the the assistant's part from the response & outout is then returned
    if "<|assistant|>" in full_output:
        return full_output.split("<|assistant|>")[-1].strip()
    return full_output.strip()

query = "How many books does the DBS Library have?"
raw_answer = "over 43,000"

#refined = refine_answer_llama(query, raw_answer)
#print("📘 Refined Answer:", refined)
'''

# evaluation of the model using a test set

# local testing file path
file_path_test = f"{local_research_path}/Transformer_Test_DataSet.json"
# github testing file path
github_urls_test = "https://raw.githubusercontent.com/sandeepkumar-84/DBS/refs/heads/dbs_applied_research_project_v1/AppliedResearch/Working%20v1/Transformer%20Version/Transformer_Test_DataSet.json"

test_set = []
load_test_local = False

# local function to load the test set
def load_test_set(path):
    if not os.path.exists(path):
        raise FileNotFoundError(f"The file {path} does not exist.")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        load_test_local = True
        print(f"loaded test set from local file: {path}")
    return data

# function to load the test set from github
def lod_test_set_from_github(path):
    try:
        response = requests.get(path)
        if response.status_code == 200:
            data = response.json()
            return data
        else:
            raise Exception(f"failed to fetch data from {path}. Status code: {response.status_code}")
    except Exception as e:
        raise Exception(f"error loading test set from GitHub: {e}")

# control statement to load the test set, first from local, if not found then from github
try:
    load_test_set(file_path_test)
    if not load_test_local:
        test_set = lod_test_set_from_github(github_urls_test)
    print("test set loaded successfully.")
except Exception as e:
    print(f"Error: {e}")

# This function gives more balanced BLEU scores. In case of higher n-grams, it is likely that there will 
# be no matching n-grams. therefore, smoothing is used to avoid zero scores.
smoothie = SmoothingFunction().method4

# evaluation function
def evaluate_transformer_model(test_set):
    # variable declaration
    results = []
    total_time = 0
    all_generated = []
    all_expected = []

    # for all the QnA pair in the test set
    for i, item in enumerate(test_set):
        query = item["query"]
        expected = item["expected_answer"]
        print(f"Evaluating Query {i+1}/{len(test_set)}")
        start_time = time.time() # to track the inference time
        retrieved_passages = retrieve_passages(query) # retrieve the passage from faiss
        generated = generate_answer(query,retrieved_passages) # generate the answer
        end_time = time.time()

        response_time = end_time - start_time
        total_time += response_time


        all_generated.append(generated)
        all_expected.append(expected)


        exact_match = int(expected.lower() in generated.lower())

        # generate bleu score
        reference = [expected.split()]
        candidate = generated.split()
        bleu = sentence_bleu(reference, candidate, smoothing_function=smoothie)

        results.append({
            "Query": query,
            "Generated": generated,
            "Expected": expected,
            "ExactMatch": exact_match,
            "BLEU": bleu,
            "TimeTaken": response_time
        })


    # generate bert score
    P, R, F1 = bert_score(all_generated, all_expected, lang="en", verbose=True)
    avg_bertscore_f1 = F1.mean().item()


    accuracy = sum(r["ExactMatch"] for r in results) / len(results)
    avg_bleu = sum(r["BLEU"] for r in results) / len(results)
    avg_time = total_time / len(results)

    print(f"\n--- Evaluation Summary ---")
    print(f"Accuracy (Exact Match): {accuracy:.2f}")
    print(f"Average BLEU Score: {avg_bleu:.2f}")
    print(f"Average BERTScore F1: {avg_bertscore_f1:.2f}")
    print(f"Average Inference Time: {avg_time:.2f} seconds\n")

    # save results in a text file
    with open(evaluation_results_path, "w") as f:
        f.write(f"Accuracy (Exact Match): {accuracy:.2f}\n")
        f.write(f"Average BLEU Score: {avg_bleu:.2f}\n")
        f.write(f"Average BERTScore F1: {avg_bertscore_f1:.2f}\n")
        f.write(f"Average Inference Time: {avg_time:.2f} seconds\n\n")
        f.write("Detailed Results:\n")
        for r in results:
            f.write(f"Query: {r['Query']}\n")
            f.write(f"Generated: {r['Generated']}\n")
            f.write(f"Expected: {r['Expected']}\n")
            f.write(f"Exact Match: {r['ExactMatch']}\n")
            f.write(f"BLEU Score: {r['BLEU']:.4f}\n")
            f.write(f"Time Taken: {r['TimeTaken']:.4f} seconds\n\n")

    return results

results = evaluate_transformer_model(test_set)
