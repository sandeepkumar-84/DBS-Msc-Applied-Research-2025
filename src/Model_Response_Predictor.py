from sentence_transformers import SentenceTransformer
import faiss
import os
import nltk
from huggingface_hub import snapshot_download
import json
from transformers import T5ForConditionalGeneration, T5Tokenizer


nltk.download('punkt')

from IntentCommonResponse import detect_intent_and_respond

local_save_path = "/content/Saved_Model_Local"
mode_to_run = 'local'
repo_id = "sandeepkumar84/dbs-chatbot-transformer-hf-v3"

t5_model_dbs = T5ForConditionalGeneration.from_pretrained("t5-base")
t5_tokenizer_dbs = T5Tokenizer.from_pretrained("t5-base")

reloaded_model_local = None
reloaded_index_local = None
reloaded_corpus_local = None
reloaded_model_hf = None
reloaded_index_hf = None
reloaded_corpus_hf = None

def reload_local(local_save_path):
    global reloaded_model_local, reloaded_index_local, reloaded_corpus_local
    reloaded_model_local = SentenceTransformer(f"{local_save_path}/sentence_transformer")
    reloaded_index_local = faiss.read_index(f"{local_save_path}/faiss_index_dbs.index")
    with open(f"{local_save_path}/corpus_dbs.json","r",encoding="utf-8") as f:
        reloaded_corpus_local = json.load(f)
    print("reloaded from LOCAL")

def reload_hf(repo_id):
    global reloaded_model_hf, reloaded_index_hf, reloaded_corpus_hf
    local_repo_dir = snapshot_download(repo_id=repo_id)
    reloaded_model_hf  = SentenceTransformer(os.path.join(local_repo_dir,"sentence_transformer"))
    reloaded_index_hf  = faiss.read_index(os.path.join(local_repo_dir,"faiss_index_dbs.index"))
    with open(os.path.join(local_repo_dir,"corpus_dbs.json"),"r") as f:
        reloaded_corpus_hf = json.load(f)
    print("reloaded from HUGGINGFACE")

def load_model_and_index():
    if mode_to_run=='local':
        reload_local(local_save_path)
    elif mode_to_run=='hf':
        reload_hf(repo_id)

def retrieve_passages(query, top_k=3):
    if mode_to_run == 'local' and reloaded_model_local is not None:
        model_dbs_transformer = reloaded_model_local
        index_dbs = reloaded_index_local
        corpus_dbs = reloaded_corpus_local
    elif mode_to_run == 'hf' and reloaded_model_local is not None:
        model_dbs_transformer = reloaded_model_hf
        index_dbs = reloaded_index_hf
        corpus_dbs = reloaded_corpus_hf

    query_embedding = model_dbs_transformer.encode([query], convert_to_numpy=True)
    distances, indices = index_dbs.search(query_embedding, top_k)
    results = [corpus_dbs[idx] for idx in indices[0]]
    return results

def generate_answer(query, context_passages):
    context = " ".join(context_passages)
    prompt = f"question: {query} context: {context}"
    inputs = t5_tokenizer_dbs(prompt, return_tensors="pt", truncation=True, padding=True)
    outputs = t5_model_dbs.generate(**inputs, max_length=128)
    answer = t5_tokenizer_dbs.decode(outputs[0], skip_special_tokens=True)
    return answer

def provide_res_to_ui(text):
    query = text
    response, tag = detect_intent_and_respond(query)
    if tag in ["greeting", "goodbye", "number", "location", "random", "swear", "salutaion", "task", "creator", "name"]:
        return response
    else:
        load_model_and_index()
        passages = retrieve_passages(query)
        answer = generate_answer(query, passages)    
    return answer

#print(chat_pre_response(chatbot_pre,"location"))
#query_to_test = "student population of DBS?"
#print("response: ", provide_res_to_ui(query_to_test))