import numpy as np
import pandas as pd
from baseline_memory_chatbot import BaselineMemoryChatbot
from conflict_memory_chatbot import ConflictMemoryChatbot
from config import client, index
import uuid
import joblib
from prompt import system_prompt


train_df = pd.read_csv("data/memory_dataset_train.csv")
train_embeddings = np.load("data/X_train_embeddings.npy")
svm_model = joblib.load("models/svm_memory_classifier.joblib")

def evaluate_memory_state(chatbot, conversation):
    for message in conversation:
        chatbot.process_memory(message)

    return chatbot.get_memory_state()

def evaluate_retrieval(chatbot, test, mode):
    for message in test["conversation"]: #looping through messages in the conversation eval dataset
        chatbot.process_memory(message) #conversation gets passed through memory processing function

    response = chatbot.chat_once(test["question"]) #question gets passed through chat()
    normalized_response = response.lower()
    #print(test[f"{mode}_keywords"])
    #print(test[f"{mode}_forbidden"])
    contains_expected = all(
    keyword.lower() in normalized_response
    for keyword in test[f"{mode}_keywords"]
    ) #has expected keywords

    contains_forbidden = any(
        keyword.lower() in normalized_response
        for keyword in test[f"{mode}_forbidden"]
    ) #forbidden

    passed = contains_expected and not contains_forbidden #if follows both conditions, is a pass
    scenario = test["name"]
    results = {"scenario":scenario, #making results a dictionary so can be used in next function
               "mode":mode,
               "conversation": test["conversation"],
               "question": test["question"],
                "response": response,
                "passed": passed,
               "contains_forbidden": contains_forbidden,
               "contains_expected": contains_expected
                }
    return results

#reworking eval function/s so that function well for streamlit app

#extraction_method types: semantic, svm, llm
#architecture types: baseline, conflict

def initialize_bot(architecture, extraction_method):
    bot_class = (BaselineMemoryChatbot 
             if architecture == 'baseline'
             else ConflictMemoryChatbot
            )
    
    classifier = None
    semantic_train_df = None
    semantic_train_embeddings = None
    if extraction_method == 'semantic':
        semantic_train_df = train_df
        semantic_train_embeddings = train_embeddings
    elif extraction_method == 'svm':
        classifier = svm_model
    elif extraction_method != 'llm':
        raise ValueError(f"invalid extraction method: {extraction_method}")

    return bot_class(
        openai_client=client,
        pinecone_index = index,
        embedding_model = 'text-embedding-3-small',
        chat_model="gpt-4.1-mini",
        extraction_method = extraction_method,
        memory_namespace = (f"{architecture}_{extraction_method}_eval_{uuid.uuid4()}"),
        system_prompt = system_prompt,
        top_k = 5,
        classifier = classifier,
        semantic_train_df=semantic_train_df,
        semantic_train_embeddings = semantic_train_embeddings
    )
    

def compare_memory_sets(predicted, expected):
    return set(predicted) == set(expected)

