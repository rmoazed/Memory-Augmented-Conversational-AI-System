# Memory-Extraction-Chatbot
A modular memory-augmented chatbot framework comparing semantic, supervised machine-learning, and LLM-based memory extraction under baseline and conflict-aware memory management 
architectures.



# App Demo


[Explore Interactive App Here](https://memory-extraction-chatbot-kgvrscdlmwj9lrh6e9sjtc.streamlit.app/)
![App Screenshot]("images/streamlit_Screenshot_eval.png")

# Overview


The Memory Extraction Chatbot is an end-to-end conversational AI system that explores how different memory extraction techniques impact user-chatbot interaction and personalization of chatbot response. 


Rather than storing every user input message as a memory indiscriminately, the chatbot predicts which user messages are important enough to retain as long-term memory. Three extraction strategies are compared:


. Semantic Similarity

. Support Vector Machine (SVM)

. Large Language Model (LLM)


The project also compares two memory architectures:


.**Baseline Memory** - appends new memories without discretion beyond importance score

.**Conflict-Aware Memory** - detects and updates contradictory memories


# Key Features

. End-to-end conversational AI pipeline

. Three memory extraction approaches

. Conflict-aware memory updates

. Vector database retrieval through Pinecone

. OpenAI GPT integration

. Interactive Streamlit dashboard

. Memory replay visualization

. User-chatbot interaction interface

. Quantitative evaluation of memory state and retrieval accuracy


## Technologies

. Python

. Streamlit

. OpenAI API

. Pinecone

. scikit-learn

. pandas

. NumPy

. sentence-transformers

. Hugging Face Transformers


# System Architecture


<img src="images/chatbot_architecture.png" width="150">


The user inputs a message. The chatbot then determines the importance score of the message. There are 5 categories that each correlate with an importance score. They are:


. 5 - critical constraints (severe food allergies, crucial medical information, etc.)

. 4 - stable preference/goal (favorite foods, favorite places, etc.)

. 3 - recurrent context (recurring appointments, schedule related, etc.)

. 2 - temporary context (upcoming trip, etc.)

. 1 - do not store (e.g. "how are you?")


Once the importance score is determined to be above a 1, the memory is stored. The message gets embedded as a vector of dimension 1,536 (standard for OpenAI embeddings) and stored in Pinecone. This memory is then accessible for the chatbot in formulating future responses (GPT response generation) if a user references something related to their previous message. 


# Memory Extraction Methods


## Semantic Similarity

User input gets embedded as vector, cosine similarity is used to determine top k most similar examples in the dataset (containing 200 examples of each category), the importance scores of those top k data points are weighted, and the weighted importance score is used as the importance score for the user input message. If the weighted importance score is above a 1, the memory gets stored in Pinecone. 


## SVM


SVM model trained on training dataset to predict importance score of user input message (training data examples are labeled with importance score). Supervised machine learning method.


## LLM

Prompt-based extraction. The chatbot is given a strict, clear prompt on how to categorize user messages and then instructed to output a JSON. This is what gets upserted to Pinecone. 


# Conflict Handling


# Evaluation 


# Streamlit Application


# Repository Structure


# Running Locally


# Future Improvements


. BERTScore evaluation

. Human Evaluation

. Hybrid memory extraction

. Larger datasets

. Fine-tuned classifier

. Multi-session memory

. Memory expiration
