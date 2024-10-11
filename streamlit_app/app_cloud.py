import pandas as pd
import streamlit as st
import requests
import json
import importlib.util
import sys

import google.generativeai as genai
import os
from sentence_transformers import SentenceTransformer
import lancedb
from typing import List, Optional
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector

genai.configure(api_key=st.secrets["gemini_key"])
llm_model = genai.GenerativeModel("models/gemini-1.0-pro")

# Load data from URL
data_vec_url = "https://raw.githubusercontent.com/hariprasath-v/Nnet101_Assistant/refs/heads/main/data/llm_answers_mistral_7b_instruct_v0_1_with_vector.csv"
data_vec = pd.read_csv(data_vec_url)
data_vec['question_answer_vector']= data_vec['question_answer_vector'].apply(lambda x: [float(i) for i in x.strip("[]").split(",") if i])

#Load sentence trasformer model
model_name = 'all-MiniLM-L6-v2'
sen_trans_model = SentenceTransformer(model_name)


schema = pa.schema([
            pa.field("question", pa.string()),
            pa.field("answer_llm", pa.string()),
            pa.field("tags", pa.string()),
            pa.field("question_answer_vector", pa.list_(pa.float32(),384))])

db = lancedb.connect("/tmp/lancedb")

data_table = pa.Table.from_pandas(data_vec[['answer_llm', 'question', 'tags', 'question_answer_vector']], schema=schema)

# Create the table in LanceDB
tbl = db.create_table("nnet101", data=data_table, mode="overwrite")


def search(query):
    # Embed the question
    emb = sen_trans_model.encode(query, show_progress_bar=False)

    # Use LanceDB to get top 5 most relevant context
    context = tbl.search(emb,vector_column_name = 'question_answer_vector',).limit(5).to_pandas()

    return context[['question','answer','tags']].to_dict(orient='records')

def build_prompt(query, search_results):
    prompt_template = """
You're a course teaching assistant. Answer the QUESTION based on the CONTEXT from the FAQ database.
Use only the facts from the CONTEXT when answering the QUESTION.

QUESTION: {question}

CONTEXT: 
{context}
""".strip()

    context = ""
    
    for doc in search_results:
        context = context + f"tags: {doc['tags']}\nquestion: {doc['question']}\nanswer: {doc['answer']}\n\n"
    
    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

def llm(prompt):
    response = llm_model.generate_content(prompt)
    
    return response.text


def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer



# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Streamlit app UI
st.title("Nnet101_Assistant")
st.write("Ask me anything related to Nnet basics, and I'll do my best to respond!")

# Text input
user_input = st.text_input("Your Question:", "")

# Process input and get response
if user_input:
    # Add user input to conversation history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Display the response
    bot_response = rag(user_input)
    st.session_state.history.append({"role": "assistant", "content": bot_response})
    st.write(f"**Assistant**: {bot_response}")

# Display conversation history
st.write("### Conversation History")
for message in st.session_state.history:
    if message["role"] == "user":
        st.write(f"**You**: {message['content']}")
    else:
        st.write(f"**Assistant**: {message['content']}")

# Clear history button
if st.button("Clear Conversation"):
    st.session_state.history.clear()  # Clear history using the clear method
    st.write("Conversation history cleared!")  # Optionally show a message

# Optionally, re-display conversation history
if not st.session_state.history:  # If history is empty, show a message
    st.write("No conversation history.")
