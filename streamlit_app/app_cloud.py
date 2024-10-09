import streamlit as st
import requests
import json

# Load JSON data directly from URL
url = "https://raw.githubusercontent.com/hariprasath-v/Nnet101_Assistant/refs/heads/main/data/nnet_101_qna_with_id.json"
response = requests.get(url)
data = response.json()

# Load minsearch from URL and save it
url = "https://raw.githubusercontent.com/alexeygrigorev/minsearch/main/minsearch.py"
response = requests.get(url)

# Save the content to a local file
with open("minsearch.py", "wb") as file:
    file.write(response.content)

import minsearch

# Create the index
index = minsearch.Index(
    text_fields=["question", "answer"],
    keyword_fields=["tag"]
)

# Fit the index with data
index.fit(data)


from openai import OpenAI

client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)



def search(query):
    results = index.search(
        query=query,
        num_results=5
    )

    return results

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
    response = client.chat.completions.create(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}]
    )
    
    return response.choices[0].message.content


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
