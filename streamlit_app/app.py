from openai import OpenAI
import minsearch
import streamlit as st
import requests
import json
import os
import minsearch




# Load data 
data_vec_url = "./data/llm_answers_mistral_7b_instruct_v0_1_with_vector.csv"
data = pd.read_csv(data_vec_url)
data['question_answer_vector']= data['question_answer_vector'].apply(lambda x: [float(i) for i in x.strip("[]").split(",") if i])



# Create the index
index = minsearch.Index(
    text_fields=["question", "answer_llm"],
    keyword_fields=["tags"]
)

# Fit the index with data
index.fit(data)

#Ollama's opeai end point
client = OpenAI(
    base_url='http://localhost:11434/v1',
    api_key='ollama',
)


#tfidf search
def search(query):
    results = index.search(
        query=query,
        num_results=5)
    return results

#build prompt
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
        context = context + \
            f"tags: {doc['tags']}\nquestion: {doc['question']}\nanswer: {doc['answer']}\n\n"

    prompt = prompt_template.format(question=query, context=context).strip()
    return prompt

#llm response
def llm(prompt):
    response = client.chat.completions.create(
        model="gemma:2b",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

#rag response
def rag(query):
    search_results = search(query)
    prompt = build_prompt(query, search_results)
    answer = llm(prompt)
    return answer


sample_questions = ["What is batch size?",
                    "How do I choose the number of hidden layers in a neural network?",
                    "What are the advantages of smaller batch sizes?",
                    "When is the output of ReLU equal to zero?",
                    "What is global max pooling?"]


st.markdown(
            """
        <style>
            .st-emotion-cache-1c7y2kd {
                flex-direction: row-reverse;
                text-align: right;
            }
        </style>
        """,
            unsafe_allow_html=True,
        )


# Initialize conversation history
if "messages" not in st.session_state:
      st.session_state["messages"] = []




# Streamlit app UI
st.title("Nnet101_Assistant")
st.write("Ask me about neural network basics, and I'll do my best to respond!")



user_input = st.chat_input("Message Nnet101")
# Chat UI using chat_input and chat_message
if user_input: 

    # Add user message to chat history
    
    st.session_state.messages.append({"role": "user", "content": user_input})

    # Process input and get response (assuming rag() is defined)
    bot_response = rag(user_input)  # Call the RAG model for response generation

    

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": bot_response})
 




# Sidebar for chat controls
with st.sidebar:
    st.header("Chat Controls")

    with st.popover("Sample Questions"):
        for q in sample_questions:
            st.markdown(q)
    
    # Button to clear conversation history
    if st.button("Clear Conversation"):
        st.session_state["messages"].clear()  # Clear message history

    # Message if history is empty
    if not st.session_state.messages:
        st.write("No conversation history.")

# Display conversation history from session state
for message in st.session_state.messages:
    if message["role"] == "user":
        with st.chat_message("user"):
            st.markdown(message["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(message["content"]) 
