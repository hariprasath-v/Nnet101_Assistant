import streamlit as st
from openai import ChatCompletion
import openai


# Set up OpenAI API Key
openai.api_key = 'YOUR_OPENAI_API_KEY'

# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar for user inputs
st.sidebar.title("Chatbot Settings")
model_name = st.sidebar.selectbox("Select Model", ["GPT-4", "GPT-3.5"])
temperature = st.sidebar.slider("Temperature", 0.0, 1.0, 0.7)

st.title("Interactive Chatbot")
st.write("Ask me anything, and I'll do my best to respond!")

# Text input
user_input = st.text_input("Your Question:", "")

# Process input and get response
if user_input:
    # Add user input to conversation history
    st.session_state.history.append({"role": "user", "content": user_input})

    # Call OpenAI API or your model's API
    response = openai.ChatCompletion.create(
        model=model_name,
        messages=st.session_state.history,
        temperature=temperature,
    )

    # Display the response
    bot_response = response['choices'][0]['message']['content']
    st.session_state.history.append({"role": "assistant", "content": bot_response})
    st.write(f"**Assistant**: {bot_response}")

# Display conversation history
st.write("### Conversation History")
for i, message in enumerate(st.session_state.history):
    if message["role"] == "user":
        st.write(f"**You**: {message['content']}")
    else:
        st.write(f"**Assistant**: {message['content']}")

# Clear history button
if st.button("Clear Conversation"):
    st.session_state.history = []
