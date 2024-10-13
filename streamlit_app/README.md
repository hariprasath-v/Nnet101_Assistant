## File information
### app.py
#### This Streamlit app runs in a local environment
#### Data used for this app:[nnet_101_qna_with_id.json](https://github.com/hariprasath-v/Nnet101_Assistant/blob/main/data/nnet_101_qna_with_id.json)
#### A TF-IDF search is used to retrieve information similar to the user query.
#### Ollama's gemma:2b model is used to generate a RAG response.
#### Response Generation Steps,
- Receive the user query.
- Retrieve similar results and build a prompt based on those results.
- Generate RAG results using the LLM.
---

### [app_cloud.py](https://nnet101assistant.streamlit.app/)
#### This Streamlit app runs on the Streamlit cloud.
#### Data used for this app:[llm_answers_mistral_7b_instruct_v0_1_with_vector.csv](https://github.com/hariprasath-v/Nnet101_Assistant/blob/main/data/llm_answers_mistral_7b_instruct_v0_1_with_vector.csv)
#### The Sentence transformer model-all-MiniLM-L6-v2 is used to encode user query.
#### Lancedb vectors are used to perform a vector similarity search.
#### Gemini-1.0-pro mode is used to generate a RAG response.
#### Response Generation Steps,
- Receive the user query and encode it using the sentence transformer model.
- Perform a vector search based on the encoded query and retrieve the top 5 results.
- Generate RAG results using the LLM
---
