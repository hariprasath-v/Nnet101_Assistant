## Notebook Information

### stackoverflow-data-pre-processing
- Data collection using the StackExchange API.  
- Generate answer summaries using the Gemini-1.0-pro model.
- The data gathering and pre-processing steps.
  - Filter the questions that do not have images.
  - Retrieve question and answer pairs only for the accepted answers.
  - Summarize the gathered answers using the Gemini-1.0-pro model.

### simple_rag
Retrieve relevant results based on user queries using TF-IDF similarity.  
Create a prompt from the retrieved results.  
Generate RAG answers using the Ollama - gemma:2b model.

### rag_text_elasticsearch
Retrieve relevant results based on user queries using Elasticsearch.  
Create a prompt from the retrieved results.  
Generate RAG answers using the Ollama - gemma:2b model.

### rag_elasticsearch_vector_similarity
LLM-powered RAG using Elasticsearch with vector-based similarity.

### unique-id-generation
Create a unique hash ID for each QnA pair.

### generate-sample-data-using-llm
Generate sample questions using the Gemini-1.0-pro model.

### text_search_evaluation_elasticsearch
Information retrieval using Elasticsearch and custom search methods.  
Calculate recall and MRR from the search results.

### rag_elasticsearch_vector_similarity
Information retrieval using Elasticsearch with vectors.  
Calculate recall and MRR from search results.

### offline-rag-evaluation
Generate answers using an LLM from Elasticsearch results and calculate the cosine similarity between the original answer and the LLM-generated answer.
- Generate answers using llama-2-7b-chat-int8
- Generate answers using mistral-7b-instruct-v0.1
- Compute cosine similarity 

Use the LLM as a judge to compare the relevance of original answers and LLM-generated answers.
- Compare the relevance of original answers and LLM-generated answers using llama-2-7b-chat-int8.

Use the LLM as a judge to compare the relevance of LLM-generated answers to questions.
- Compare the relevance of LLM-generated answers to questions using llama-2-7b-chat-int8.
