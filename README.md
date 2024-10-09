# Nnet101_Assistant

This repository contains the final project for the **[LLM Zoom Camp](https://github.com/DataTalksClub/llm-zoomcamp/tree/main)** course. It demonstrates the application of Retrieval-Augmented Generation (RAG) techniques using StackExchange data to answer questions related to foundational neural network concepts.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Analysis](#analysis)
- [Technologies](#technologies)
- [Installation](#installation)



## Project Overview
This chatbot leverages Retrieval-Augmented Generation (RAG) to answer questions about neural networks. It combines a retriever to find relevant answers from a curated dataset and a generator to deliver clear, concise responses, making it a valuable tool for users seeking foundational knowledge in neural networks.

## Data Collection
The data for this project was gathered using the StackExchange API and focused on fundamental questions related to neural networks. The collected answers vary in length. The Gemini 1.0 model was used to create a short summary of the answers.
You can find the data [here](https://github.com/hariprasath-v/Nnet101_Assistant/blob/main/data/Stackoverflow_data(neural_networks_stats)_pre_processed_Gemini_LLM.csv)


## Analysis

### Text search evaluation

| Type                | Hit Rate | MRR                          |
|---------------------|----------|------------------------------|
| text_elasticsearch   | 0.5828   | 0.43964666666666774          |
| text_customsearch    | 0.5572   | 0.4396800000000009           |

### Vector search evaluation

| Type                                | Hit Rate | MRR                          |
|-------------------------------------|----------|------------------------------|
| question_vector_elasticsearch       | 0.6256   | 0.5150333333333336          |
| answer_vector_elasticsearch         | 0.8308   | 0.7089066666666664          |
| question-answer_vector_elasticsearch| 0.8548   | 0.7323599999999993          |
| custom-combined_vector_scoring_elasticsearch | 0.832   | 0.7055066666666656  |

### mistral-7b-instruct-v0.1 cosine similarity(original answers vs llm generated answers)

| Count      | Mean     | Std Dev | Min       | 25%      | 50%      | 75%      | Max      |
|------------|----------|---------|-----------|----------|----------|----------|----------|
| 2500.000000| 0.709169 | 0.157913| -0.068219 | 0.621302 | 0.741953 | 0.825930 | 0.986987 |

### llama-2-7b-chat-int8 cosine similarity(original answers vs llm generated answers)

| Count      | Mean     | Std Dev | Min       | 25%      | 50%      | 75%      | Max      |
|------------|----------|---------|-----------|----------|----------|----------|----------|
| 2500.000000| 0.675582 | 0.161148| -0.020848 | 0.582057 | 0.705028 | 0.792661 | 0.981918 |


## Technologies
- Data: Stackapps API
- LLM: Gemini, Mistral, llama, Ollama, cloudflare
- Knowledge base: TF-IDF search, Elasticsearch

## Installation

The following processes are required to run Elasticsearch and Ollama.

1. Run Elasticsearch
```bash
docker run -it \
    --rm \
    --name elasticsearch \
    -p 9200:9200 \
    -p 9300:9300 \
    -e "discovery.type=single-node" \
    -e "xpack.security.enabled=false" \
    docker.elastic.co/elasticsearch/elasticsearch:8.4.3
```
2. Run Ollama
```bash
docker run -it \
    --rm \
    -v ollama:/root/.ollama \
    -p 11434:11434 \
    --name ollama \
    ollama/ollama
```
3. Run gemma 2b
```bash
docker exec -it ollama ollama run gemma:2b
```

4. Install requirements
```bash
pip install requirements
```
   


