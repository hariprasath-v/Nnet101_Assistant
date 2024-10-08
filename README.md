# Nnet101_Assistant

This repository contains the final project for the **[LLM Zoom Camp](https://github.com/DataTalksClub/llm-zoomcamp/tree/main)** course. It demonstrates the application of Retrieval-Augmented Generation (RAG) techniques using StackExchange data to answer questions related to neural networks.

## Table of Contents

- [Project Overview](#project-overview)
- [Data Collection](#data-collection)
- [Architecture](#architecture)
- [Getting Started](#getting-started)
- [Installation](#installation)
- [Usage](#usage)
- [Future Work](#future-work)
- [Contributing](#contributing)
- [License](#license)


## Project Overview
This chatbot leverages Retrieval-Augmented Generation (RAG) to answer questions about neural networks. It combines a retriever to find relevant answers from a curated dataset and a generator to deliver clear, concise responses, making it a valuable tool for users seeking foundational knowledge in neural networks.

## Data Collection
The data for this project was gathered using the StackExchange API and focused on fundamental questions related to neural networks. The collected answers vary in length. The Gemini 1.0 model was used to create a short summary of the answers.
You can find the data [here](https://github.com/hariprasath-v/Nnet101_Assistant/blob/main/data/Stackoverflow_data(neural_networks_stats)_pre_processed_Gemini_LLM.csv)

## Architecture
The chatbot architecture is built on a **Retrieval-Augmented Generation (RAG)** framework that combines vector-based search with language model generation for accurate and concise responses:

1. **Vector-Based Search with ElasticSearch**: User queries are processed as vector embeddings and searched against a database using **ElasticSearch** to retrieve the most relevant information.

2. **Prompt Creation**: The retrieved results are dynamically formatted into a structured prompt, which is fed into a **Large Language Model (LLM)**.

3. **Response Generation**: The LLM processes the prompt and generates a clear, concise summary tailored to answer the user’s question accurately.

