# Modular RAG Document Assistant
## A Learning Lab for Retrieval Augmented Generation (RAG)

This repository implements a modular Retrieval Augmented Generation (RAG) system designed for experimentation, learning, and demonstration of modern document question answering pipelines.

The project allows users to explore different document ingestion methods, chunking strategies, embedding models, retrieval methods, and language models through an interactive interface.

The goal is to make RAG pipelines transparent, configurable, and educational so that anyone studying LLM systems can understand how each component affects performance.

The final interface is implemented using Streamlit, allowing users to interactively configure the pipeline and observe how answers are generated.

## Project Goals

This project helps users understand:

• How modern RAG pipelines work
• How different chunking strategies affect retrieval quality
• How embedding models influence semantic search
• How retrieval strategies (dense, sparse, hybrid) behave
• How LLMs generate answers using retrieved context

Instead of providing a fixed pipeline, this project allows users to experiment with different components.

## RAG Pipeline Overview

Document Sources
↓
Document Loader
↓
Text Chunking
↓
Embedding Model
↓
Vector Database
↓
Retriever
↓
(optional) Reranker
↓
LLM Generator
↓
Final Answer with Citations

Each step of this pipeline is modular and configurable.

## Features
Modular Architecture
Pipeline Stage	Available Options
Document Loading	Local files, Web pages, GitHub repositories
Chunking	Fixed, Recursive, Sliding Window, Semantic
Embeddings	Sentence Transformers, OpenAI
Vector Database	Chroma
Retrieval	Dense, Sparse (BM25), Hybrid
Generation	OpenAI, Ollama
Reranking	Optional
Evaluation	Basic evaluation utilities

## Repository Structure

rag-document-assistant

app/
   app.py

data/
   raw_docs/
   chroma_db/
   chroma_uploads/
   benchmarks/

examples/

notebooks/

src/

   loaders/
   chunkers/
   embeddings/
   vectordb/
   retrievers/
   generators/
   rerankers/
   evaluators/
   utils/

   config.py
   registry.py
   pipeline.py

src_legacy/

requirements.txt
README.md

## Installation

### Clone the repository

git clone https://github.com/Kollipara-Hema/rag-document-assistant.git
cd rag-document-assistant


### Create environment

python -m venv .venv
source .venv/bin/activate


### Install dependencies

pip install -r requirements.txt

## Running the Application

Start the Streamlit interface:

streamlit run app/app.py


This launches the interactive interface where users can configure the pipeline and ask questions.

## Supported Document Sources

• Local files (markdown, txt, json, yaml)
• Web pages via URL
• GitHub repositories
• Uploaded documents through UI

Future extensions may include:

• PDF ingestion pipelines
• database connectors
• API based document ingestion

## Chunking Strategies

Fixed Chunking

Recursive Chunking

Sliding Window Chunking

Semantic Chunking

## Embedding Models

Sentence Transformers (local models)

OpenAI Embeddings

## Vector Database

Chroma (local vector store)

Future support may include FAISS, Pinecone, or Weaviate.

## Retrieval Methods

Dense Retrieval

Sparse Retrieval (BM25)

Hybrid Retrieval

## Language Models

OpenAI LLMs

Ollama local models

## Evaluation Utilities

Evaluation modules help measure:

• retrieval quality
• answer relevance
• pipeline performance

## Learning Notebooks

The notebooks folder contains experiments exploring:

• chunking strategies
• embedding comparisons
• retrieval benchmarking
• RAG pipeline behavior

## Educational Design

This repository is structured as a learning resource.

Modules contain comments explaining:

• why a component is used
• how it works
• alternative implementations used in industry

## Future Improvements

Possible extensions include:

• reranking models using cross encoders
• RAG evaluation frameworks such as RAGAS
• agent pipelines using LangGraph
• automated retrieval benchmarking

## Author

Hema Sri Sai Kollipara
AI / Machine Learning Engineer
PhD in Statistics – Michigan State University

## License

Educational and research use.