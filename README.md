# RAG Converter

A tool for preparing documents for RAG (Retrieval-Augmented Generation) systems.
Part of "Dirk's RAG Suite" - Developed by Ai-engineering.ai (Dirk Wonhöfer)

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://rag-converter.streamlit.app)

## Features

- Support for various document formats (PDF, Markdown, Text, HTML)
- Intelligent document processing and text extraction
- Various chunking strategies:
  - Semantic chunking (based on sentence structures)
  - Length-based chunking (with intelligent breaks)
  - Structure-based chunking (based on document structure)
- Flexible export options (JSON, JSONL, CSV, Markdown)
- User-friendly web interface
- Live preview of chunks
- Progress indicator during processing
- Smart document structuring and topic analysis
- Direct export to vector databases (Pinecone, Weaviate, ChromaDB)

## Online Usage

You can use the hosted version of RAG Converter at:
[https://rag-converter.streamlit.app](https://rag-converter.streamlit.app)

## Local Installation

### Quick Start (Windows)

1. Run `setup.bat`. This will:
   
   - Install Python (if not already present)
   - Install all necessary dependencies
   - Create a desktop shortcut
   - Download the language model

2. Click on the "RAG Converter" desktop shortcut

3. The browser will automatically open with the user interface

### Manual Installation

If you prefer manual installation:

1. Make sure Python 3.10+ is installed

2. Create a virtual Python environment:
   
   ```bash
   python -m venv venv
   ```

3. Activate the virtual environment:
- Windows:
  
  ```bash
  .\venv\Scripts\activate
  ```

- Linux/Mac:
  
  ```bash
  source venv/bin/activate
  ```
4. Install the dependencies:
   
   ```bash
   pip install -r requirements.txt
   ```

5. Install the English language model:
   
   ```bash
   python -m spacy download en_core_web_sm
   ```

6. Start the application:
   
   ```bash
   streamlit run src/main.py
   ```

## Usage

1. Open the application in your browser (default at http://localhost:8501)

2. Upload your documents:
   
   - Supported formats: PDF, Markdown, Text, HTML
   - Multiple documents simultaneously possible
   - Maximum file size: 200MB per file

3. Select chunking settings:
   
   - Chunking strategy
   - Chunk size (100-2000 characters)
   - Overlap (0-50%)

4. Choose the export format:
   
   - JSON (structured data with metadata)
   - JSONL (line-by-line JSON entries)
   - CSV (tabular format)
   - Markdown (formatted text)
   - Vector DB (direct export to vector databases)

5. Click "Process Documents"

6. Check the results in the preview

7. Download the processed chunks

## Deployment

### Deploy to Streamlit Cloud

1. Fork this repository to your GitHub account

2. Go to [Streamlit Cloud](https://streamlit.io/cloud)

3. Click "New app" and select this repository

4. Set the following:
   
   - Main file path: `src/main.py`
   - Python version: 3.10.11

5. Click "Deploy"

### Environment Variables

For vector database integration, set the following environment variables in Streamlit Cloud:

- `OPENAI_API_KEY` (optional, for Weaviate hybrid search)
- `PINECONE_API_KEY` (if using Pinecone)
- `PINECONE_ENVIRONMENT` (if using Pinecone)

## Chunking Strategies in Detail

### Semantic Chunking

- Uses AI-based language processing
- Considers sentence structures and relationships
- Ideal for natural texts
- Maintains context between chunks

### Length-based Chunking

- Divides text into uniform pieces
- Intelligent: Breaks at sentence and paragraph boundaries
- Configurable chunk size
- Good for uniform processing

### Structure-based Chunking

- Uses existing document structure
- Recognizes headings and sections
- Ideal for structured documents
- Maintains document hierarchy

## License

Copyright (c) 2024 Ai-engineering.ai - Dirk Wonhöfer

All rights reserved. This project is part of "Dirk's RAG Suite" and may only be used, copied, or modified with express permission. 