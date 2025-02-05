#!/bin/bash

# Make the script executable
chmod +x setup.sh

# Install system dependencies
apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    libmagic1

# Install Python dependencies
pip install -r requirements.txt

# Install spaCy model
python -m spacy download en_core_web_sm 