

# Use a lightweight Python image
FROM python:3.10-slim

# Prevent interactive prompts
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies (for audio & GUI packages)
RUN apt-get update && apt-get install -y \
    python3-dev \
    build-essential \
    portaudio19-dev \
    ffmpeg \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set working directory inside the container
WORKDIR /app

# Copy requirements.txt from project root
COPY requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy project files (excluding data files via .dockerignore)
COPY . .

# NLTK setup
RUN python -m nltk.downloader punkt

# Command to run the main chatbot UI
CMD ["python", "src/UI_Chatbot_Interface.py"]
