# Meme Interpretation Pipeline

A two-stage AI pipeline for analyzing memes using computer vision, OCR, and natural language processing.

## Features

- **Stage 1**: Image captioning and text extraction using Gemini AI and EasyOCR
- **Stage 2**: Multi-task classification for sentiment, emotion, humor, sarcasm, and more
- **Web Interface**: User-friendly Gradio interface for easy interaction

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Or run the setup script:
```bash
python setup.py
```

### 2. Get Google API Key

1. Visit [Google AI Studio](https://makersuite.google.com/app/apikey)
2. Create a new API key
3. Copy the API key

### 3. Configure Environment

Create a `.env` file in the project directory:
```
GOOGLE_API_KEY=your_google_api_key_here
```

### 4. Run the Application

```bash
python v.py
```

The application will start a web interface accessible at `http://localhost:7860`

## Usage

1. Upload a meme image using the web interface
2. The system will:
   - Generate a caption for the image
   - Extract text from the image using OCR
   - Explain the relationship between the caption and text
   - Classify the meme's sentiment, emotion, humor level, etc.

## Requirements

- Python 3.8+
- Google API key for Gemini
- Internet connection for model downloads

## Troubleshooting

- **API Key Error**: Make sure your Google API key is correctly set in the `.env` file
- **Model Download Issues**: Ensure you have a stable internet connection for downloading BERT models
- **Memory Issues**: The application requires significant RAM for the transformer models

## Architecture

- **Stage 1**: Gemini AI for image captioning + EasyOCR for text extraction
- **Stage 2**: KeyBERT for keyphrase extraction + Custom UET transformer + Multi-task classifier
