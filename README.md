# Meme Interpretation Pipeline

A two-stage AI pipeline for analyzing memes using computer vision, OCR, and natural language processing.

## Features

- **Stage 1**: Image captioning and text extraction using Gemini AI and EasyOCR
- **Stage 2**: Multi-task classification for sentiment, emotion, humor, sarcasm, and more
- **Web Interface**: User-friendly Gradio interface for easy interaction

## Setup (For Beginners)

Follow these exact steps to get the app running on your machine:

### 1. Clone the Repository
Open your terminal or command prompt and run:
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment
It is highly recommended to create an isolated environment so packages don't conflict with your system.
**On Windows:**
```bash
python -m venv .venv
.\.venv\Scripts\activate
```
**On Mac/Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
With your virtual environment activated, install all the required packages:
```bash
pip install -r requirements.txt
```
*(Note: We have strictly pinned stable versions of FastAPI, Pydantic, and Gradio to prevent common bugs).*

### 4. Get a Google API Key (Gemini)
1. Visit [Google AI Studio](https://aistudio.google.com/app/apikey)
2. Create a new API key.
3. Copy the API key to your clipboard.

### 5. Configure your Environment
Create a new file named exactly `.env` in the project folder and paste your key inside it like this:
```
GOOGLE_API_KEY=your_actual_google_api_key_here
```

### 6. Run the Application!
Start the server by running:
```bash
python v.py
```
*(Or `python test_app.py`)*

The application will start, and you can view the web interface in your browser at `http://127.0.0.1:7860`

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
