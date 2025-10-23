# =============================================
# 🎥🖼️ Two-Stage Meme Interpretation (Video + Image)
# =============================================

import gradio as gr
from PIL import Image
import numpy as np
import easyocr
import re
import io
import os
import cv2
from dotenv import load_dotenv
import google.generativeai as genai
from keybert import KeyBERT
from transformers import BertTokenizer, BertModel
import torch
import torch.nn as nn
import torch.nn.functional as F

# -----------------------------
# 🔧 Setup
# -----------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
ocr_reader = easyocr.Reader(['en'])
device = "cuda" if torch.cuda.is_available() else "cpu"

# -----------------------------
# 🔤 OCR Text Cleaning
# -----------------------------
def clean_ocr_text(text):
    text = re.sub(r"[^\x20-\x7E\n]", "", text)
    text = re.sub(r"\s+\n", "\n", text)
    text = re.sub(r"\n\s+", "\n", text)
    text = re.sub(r" +", " ", text)
    return text.strip()

# -----------------------------
# 🖼️ Caption Generation
# -----------------------------
def generate_caption(image):
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_bytes = img_byte_arr.getvalue()
    try:
        response = gemini_model.generate_content([
            {"mime_type": "image/png", "data": img_bytes},
            {"text": "Describe this image in a single sentence."}
        ])
        return response.text.strip()
    except Exception as e:
        return f"Error generating caption: {str(e)}"

# -----------------------------
# 💬 Meme Explanation
# -----------------------------
def explain_meme(caption, meme_text):
    prompt = f"""
You are a meme interpretation AI. Below is an image caption and meme text.

Image Caption: "{caption}"
Meme Text: "{meme_text}"

Explain the relationship between the caption and the meme text in a short paragraph.
Identify if it's humorous, exaggerated, sarcastic, or ironic.
"""
    try:
        response = gemini_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Error from Gemini:\n{str(e)}"

# -----------------------------
# 🧠 KeyBERT + Transformer + MLP
# -----------------------------
kw_model = KeyBERT()

def extract_keyphrases(text, num_phrases=5):
    phrases = kw_model.extract_keywords(text, top_n=num_phrases)
    return [phrase[0] for phrase in phrases]

class UETBlock(nn.Module):
    def __init__(self, hidden_size=768):
        super(UETBlock, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU()
        )
        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        self.decoder = nn.Sequential(
            nn.Conv1d(hidden_size, hidden_size, 3, padding=1),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(2, 0, 1).permute(1, 0, 2)
        attn_mask = torch.ones(x.shape[:-1], dtype=torch.long, device=x.device)
        out = self.transformer(inputs_embeds=x, attention_mask=attn_mask).last_hidden_state
        out = out.permute(0, 2, 1)
        out = self.decoder(out)
        return out

class MultiTaskClassifier(nn.Module):
    def __init__(self, hidden_size=768):
        super(MultiTaskClassifier, self).__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        self.sentiment = nn.Linear(hidden_size, 3)
        self.emotion = nn.Linear(hidden_size, 5)
        self.humor = nn.Linear(hidden_size, 2)
        self.sarcasm = nn.Linear(hidden_size, 2)
        self.offensive = nn.Linear(hidden_size, 2)
        self.motivation = nn.Linear(hidden_size, 2)

    def forward(self, x):
        x = F.relu(self.linear(x))
        return {
            "sentiment": self.sentiment(x),
            "emotion": self.emotion(x),
            "humor": self.humor(x),
            "sarcasm": self.sarcasm(x),
            "offensive": self.offensive(x),
            "motivation": self.motivation(x)
        }

class Stage2Pipeline:
    def __init__(self):
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.uet = UETBlock().to(device)
        self.mtl = MultiTaskClassifier().to(device)

    def forward(self, explanation_text):
        keyphrases = extract_keyphrases(explanation_text)
        input_text = " ".join(keyphrases)
        inputs = self.tokenizer(input_text, return_tensors="pt", padding=True, truncation=True).to(device)
        with torch.no_grad():
            embeddings = self.uet.transformer(**inputs).last_hidden_state
        uet_input = embeddings.permute(0, 2, 1)
        features = self.uet(uet_input)
        pooled = torch.mean(features, dim=2)
        predictions = self.mtl(pooled)
        return predictions

stage2_model = Stage2Pipeline()

# -----------------------------
# 🎞️ Key Frame Extraction
# -----------------------------
def extract_key_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    key_frame_index = total_frames // 2
    cap.set(cv2.CAP_PROP_POS_FRAMES, key_frame_index)
    success, frame = cap.read()
    cap.release()
    if not success:
        return None
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    return Image.fromarray(frame_rgb)

# -----------------------------
# 🔄 Combined Processor
# -----------------------------
def process_input(image=None, video=None):
    if image is None and video is None:
        return "Please upload an image or a video.", "N/A", "N/A", "N/A"

    # If video given, extract a key frame
    if video is not None:
        frame = extract_key_frame(video)
        if frame is None:
            return "Could not extract frame from video.", "N/A", "N/A", "N/A"
        img = frame
    else:
        img = image

    # Caption + OCR + Explanation
    caption = generate_caption(img)
    img_np = np.array(img)
    ocr_result = ocr_reader.readtext(img_np)
    extracted_text = "\n".join([entry[1] for entry in ocr_result])
    cleaned_text = clean_ocr_text(extracted_text)
    explanation = explain_meme(caption, cleaned_text)

    # Stage 2 prediction
    try:
        preds = stage2_model.forward(explanation)
        label_map = lambda t: torch.argmax(torch.softmax(t, dim=1)).item()
        result_str = "\n".join([
            f"Sentiment: {label_map(preds['sentiment'])}",
            f"Emotion: {label_map(preds['emotion'])}",
            f"Humor: {label_map(preds['humor'])}",
            f"Sarcasm: {label_map(preds['sarcasm'])}",
            f"Offensive: {label_map(preds['offensive'])}",
            f"Motivation: {label_map(preds['motivation'])}",
        ])
    except Exception as e:
        result_str = f"Error in Stage 2: {str(e)}"

    return caption, cleaned_text, explanation, result_str

# -----------------------------
# 🎛️ Gradio UI
# -----------------------------
interface = gr.Interface(
    fn=process_input,
    inputs=[
        gr.Image(label="Upload Image (optional)", type="pil"),
        gr.Video(label="Upload Video (optional)")
    ],
    outputs=[
        gr.Textbox(label="Generated Caption"),
        gr.Textbox(label="Extracted Text"),
        gr.Textbox(label="Meme Explanation"),
        gr.Textbox(label="Sentiment & Emotion Analysis")
    ],
    title="🎥🖼️ Meme Interpretation (Image + Video)",
    description="Upload a meme image or video — the model extracts visuals, describes them, explains humor, and classifies tone."
)

if __name__ == "__main__":
    interface.launch(share=True)
