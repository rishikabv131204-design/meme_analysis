
#  Two-Stage Meme Interpretation (Video Version)

import gradio as gr
from PIL import Image
import numpy as np
import easyocr, re, io, os, cv2, torch, torch.nn as nn, torch.nn.functional as F
from dotenv import load_dotenv
import google.generativeai as genai
from keybert import KeyBERT
from transformers import BertTokenizer, BertModel

print("Starting Meme Interpretation App")


#  1. Setup

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.5-flash")
ocr_reader = easyocr.Reader(['en'])
device = "cuda" if torch.cuda.is_available() else "cpu"
kw_model = KeyBERT()


#  2. Helpers

def clean_ocr_text(t):
    t = re.sub(r"[^\x20-\x7E\n]", "", t)
    t = re.sub(r"\s+\n", "\n", t)
    t = re.sub(r"\n\s+", "\n", t)
    t = re.sub(r" +", " ", t)
    return t.strip()

def generate_caption(image):
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    try:
        r = gemini_model.generate_content([
            {"mime_type": "image/png", "data": buf.getvalue()},
            {"text": "Describe this image in a single sentence."}
        ])
        return r.text.strip()
    except Exception as e:
        return f"Error: {e}"

def explain_meme(caption, text):
    prompt = f"""
You are a meme interpretation AI.

Image Caption: "{caption}"
Meme Text: "{text}"

Explain the relationship between them in one paragraph, noting if it's humorous, sarcastic, or ironic.
"""
    try:
        r = gemini_model.generate_content(prompt)
        return r.text.strip()
    except Exception as e:
        return f"Error: {e}"

def extract_keyphrases(text, n=5):
    return [p[0] for p in kw_model.extract_keywords(text, top_n=n)]


#  3. Stage-2 Model

class UETBlock(nn.Module):
    def __init__(self, hidden=768):
        super().__init__()
        self.encoder = nn.Sequential(nn.Conv1d(hidden, hidden, 3, 1, 1), nn.ReLU())
        self.transformer = BertModel.from_pretrained("bert-base-uncased")
        self.decoder = nn.Sequential(nn.Conv1d(hidden, hidden, 3, 1, 1), nn.ReLU())
    def forward(self, x):
        x = self.encoder(x)
        x = x.permute(2,0,1).permute(1,0,2)
        attn = torch.ones(x.shape[:-1], dtype=torch.long, device=x.device)
        o = self.transformer(inputs_embeds=x, attention_mask=attn).last_hidden_state
        o = o.permute(0,2,1)
        return self.decoder(o)

class MultiTaskClassifier(nn.Module):
    def __init__(self, hidden=768):
        super().__init__()
        self.lin = nn.Linear(hidden, hidden)
        self.sentiment = nn.Linear(hidden, 3)
        self.emotion   = nn.Linear(hidden, 5)
        self.humor     = nn.Linear(hidden, 2)
        self.sarcasm   = nn.Linear(hidden, 2)
        self.offensive = nn.Linear(hidden, 2)
        self.motivation= nn.Linear(hidden, 2)
    def forward(self, x):
        x = F.relu(self.lin(x))
        return {
            "sentiment": self.sentiment(x),
            "emotion":   self.emotion(x),
            "humor":     self.humor(x),
            "sarcasm":   self.sarcasm(x),
            "offensive": self.offensive(x),
            "motivation":self.motivation(x)
        }

class Stage2Pipeline:
    def __init__(self):
        print("Stage2Pipeline constructor running...")
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.uet = UETBlock().to(device)
        self.mtl = MultiTaskClassifier().to(device)
    def forward(self, text):
        keys = extract_keyphrases(text)
        input_text = " ".join(keys) or text
        inputs = self.tokenizer(input_text, return_tensors="pt",
                                padding=True, truncation=True).to(device)
        with torch.no_grad():
            emb = self.uet.transformer(**inputs).last_hidden_state
        feats = self.uet(emb.permute(0,2,1))
        pooled = torch.mean(feats, dim=2)
        return self.mtl(pooled)

stage2_model = Stage2Pipeline()
print("Stage2Pipeline initialized successfully!\n")


#  4. Video Handling

def extract_key_frame(path):
    cap = cv2.VideoCapture(path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, tot//2)
    ok, frame = cap.read()
    cap.release()
    if not ok: return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))


# 🔄 5. Full Pipeline

def process_image(image):
    if image is None:
        return "Upload an image.","N/A","N/A","N/A"
    
    caption = generate_caption(image)
    ocr = ocr_reader.readtext(np.array(image))
    text = clean_ocr_text("\n".join([t[1] for t in ocr]))
    explanation = explain_meme(caption, text)

    try:
        preds = stage2_model.forward(explanation)
        choose = lambda t: torch.argmax(torch.softmax(t, dim=1)).item()
        result = "\n".join([
            f"Sentiment:  {choose(preds['sentiment'])}",
            f"Emotion:    {choose(preds['emotion'])}",
            f"Humor:      {choose(preds['humor'])}",
            f"Sarcasm:    {choose(preds['sarcasm'])}",
            f"Offensive:  {choose(preds['offensive'])}",
            f"Motivation: {choose(preds['motivation'])}"
        ])
    except Exception as e:
        result = f"Error in Stage 2: {e}"

    return caption, text, explanation, result


def process_video(video):
    if video is None:
        return "Upload a video.","N/A","N/A","N/A"
    frame = extract_key_frame(video)
    if frame is None:
        return "No frame found","N/A","N/A","N/A"

    caption = generate_caption(frame)
    ocr = ocr_reader.readtext(np.array(frame))
    text = clean_ocr_text("\n".join([t[1] for t in ocr]))
    explanation = explain_meme(caption, text)

    try:
        preds = stage2_model.forward(explanation)
        choose = lambda t: torch.argmax(torch.softmax(t, dim=1)).item()
        result = "\n".join([
            f"Sentiment:  {choose(preds['sentiment'])}",
            f"Emotion:    {choose(preds['emotion'])}",
            f"Humor:      {choose(preds['humor'])}",
            f"Sarcasm:    {choose(preds['sarcasm'])}",
            f"Offensive:  {choose(preds['offensive'])}",
            f"Motivation: {choose(preds['motivation'])}"
        ])
    except Exception as e:
        result = f"Error in Stage 2: {e}"

    return caption, text, explanation, result


#  6. Final UI — Clean Upload + Full Gradient BG + Title


custom_css = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;900&display=swap');

* {
    font-family: 'Inter', sans-serif !important;
    transition: all 0.3s ease-in-out !important;
}

/* === FULL PAGE GRADIENT BACKGROUND === */
html, body {
    height: 100% !important;
    background: linear-gradient(135deg, #e0f7ff 0%, #b3e5fc 40%, #81d4fa 100%) !important;
    margin: 0 !important;
}

/* === CONTAINER === */
.gradio-container {
    max-width: 1300px !important;
    margin: 0 auto !important;
    padding: 50px 20px !important;
}

/* === HEADER === */
h1 {
    text-align: center !important;
    font-size: 54px !important;
    font-weight: 900 !important;
    color: #01579b !important;
    letter-spacing: -0.5px !important;
    text-shadow: 2px 2px 6px rgba(1, 87, 155, 0.15) !important;
    margin-bottom: 15px !important;
}

.subtitle {
    text-align: center !important;
    font-size: 18px !important;
    color: #0288d1 !important;
    margin-bottom: 40px !important;
}

/* === SIMPLE UPLOAD AREA === */
.upload-container {
    background: white !important;
    border: 2px solid #81d4fa !important;
    border-radius: 20px !important;
    padding: 30px !important;
    text-align: center !important;
    box-shadow: 0 10px 30px rgba(2, 132, 199, 0.12) !important;
    transition: all 0.3s ease !important;
}

.upload-container:hover {
    border-color: #0288d1 !important;
    box-shadow: 0 14px 36px rgba(2, 132, 199, 0.2) !important;
    transform: translateY(-3px) !important;
}

.upload-container label span {
    font-size: 18px !important;
    font-weight: 700 !important;
    color: #0284c7 !important;
}

/* === BUTTONS === */
.big-buttons {
    display: flex !important;
    justify-content: center !important;
    align-items: center !important;
    gap: 25px !important;
    margin-top: 25px !important;
    margin-bottom: 40px !important;
}

.analyze-btn {
    background: linear-gradient(135deg, #0288d1, #03a9f4) !important;
    color: white !important;
    font-weight: 700 !important;
    font-size: 17px !important;
    padding: 18px 50px !important;
    border-radius: 16px !important;
    border: none !important;
    box-shadow: 0 6px 22px rgba(2, 136, 209, 0.3) !important;
}

.analyze-btn:hover {
    background: linear-gradient(135deg, #0277bd, #0288d1) !important;
    transform: translateY(-3px) !important;
}

.clear-btn {
    background: white !important;
    color: #0288d1 !important;
    border: 2px solid #81d4fa !important;
    font-weight: 700 !important;
    font-size: 17px !important;
    padding: 18px 50px !important;
    border-radius: 16px !important;
    box-shadow: 0 6px 20px rgba(2, 136, 209, 0.15) !important;
}

.clear-btn:hover {
    background: #e1f5fe !important;
    box-shadow: 0 8px 24px rgba(2, 136, 209, 0.25) !important;
}

/* === OUTPUT CARDS === */
.glass {
    background: rgba(255, 255, 255, 0.75) !important;
    backdrop-filter: blur(14px) saturate(180%) !important;
    border-radius: 22px !important;
    border: 1px solid rgba(255, 255, 255, 0.3) !important;
    box-shadow: 0 10px 40px rgba(3, 105, 161, 0.15) !important;
}

.output-grid {
    display: grid !important;
    grid-template-columns: repeat(auto-fit, minmax(480px, 1fr)) !important;
    gap: 30px !important;
    margin-top: 30px !important;
}

/* === TEXTAREAS === */
textarea {
    background: rgba(236, 249, 255, 0.9) !important;
    border: 2px solid #b3e5fc !important;
    border-radius: 18px !important;
    padding: 18px !important;
    color: #01579b !important;
    font-size: 15px !important;
    min-height: 140px !important;
}

footer {
      !important;
}
"""

# Build Updated Interface
with gr.Blocks(css=custom_css, theme=gr.themes.Soft(primary_hue="blue", neutral_hue="gray")) as interface:
    
    #  Set custom browser tab title
    interface.title = "🎥 Memesense"

    gr.HTML("<h1>🎥 Memesense</h1>")
    # gr.HTML("<p class='subtitle'>Decode Humor, Emotion & Context from Meme Videos Instantly</p>")

    # Tabbed Upload Area
    with gr.Tabs():
        with gr.Tab("🖼️ Image Meme"):
            with gr.Row():
                with gr.Column(elem_classes="upload-container"):
                    image_input = gr.Image(
                        type="pil",
                        label="📤 Upload Meme Image",
                        elem_classes="upload-box"
                    )
            with gr.Row(elem_classes="big-buttons"):
                analyze_img_btn = gr.Button("🔍 Analyze Image Meme", elem_classes="analyze-btn")
                clear_img_btn = gr.Button("🧹 Clear", elem_classes="clear-btn")

        with gr.Tab("🎥 Video Meme"):
            with gr.Row():
                with gr.Column(elem_classes="upload-container"):
                    video_input = gr.Video(
                        label="📤 Upload Meme Video",
                        elem_classes="upload-box"
                    )
            with gr.Row(elem_classes="big-buttons"):
                analyze_vid_btn = gr.Button("🔍 Analyze Video Meme", elem_classes="analyze-btn")
                clear_vid_btn = gr.Button("🧹 Clear", elem_classes="clear-btn")

    # Output Layout
    with gr.Row(elem_classes="output-grid"):
        with gr.Column(elem_classes="glass"):
            caption_output = gr.Textbox(
                label="🖼️ Generated Caption",
                lines=5, interactive=False,
                placeholder="AI-generated caption for your meme will appear here..."
            )
        with gr.Column(elem_classes="glass"):
            text_output = gr.Textbox(
                label="📝 OCR Extracted Text",
                lines=5, interactive=False,
                placeholder="Text detected from your meme video..."
            )

    with gr.Row(elem_classes="output-grid"):
        with gr.Column(elem_classes="glass"):
            explanation_output = gr.Textbox(
                label="💡 Meme Explanation",
                lines=7, interactive=False,
                placeholder="AI interpretation of humor, irony, and meaning..."
            )
        with gr.Column(elem_classes="glass"):
            analysis_output = gr.Textbox(
                label="📊 Sentiment & Emotion Analysis",
                lines=7, interactive=False,
                placeholder="Sentiment, emotion, sarcasm, and motivation details..."
            )

    # Button Connections
    analyze_vid_btn.click(
        fn=process_video,
        inputs=video_input,
        outputs=[caption_output, text_output, explanation_output, analysis_output]
    )
    
    analyze_img_btn.click(
        fn=process_image,
        inputs=image_input,
        outputs=[caption_output, text_output, explanation_output, analysis_output]
    )

    def clear_all():
        return None, None, "", "", "", ""

    clear_vid_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[video_input, image_input, caption_output, text_output, explanation_output, analysis_output]
    )
    clear_img_btn.click(
        fn=clear_all,
        inputs=[],
        outputs=[video_input, image_input, caption_output, text_output, explanation_output, analysis_output]
    )

if __name__ == "__main__":
    interface.launch(share=True)
