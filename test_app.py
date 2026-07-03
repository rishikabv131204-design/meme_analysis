# =============================================
# 🎥 Two-Stage Meme Interpretation (Video Version)
# =============================================
import gradio as gr
from PIL import Image
import numpy as np
import easyocr, re, io, os, cv2, torch, torch.nn as nn, torch.nn.functional as F
from dotenv import load_dotenv
import google.generativeai as genai
from keybert import KeyBERT
from transformers import BertTokenizer, BertModel

print("🚀 Starting Meme Interpretation App")


# ------------------------------------------------
# 🔧 1. Setup
# ------------------------------------------------
load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
gemini_model = genai.GenerativeModel("gemini-2.0-flash")
ocr_reader = easyocr.Reader(['en'])
device = "cuda" if torch.cuda.is_available() else "cpu"
kw_model = KeyBERT()

# ------------------------------------------------
# 🔤 2. Helpers
# ------------------------------------------------
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

# ------------------------------------------------
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
        print("✅ Stage2Pipeline constructor running...")
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
print("✅ Stage2Pipeline initialized successfully!\n")

# ------------------------------------------------
# 🎞 4. Video Handling
# ------------------------------------------------
def extract_key_frame(path):
    cap = cv2.VideoCapture(path)
    tot = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.set(cv2.CAP_PROP_POS_FRAMES, tot//2)
    ok, frame = cap.read()
    cap.release()
    if not ok: return None
    return Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

# ------------------------------------------------
# 🔄 5. Full Pipeline
# ------------------------------------------------
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

# ------------------------------------------------
# 🎨 6. Custom CSS — Big Labels, Medium Textboxes
# ------------------------------------------------
custom_css = """
body { background-color:#ffedd5!important; }
h1 { font-size:46px!important; font-weight:900!important; color:#7c2d12!important; text-align:center!important; }

/* Larger label text */
label span[data-testid="block-info"] {
    font-size: 26px !important;
    font-weight: 800 !important;
    color: #4a1d0d !important;
    margin-bottom: 8px !important;
    display: block !important;
}

/* Container spacing */
label.container { padding-bottom: 6px !important; }

/* Textbox content style */
textarea[data-testid="textbox"] {
    font-size: 20px !important;
    font-weight: 600 !important;
    color: #111111 !important;
    height: 320px !important;
    max-height: 320px !important;
    min-height: 320px !important;
    overflow-y: auto !important;
    padding: 14px !important;
    border-radius: 10px !important;
    background-color: #fffdf8 !important;
}

/* Placeholder styling */
textarea[data-testid="textbox"]::placeholder {
    font-size: 17px !important;
    color: #8b8b8b !important;
}

/* Ensure consistent height */
label .input-container, .input-container.svelte-1ae7ssi {
    min-height: 320px !important;
}
"""

# ------------------------------------------------
# 🎛 7. Gradio Interface
# ------------------------------------------------
interface_video = gr.Interface(
    fn=process_video,
    inputs=gr.Video(label="Upload Meme Video"),
    outputs=[
        gr.Textbox(label="Generated Caption", elem_id="big-text", lines=3, placeholder="Generated Caption..."),
        gr.Textbox(label="Extracted Text", elem_id="big-text", lines=10, placeholder="Extracted OCR Text..."),
        gr.Textbox(label="Meme Explanation", elem_id="big-text", lines=8, placeholder="Meme Explanation..."),
        gr.Textbox(label="Sentiment & Emotion Analysis", elem_id="big-text", lines=4, placeholder="Sentiment & Emotion Analysis...")
    ],
    title="🎥 Meme Interpretation Pipeline",
    description="Upload a meme video — extracts a key frame, describes it, explains humor, and classifies tone.",
    css=custom_css
)

interface_image = gr.Interface(
    fn=process_image,
    inputs=gr.Image(type="pil", label="Upload Meme Image"),
    outputs=[
        gr.Textbox(label="Generated Caption", elem_id="big-text", lines=3, placeholder="Generated Caption..."),
        gr.Textbox(label="Extracted Text", elem_id="big-text", lines=10, placeholder="Extracted OCR Text..."),
        gr.Textbox(label="Meme Explanation", elem_id="big-text", lines=8, placeholder="Meme Explanation..."),
        gr.Textbox(label="Sentiment & Emotion Analysis", elem_id="big-text", lines=4, placeholder="Sentiment & Emotion Analysis...")
    ],
    title="🖼️ Meme Interpretation Pipeline",
    description="Upload a meme image — describes it, explains humor, and classifies tone.",
    css=custom_css
)

interface = gr.TabbedInterface(
    interface_list=[interface_image, interface_video],
    tab_names=["🖼️ Image Meme", "🎥 Video Meme"],
    title="🎥 Memesense"
)

if __name__ == "__main__":
    interface.launch(share=True)
