import gradio as gr
import numpy as np
from PIL import Image
import torch
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import uvicorn
from pathlib import Path
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse
import torch

app = FastAPI()
ckpt = torch.load("tinyresnet1d.pt", map_location="cpu")
print(type(ckpt))


ckpt = torch.load("tinyresnet1d.pt", map_location="cpu")
print(type(ckpt))

def predict(img, conf, imgsz):

    if img is None:
        return None, "Waiting for image..."

    if isinstance(img, np.ndarray):
        img = Image.fromarray(img)

    img = img.convert("RGB")

    return img, "TinyResNet loaded successfully"


def webcam_live():
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #06b6d4, #8b5cf6)
    }
    /* Target the unique ID */
    #submit-btn > button {
        background: linear-gradient(90deg, #5c6bc0, #8e24aa);
        color: white;
        font-size: 18px;
        padding: 14px 32px;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    #submit-btn > button:hover {
        transform: scale(1.05);
        filter: brightness(1.1);
    }

    /* Target a reusable class */
    .preview-card {
        background-color: #1e1e2f;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    """
    with gr.Blocks(css = custom_css, title="Live webcam") as demo:
        with gr.Row():
            cam = gr.Image(
                sources=["webcam"],   # use the webcam
                streaming=True,       # enable streaming
                type="numpy",
                image_mode="RGB",
                label="📷 Live webcam feed",
                elem_classes=["preview-card"]
            )
            out_img = gr.Image(label="Annotated", elem_classes=["preview-card"])
        with gr.Accordion("⚙ Detection Settings", open=False, elem_classes = ["preview-card"]):
            conf  = gr.Slider(0.01, 0.9, value=0.25, step=0.01, label="Confidence")
            imgsz = gr.Slider(320, 1024, value=640, step=32, label="Image size")
            out_txt = gr.Textbox(label="🧠 Detection result", elem_classes = ["preview-card"])
        cam.stream(predict, inputs=[cam, conf, imgsz], outputs=[out_img, out_txt], time_limit=0.5)
    return demo

def upload_manual():
    custom_css = """
    .gradio-container {
        background: linear-gradient(135deg, #06b6d4, #8b5cf6)
    }
    /* Target the unique ID */
    #submit-btn > button {
        background: linear-gradient(90deg, #5c6bc0, #8e24aa);
        color: white;
        font-size: 18px;
        padding: 14px 32px;
        border: none;
        border-radius: 12px;
        box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
        transition: all 0.2s ease;
    }
    #submit-btn > button:hover {
        color: #06b6d4;
        transform: scale(1.05);
        filter: brightness(1.1);
    }

    /* Target a reusable class */
    .preview-card {
        background-color: #1e1e2f;
        border-radius: 16px;
        padding: 16px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.3);
    }
    """
    with gr.Blocks(css = custom_css, title="Image Upload") as demo:
        with gr.Row():
            img = gr.Image(
                sources=["upload"],   # file upload
                type="pil",
                image_mode="RGB",
                label="Upload an image",
                elem_classes=["preview-card"]
            )
            out_img = gr.Image(label="Annotated (Upload)", elem_classes = ["preview-card"])
        with gr.Accordion("⚙ Detection Settings", open=False, elem_classes=["preview-card"]):
            conf  = gr.Slider(0.01, 0.9, value=0.25, step=0.01, label="Confidence")
            imgsz = gr.Slider(320, 1024, value=640, step=32, label="Image size")
        out_txt = gr.Textbox(label="🧠 Detection result", elem_classes=["preview-card"])
        btn = gr.Button("Submit", elem_id="submit-btn")
        btn.click(predict, inputs=[img, conf, imgsz], outputs=[out_img, out_txt])
    return demo
    

app = gr.mount_gradio_app(app, upload_manual(), path="/upload")
app = gr.mount_gradio_app(app, webcam_live(), path="/webcam")


BASE_DIR = Path(__file__).parent
app.mount("/home_assets",   StaticFiles(directory=BASE_DIR / "home"),   name="home_assets")
app.mount("/upload_assets", StaticFiles(directory=BASE_DIR / "upload"), name="upload_assets")
app.mount("/webcam_assets", StaticFiles(directory=BASE_DIR / "webcam"), name="webcam_assets")


# Serve HTML routes
@app.get("/", response_class=RedirectResponse)
def root():
    return RedirectResponse(url="/home.html")

@app.get("/home.html", response_class=FileResponse)
def home():
    return FileResponse(BASE_DIR / "home/home.html")

@app.get("/upload.html", response_class=FileResponse)
def upload():
    return FileResponse(BASE_DIR / "upload/upload.html")

@app.get("/webcam.html", response_class=FileResponse)
def webcam():
    return FileResponse(BASE_DIR / "webcam/webcam.html")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)

