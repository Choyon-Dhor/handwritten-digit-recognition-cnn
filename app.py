from pathlib import Path
import numpy as np
from PIL import Image, ImageOps
import torch
import torch.nn as nn
import torch.nn.functional as F
import gradio as gr

try:
    import spaces
except ImportError:
    class spaces:
        @staticmethod
        def GPU(fn):
            return fn

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 64, 3)
        self.fc1 = nn.Linear(1600, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        # Match Keras (B, H, W, C) channel-last memory layout during flatten
        x = x.permute(0, 2, 3, 1).contiguous().view(-1, 1600)
        x = F.relu(self.fc1(x))
        return F.softmax(self.fc2(x), dim=1)

CKPT = Path(__file__).parent / "digit_cnn_pytorch.pt"
net = CNN()
net.load_state_dict(torch.load(CKPT, map_location="cpu"))
net.eval()

def to_pil(data: dict | np.ndarray | Image.Image | None) -> Image.Image | None:
    match data:
        case None:
            return None
        case Image.Image():
            return data
        case np.ndarray():
            return Image.fromarray(data.astype(np.uint8))
        case dict():
            layers = data.get("layers")
            layer = data.get("composite") or (layers[0] if layers else None) or data.get("background")
            return to_pil(layer)
        case _:
            raise TypeError(f"Unsupported image type: {type(data)}")

@spaces.GPU
def predict(sketch: dict | np.ndarray | Image.Image | None) -> tuple[dict[str, float], Image.Image]:
    img = to_pil(sketch)
    if img is None:
        return {str(i): 0.0 for i in range(10)}, Image.new("L", (28, 28), 0)

    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, (255, 255, 255))
        bg.paste(img, mask=img.split()[3])
        img = bg

    gray = ImageOps.invert(img.convert("L"))
    bbox = gray.getbbox()
    if not bbox:
        return {str(i): 0.0 for i in range(10)}, Image.new("L", (28, 28), 0)

    cropped = gray.crop(bbox)
    w, h = cropped.size
    scale = 20.0 / max(w, h)
    nw, nh = max(1, int(round(w * scale))), max(1, int(round(h * scale)))
    scaled = cropped.resize((nw, nh), Image.Resampling.BILINEAR)

    canvas = Image.new("L", (28, 28), 0)
    canvas.paste(scaled, ((28 - nw) // 2, (28 - nh) // 2))

    # Shift center of mass to (13.5, 13.5) to match MNIST distribution
    arr = np.array(canvas, dtype=np.float32)
    if (m := arr.sum()) > 0:
        y, x = np.indices(arr.shape)
        dx = max(-4, min(4, int(round(13.5 - (x * arr).sum() / m))))
        dy = max(-4, min(4, int(round(13.5 - (y * arr).sum() / m))))
        exp = ImageOps.expand(canvas, border=6, fill=0)
        canvas = exp.crop((6 - dx, 6 - dy, 34 - dx, 34 - dy))

    t = torch.from_numpy(np.array(canvas, dtype=np.float32) / 255.0).view(1, 1, 28, 28)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    net.to(dev)

    with torch.no_grad():
        probs = net(t.to(dev)).cpu().numpy()[0]

    return {str(i): float(probs[i]) for i in range(10)}, canvas

with gr.Blocks(title="Handwritten Digit Recognition") as demo:
    gr.HTML("""
    <div style="text-align: center; padding: 18px; background: rgba(99, 102, 241, 0.1); border: 1px solid rgba(99, 102, 241, 0.25); border-radius: 12px; margin-bottom: 20px;">
        <h1 style="font-size: 2rem; font-weight: 700; margin: 0 0 6px;">Handwritten Digit Recognition</h1>
        <p style="color: #94a3b8; margin: 0;">CNN Model &bull; 99.06% MNIST Accuracy &bull; ZeroGPU Active</p>
    </div>
    """)
    with gr.Row():
        with gr.Column(scale=1):
            pad = gr.Sketchpad(
                label="Draw Digit (0-9)",
                type="pil",
                image_mode="RGBA",
                canvas_size=(300, 300),
                brush=gr.Brush(default_size=20, colors=["#000000"])
            )
            with gr.Row():
                clear = gr.ClearButton(components=[pad], value="Clear")
                btn = gr.Button("Predict", variant="primary")
        with gr.Column(scale=1):
            out_label = gr.Label(num_top_classes=3, label="Predictions")
            out_img = gr.Image(label="MNIST Normalized (28x28)", height=150, width=150)

    btn.click(predict, inputs=[pad], outputs=[out_label, out_img])
    pad.change(predict, inputs=[pad], outputs=[out_label, out_img])

if __name__ == "__main__":
    demo.launch()
