from flask import Flask, render_template, request, jsonify
import base64, io
import numpy as np
from PIL import Image
import torch

from model import Net, idx_to_char

# --- Config knobs (keep train + web consistent) ---
INVERT = False  # canvas is black bg + white strokes -> usually False
THRESH = 0.10   # crop threshold on [0..1] pixels
# --------------------------------------------------

app = Flask(__name__)

model = Net()
model.load_state_dict(torch.load("model.pt", map_location="cpu"))
model.eval()

def emnist_fix_orientation_np(arr: np.ndarray) -> np.ndarray:
    """Match EMNIST orientation fix used in training:
    transpose H/W then flip vertically.
    """
    arr = arr.T
    arr = np.flipud(arr)
    return arr.copy()

def crop_pad_resize(arr: np.ndarray, thresh: float = 0.10) -> np.ndarray:
    """Crop to content on original resolution, pad to square, then resize to 28x28."""
    ys, xs = np.where(arr > thresh)
    if len(xs) == 0 or len(ys) == 0:
        img28 = Image.fromarray((arr * 255).astype("uint8")).resize((28, 28), Image.BILINEAR)
        return np.array(img28).astype("float32") / 255.0

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()
    crop = arr[y0:y1+1, x0:x1+1]

    h, w = crop.shape
    s = max(h, w)
    pad_y = (s - h) // 2
    pad_x = (s - w) // 2

    square = np.pad(
        crop,
        ((pad_y, s - h - pad_y), (pad_x, s - w - pad_x)),
        mode="constant",
        constant_values=0.0,
    )

    img28 = Image.fromarray((square * 255).astype("uint8")).resize((28, 28), Image.BILINEAR)
    return np.array(img28).astype("float32") / 255.0

def normalize(arr28: np.ndarray) -> np.ndarray:
    """Same as transforms.Normalize((0.5,), (0.5,)) => (x-0.5)/0.5."""
    return (arr28 - 0.5) / 0.5

def preprocess(data_url: str) -> torch.Tensor:
    b64 = data_url.split(",", 1)[1]
    raw = base64.b64decode(b64)

    img = Image.open(io.BytesIO(raw)).convert("L")
    arr = np.array(img).astype("float32") / 255.0  # 0..1

    if INVERT:
        arr = 1.0 - arr

    # crop+pad+resize -> 28x28
    arr28 = crop_pad_resize(arr, thresh=THRESH)

    # EMNIST orientation fix (must match training)
    arr28 = emnist_fix_orientation_np(arr28)

    # normalize to -1..1 (must match training)
    arr28_n = normalize(arr28).astype("float32")

    # Debug: save what the model sees before normalize
    Image.fromarray((arr28 * 255).astype("uint8")).save("debug_28x28.png")

    x = torch.from_numpy(arr28_n).unsqueeze(0).unsqueeze(0)  # (1,1,28,28)
    return x

@app.get("/")
def home():
    return render_template("index.html")

@app.post("/predict")
def predict_route():
    payload = request.get_json(force=True)
    x = preprocess(payload["image"])

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
        top_p, top_i = torch.topk(probs, 3)

    out = [{"char": idx_to_char(int(i)), "prob": float(p)} for p, i in zip(top_p.tolist(), top_i.tolist())]
    return jsonify(out)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
