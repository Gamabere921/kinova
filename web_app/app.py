# web_app/app.py
import os
import io
import uvicorn
from typing import List, Tuple
from fastapi import FastAPI, File, UploadFile, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

import torch
from torchvision import transforms
from PIL import Image

# Ajusta base dir al folder actual (web_app)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Carga config (si usas configs/data.yaml desde repo root)
import yaml
CONFIG_PATH = os.path.join(os.path.dirname(BASE_DIR), "configs", "data.yaml")
with open(CONFIG_PATH, "r") as f:
    config = yaml.safe_load(f)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# IMPORTAR tu clase de modelo
from models.classifier import GarbageClassifier  # según tu repo

# Orden de clases según utils/dataset.py
CLASS_NAMES = ["glass", "paper", "cardboard", "plastic", "metal", "trash"]

# Transforms (deben coincidir con validación)
transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

# Cargar modelo una sola vez al arrancar
MODEL_WEIGHTS = os.environ.get("MODEL_WEIGHTS", "runs/efficientnet_b0_best.pth")
def load_model(weights_path: str):
    model = GarbageClassifier(backbone=config["model_name"], num_classes=config["num_classes"])
    ckpt = torch.load(weights_path, map_location=DEVICE)
    if isinstance(ckpt, dict) and 'model_state_dict' in ckpt:
        model.load_state_dict(ckpt['model_state_dict'])
    else:
        model.load_state_dict(ckpt)
    model.to(DEVICE)
    model.eval()
    return model

MODEL = load_model(MODEL_WEIGHTS)

# FastAPI + templates + static
app = FastAPI(title="Waste Classifier API")

# 👉 Ahora rutas absolutas dentro de web_app
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(BASE_DIR, "static")),
    name="static"
)

# ---- Utils ----
def predict_tensor(img_tensor: torch.Tensor, topk: int = 3) -> Tuple[int, List[float]]:
    with torch.no_grad():
        outputs = MODEL(img_tensor.to(DEVICE))
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
    topk_idx = probs.argsort()[::-1][:topk].tolist()
    topk_probs = [float(probs[i]) for i in topk_idx]
    return topk_idx, topk_probs, probs

# ---- Rutas ----
@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    filename = file.filename
    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        return JSONResponse({"error": "Invalid image"}, status_code=400)

    tensor = transform(image).unsqueeze(0)

    topk_idx, topk_probs, all_probs = predict_tensor(tensor, topk=5)
    result = {
        "top1": {"class": CLASS_NAMES[topk_idx[0]], "prob": topk_probs[0]},
        "topk": [{"class": CLASS_NAMES[i], "prob": topk_probs[j]} for j, i in enumerate(topk_idx)],
        "all_probs": {CLASS_NAMES[i]: float(all_probs[i]) for i in range(len(CLASS_NAMES))}
    }
    return JSONResponse(result)

# Run (para desarrollo)
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
