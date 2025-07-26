# 1_backend/main.py
# -------------------------------------------------
from fastapi import FastAPI, Request, UploadFile, File
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ultralytics import YOLO
import uuid, shutil, os

# ───────────────────────────────────────────────
# Paths
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__ + "/.."))  # snake-spotter
TEMPLATES_DIR = os.path.join(PROJECT_ROOT, "templates")
STATIC_DIR    = os.path.join(PROJECT_ROOT, "static")
UPLOAD_DIR    = os.path.join(STATIC_DIR, "uploads")
MODEL_PATH    = os.path.join(PROJECT_ROOT,
                             "runs", "detect", "train2", "weights", "best.pt")
# ───────────────────────────────────────────────
# Ensure folders exist
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ───────────────────────────────────────────────
# FastAPI & Jinja2
app = FastAPI(title="Snake-Spotter API")
templates = Jinja2Templates(directory=TEMPLATES_DIR)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ───────────────────────────────────────────────
# Load YOLOv8 model once at startup
model = YOLO(MODEL_PATH)
print("💡 Model loaded from:", MODEL_PATH)

# ───────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    """Render upload form."""
    return templates.TemplateResponse("index.html",
                                      {"request": request})

# ───────────────────────────────────────────────
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request,
                  file: UploadFile = File(..., description="Snake image")):
    """Save upload → run YOLOv8 → show result."""
    # ① Save uploaded file
    ext = os.path.splitext(file.filename)[1].lower()
    unique_name = f"{uuid.uuid4()}{ext}"
    save_path = os.path.join(UPLOAD_DIR, unique_name)

    with open(save_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # ② Run inference
    results = model(save_path, conf=0.25)[0]   # single-image list → take first
    if len(results.boxes) == 0:
        label_text = "No snake detected"
    else:
        cls_id   = int(results.boxes.cls[0])
        conf     = float(results.boxes.conf[0])
        species  = model.names[cls_id]          # 'snake' or your class names
        label_text = f"Detected: {species}  (confidence {conf:.2f})"

        # (Optional) save annotated image over the upload
        results.save(filename=save_path)

    # ③ Return HTML with result
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "result":  label_text,
            "image_url": f"/static/uploads/{unique_name}"
        }
    )