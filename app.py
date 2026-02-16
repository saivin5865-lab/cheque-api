from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import io

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")  # replace with your trained model path

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run YOLO prediction
    results = model(image)

    # Extract bounding boxes
    detections = []
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls)
        label = model.names[cls]
        detections.append({
            "label": label,
            "bbox": [x1, y1, x2, y2]
        })

    return {"detections": detections}
