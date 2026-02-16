from fastapi import FastAPI, File, UploadFile
from ultralytics import YOLO
import shutil
import uuid
import os

app = FastAPI(title="Cheque Field Detection API")

# Load model once at startup
model = YOLO("best.pt")

@app.get("/")
def home():
    return {"message": "Cheque Detection API Running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):

    temp_filename = f"{uuid.uuid4()}.jpg"

    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    results = model.predict(temp_filename, conf=0.25)

    detections = []

    for box in results[0].boxes:
        cls_id = int(box.cls)
        conf = float(box.conf)
        class_name = model.names[cls_id]
        coords = box.xyxy.tolist()[0]

        detections.append({
            "field": class_name,
            "confidence": round(conf, 3),
            "bbox": coords
        })

    os.remove(temp_filename)

    return {
        "status": "success",
        "detections": detections
    }
