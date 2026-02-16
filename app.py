from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np
import io

app = FastAPI(title="YOLO + EasyOCR API")

# Load YOLO model (make sure best.pt is in the project root)
model = YOLO("best.pt")

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

@app.get("/")
def read_root():
    return {"message": "YOLO + EasyOCR API is running"}

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    # Read uploaded image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    # Run YOLO detection
    results = model(image)

    extracted_data = {}

    # Loop through detected boxes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls)
        field_name = model.names[cls]

        # Crop detected field
        cropped = image.crop((x1, y1, x2, y2))
        cropped_np = np.array(cropped)

        # OCR on cropped field
        ocr_result = reader.readtext(cropped_np)

        text = " ".join([res[1] for res in ocr_result]) if ocr_result else ""
        extracted_data[field_name] = text.strip()

    return extracted_data
