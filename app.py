from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
from PIL import Image
import easyocr
import numpy as np
import io

app = FastAPI()

# Load YOLO model
model = YOLO("best.pt")

# Load EasyOCR
reader = easyocr.Reader(['en'], gpu=False)

@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")

    results = model(image)

    extracted_data = {}

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        cls = int(box.cls)
        field_name = model.names[cls]

        # Crop detected field
        cropped = image.crop((x1, y1, x2, y2))

        # Convert to numpy for EasyOCR
        cropped_np = np.array(cropped)

        # Run OCR
        ocr_result = reader.readtext(cropped_np)

        text = ""
        if len(ocr_result) > 0:
            text = " ".join([res[1] for res in ocr_result])

        extracted_data[field_name] = text.strip()

    return extracted_data
