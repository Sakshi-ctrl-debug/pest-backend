from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
from tflite_runtime.interpreter import Interpreter
import io
import json
from pest_info import PEST_INFO

app = FastAPI()

# Load model once at startup
print("🔄 Loading TFLite model...")
interpreter = Interpreter(model_path="pest_model.tflite")
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
print("✅ Model loaded successfully")

# Load class names
with open("class_names.json") as f:
    classes = json.load(f)

print(f"✅ Loaded {len(classes)} pest classes")
print("🚀 Backend is ready for inference!")
print("")

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "message": "Backend is running and ready for pest detection",
        "model": "pest_model.tflite",
        "classes": len(classes)
    }

@app.get("/")
async def root():
    return {
        "name": "Pest Detection Backend",
        "version": "1.0",
        "endpoints": {
            "health": "/health",
            "detect": "/detect-pest",
            "docs": "/docs"
        }
    }

def preprocess(image):
    image = image.resize((224, 224), Image.Resampling.LANCZOS)
    img = np.array(image, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img

@app.post("/detect-pest")
async def detect_pest(file: UploadFile = File(...)):
    try:
        print("🔥 REQUEST RECEIVED")

        contents = await file.read()
        print(f"📦 File size: {len(contents)} bytes")

        # Load image
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Preprocess
        processed = preprocess(image)

        # Predict
        interpreter.set_tensor(input_details[0]["index"], processed.astype(np.float32))
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])[0]
        idx = int(np.argmax(pred))

        pest_name = classes[idx]
        confidence = float(pred[idx]) * 100

        print(f"🐛 Prediction: {pest_name} ({confidence})")

        # ✅ UPDATED INFO HANDLING (BILINGUAL)
        raw_info = PEST_INFO.get(pest_name)

        if raw_info:
            info = {
                "marathi_name": raw_info.get("marathi_name", "N/A"),

                "damage_en": raw_info.get("damage", {}).get("en", "No data"),
                "damage_mr": raw_info.get("damage", {}).get("mr", "माहिती उपलब्ध नाही"),

                "prevention_en": raw_info.get("prevention", {}).get("en", "No data"),
                "prevention_mr": raw_info.get("prevention", {}).get("mr", "माहिती उपलब्ध नाही"),

                "treatment_en": raw_info.get("treatment", {}).get("en", "No data"),
                "treatment_mr": raw_info.get("treatment", {}).get("mr", "माहिती उपलब्ध नाही"),
            }
        else:
            info = {
                "marathi_name": "N/A",
                "damage_en": "No data available",
                "damage_mr": "माहिती उपलब्ध नाही",
                "prevention_en": "No data available",
                "prevention_mr": "माहिती उपलब्ध नाही",
                "treatment_en": "No data available",
                "treatment_mr": "माहिती उपलब्ध नाही",
            }

        result = {
            "pest": pest_name,
            "confidence": round(confidence, 2),
            "info": info
        }

        print("✅ Sending response:", result)

        return result

    except Exception as e:
        print("💥 ERROR:", str(e))
        return {"error": str(e)}