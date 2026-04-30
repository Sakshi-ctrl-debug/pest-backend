from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from PIL import Image
import numpy as np
import tensorflow as tf
import io
import json
from pest_info import PEST_INFO

app = FastAPI()

# Load model once at startup
print("🔄 Loading TensorFlow model...")
model = tf.keras.models.load_model("pest_model.h5")
print("✅ Model loaded successfully")

# Load class names
with open("class_names.json") as f:
    classes = json.load(f)

print(f"✅ Loaded {len(classes)} pest classes")
print("🚀 Backend is ready for inference!")
print("")

# Marathi names for pests
pest_marathi = {
    "aphids": "मावा",
    "army_worm": "लष्करी अळी",
    "beetle": "भुंगा",
    "bollworm": "बोंडअळी",
    "grasshopper": "टोळ",
    "mites": "कोळी कीटक",
    "red_spider": "लाल कोळी कीटक",
    "sawfly": "करवत माशी/करवत अळी ",
    "stem_borer": "खोड किडा",
    "whitefly": "पांढरी माशी"
}

def preprocess(image):
    # Optimized preprocessing
    image = image.resize((224, 224), Image.Resampling.LANCZOS)  # Better quality resize
    img = np.array(image, dtype=np.float32) / 255.0  # Use float32 for better precision
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
        pred = model.predict(processed)[0]
        idx = int(np.argmax(pred))

        pest_name = classes[idx]
        confidence = float(pred[idx])

        print(f"🐛 Prediction: {pest_name} ({confidence})")

        # Get pest info
        info = PEST_INFO.get(pest_name, {
            "damage": "No data available",
            "prevention": "No data available",
            "treatment": "No data available"
        })

        # Get Marathi name
        marathi_name = pest_marathi.get(pest_name, pest_name)

        result = {
            "pest": pest_name,
            "marathi_name": marathi_name,
            "confidence": confidence,
            "info": info
        }

        print("✅ Sending response:", result)

        return result

    except Exception as e:
        print("💥 ERROR:", str(e))
        return {"error": str(e)}
