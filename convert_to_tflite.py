import tensorflow as tf
import numpy as np

def convert_to_tflite():
    """Convert Keras model to TensorFlow Lite for faster inference"""

    print("🔄 Loading Keras model...")
    model = tf.keras.models.load_model("pest_model.h5")
    print("✅ Model loaded")

    print("🔄 Converting to TFLite...")
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Optimization settings
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]  # Use float16 for smaller size

    tflite_model = converter.convert()
    print("✅ Conversion complete")

    # Save the model
    with open('pest_model.tflite', 'wb') as f:
        f.write(tflite_model)

    print(f"📁 TFLite model saved: {len(tflite_model)} bytes")
    print("🚀 Ready for on-device inference!")

if __name__ == "__main__":
    convert_to_tflite()