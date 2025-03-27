
from fastapi import FastAPI
from pydantic import BaseModel
import numpy as np
import tensorflow as tf

# Load the exported TensorFlow SavedModel
model = tf.keras.models.load_model("mnist_model")

# Initialize FastAPI app
app = FastAPI()

# Define input format
class InputData(BaseModel):
    pixels: list  # expects a 28x28 nested list OR flattened 784-length list

@app.post("/predict")
def predict(data: InputData):
    try:
        image_array = np.array(data.pixels)

        # Reshape if flattened
        if image_array.shape == (784,):
            image_array = image_array.reshape(28, 28)

        # Normalize and reshape for prediction
        image_array = image_array.astype('float32') / 255.0
        image_array = image_array.reshape(1, 28, 28)

        # Make prediction
        prediction = model.predict(image_array)
        predicted_class = int(np.argmax(prediction))
        confidence = float(np.max(prediction))

        return {
            "prediction": predicted_class,
            "confidence": round(confidence, 4)
        }
    except Exception as e:
        return {"error": str(e)}
