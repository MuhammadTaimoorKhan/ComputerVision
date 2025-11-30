from fastapi import FastAPI, UploadFile, File
import uvicorn
import numpy as np
import cv2
from tensorflow.keras.models import load_model
import os

app = FastAPI()

# ------------------ LOAD MODEL ------------------
model = load_model("model.keras")   # <-- change to your file


# ------------------ PREDICT FUNCTION ------------------
def predict(image_path, model):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)   # grayscale
    img = img / 255.0
    img = cv2.resize(img, (28, 28))
    img = img.reshape(-1, 28, 28, 1)
    
    prediction = model.predict(img)
    pred = int(np.argmax(prediction))
    return pred


# ------------------ API ROUTE ------------------
@app.post("/predict")
async def predict_digit(file: UploadFile = File(...)):

    # Save uploaded file temporarily
    temp_path = "temp_image.png"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    # Predict
    result = predict(temp_path, model)

    # Remove temp file
    os.remove(temp_path)

    return {"prediction": result}


# ------------------ RUN SERVER ------------------
if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
