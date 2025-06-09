from fastapi import FastAPI, File, UploadFile
from PIL import Image
import pytesseract
import cv2
import numpy as np
import io
import re


pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

app = FastAPI()

def preprocess_image(image):
    """ Convert the image to grayscale for better OCR accuracy """
    image_cv = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    return gray  

def clean_text(text):
    """ Clean OCR text by removing extra spaces and unwanted symbols """
    text = re.sub(r'[^a-zA-Z0-9\s\.\-]', '', text)  
    text = re.sub(r'(\d+)[\-,](\d{2})', r'\1.\2', text)  
    text = re.sub(r'\s+', ' ', text).strip()  
    return text

@app.post("/upload")
async def ocr_menu_image(file: UploadFile = File(...)):
    try:
        
        allowed_types = ["image/jpeg", "image/png", "image/jpg"]
        if file.content_type not in allowed_types:
            return {"error": "Invalid file type. Please upload a JPG or PNG image."}

        
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))

        
        processed_image = preprocess_image(image)

        
        raw_text = pytesseract.image_to_string(processed_image, config="--psm 6")

        
        cleaned_text = "\n".join([line.strip() for line in raw_text.split("\n") if line.strip()])

        
        words_prices = []
        for line in cleaned_text.split("\n"):
            words = []
            numbers = []
            
            for part in line.split():  
                part_clean = clean_text(part)  
                if re.match(r"^\d+\.\d{2}$", part_clean):  
                    numbers.append(part_clean)
                else:
                    words.append(part_clean)

            
            if words and numbers:
                words_prices.append([" ".join(words), " ".join(numbers)])

        return {"menu": words_prices}

    except Exception as e:
        return {"error": str(e)}

import uvicorn

if __name__ == "__main__":
    uvicorn.run("ocr_fast:app", host="0.0.0.0", port=8000, reload=True)



