from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image, ImageDraw
import torch
import io
import base64

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"], 
)
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")
@app.post("/generatetext")
async def GenerateTextForInstagram(file: UploadFile = File(...)):
    try:
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        inputs = processor(img, return_tensors="pt");
        out = model.generate(**inputs)
        return JSONResponse(content={"text":processor.decode(out[0], skip_special_tokens=True)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)