from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from transformers import AutoProcessor, BlipForConditionalGeneration
from PIL import Image
import io

app = FastAPI()

processor = AutoProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")
@app.post("/classificar-imagem/")
async def classificar_imagem(file: UploadFile = File(...)):
    try:
        # Leia a imagem
        image_data = await file.read()
        img = Image.open(io.BytesIO(image_data)).convert("RGB")
        inputs = processor(img, return_tensors="pt").to("cuda")
        
        out = model.generate(**inputs)
        
        results = processor.decode(out[0], skip_special_tokens=True)
        return JSONResponse(content=results)
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)
