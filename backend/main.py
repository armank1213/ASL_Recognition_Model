from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import torch
from PIL import Image
import io
import logging
from model import ASLNet, process_image, predict_letter

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "http://localhost:5173",
        "http://127.0.0.1:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

try:
    logger.info("Loading model...")
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = ASLNet().to(device)
    model.load_state_dict(torch.load('asl_model.pth'))
    model.eval()
    logger.info("Model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    try:
        logger.info(f"Received file: {file.filename}, content_type: {file.content_type}")
        
        contents = await file.read()
        if not contents:
            logger.error("Received empty file")
            raise HTTPException(status_code=400, detail="Empty file received")
        
        logger.info("Processing image...")
        image_tensor = process_image(contents)
        if image_tensor is None:
            logger.error("Failed to process image")
            raise HTTPException(status_code=400, detail="Failed to process image")
            
        logger.info(f"Image tensor shape: {image_tensor.shape}")
        logger.info("Making prediction...")
        
        prediction = predict_letter(model, image_tensor, device)
        logger.info(f"Prediction result: {prediction}")
        
        return {"prediction": prediction}
    except Exception as e:
        logger.error(f"Error in predict endpoint: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)