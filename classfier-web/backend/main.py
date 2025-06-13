from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from PIL import Image
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from io import BytesIO
import numpy as np
import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Chinese-Japanese Landscape Painting Classifier API",
    description="API for classifying landscape paintings as either Chinese or Japanese",
    version="1.0.0",
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the model architecture (same as in training)
class_names = ["Chinese", "Japanese"]
model = models.resnet18()
model.fc = nn.Linear(model.fc.in_features, len(class_names))

# Load the trained model
try:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    checkpoint = torch.load("best_model.pth", map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info(f"Model loaded successfully. Best validation accuracy: {checkpoint.get('val_accuracy', 'N/A')}")
except Exception as e:
    logger.error(f"Failed to load model: {str(e)}")
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define image transformation (same as in training)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

@app.get("/")
def read_root() -> Dict[str, str]:
    """
    Root endpoint that returns basic API information.
    """
    return {
        "message": "Chinese-Japanese Landscape Painting Classifier API",
        "status": "active",
        "version": "1.0.0"
    }

@app.get("/health")
def health_check() -> Dict[str, str]:
    """
    Health check endpoint to verify the API is running correctly.
    """
    return {"status": "healthy"}

@app.post("/classify")
async def classify_image(file: UploadFile = File(...)) -> Dict[str, Any]:
    """
    Classify an uploaded image as either Chinese or Japanese landscape painting.
    
    Args:
        file: The image file to classify
        
    Returns:
        A dictionary containing the classification result and confidence score
    """
    # Log the request
    logger.info(f"Received classification request for file: {file.filename}")
    
    start_time = time.time()
    
    # Validate file
    if not file.content_type.startswith("image/"):
        logger.warning(f"Invalid file type: {file.content_type}")
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process the image
        contents = await file.read()
        image = Image.open(BytesIO(contents)).convert("RGB")
        
        # Transform the image
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        # Make prediction
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        # Calculate processing time
        processing_time = time.time() - start_time
        
        # Prepare result
        result = {
            "class": class_names[predicted_class_idx],
            "confidence": confidence,
            "processing_time_ms": round(processing_time * 1000, 2)
        }
        
        logger.info(f"Classification result: {result['class']} with {result['confidence']:.4f} confidence")
        return result
    
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
