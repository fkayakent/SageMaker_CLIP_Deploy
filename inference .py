import os
import json
import logging
import torch
from PIL import Image
import io
import base64
import numpy as np
from transformers import CLIPProcessor, CLIPModel

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

def model_fn(model_dir):
    """Load the CLIP model for inference."""
    logger.info("Loading model...")
    
    # Load model and processor from the saved directory
    model = CLIPModel.from_pretrained(model_dir)
    processor = CLIPProcessor.from_pretrained(model_dir)
    
    if torch.cuda.is_available():
        model = model.to("cuda")
        
    return {
        "model": model,
        "processor": processor
    }

def input_fn(request_body, request_content_type):
    """Parse input data for inference."""
    logger.info(f"Received request with content type: {request_content_type}")
    
    if request_content_type == "application/json":
        input_data = json.loads(request_body)
        
        # Decode base64 image
        image_bytes = base64.b64decode(input_data["inputs"])
        image = Image.open(io.BytesIO(image_bytes))
        
        # Get candidate labels from parameters
        candidate_labels = input_data["parameters"]["candidate_labels"]
        
        return {
            "image": image,
            "candidate_labels": candidate_labels
        }
    
    raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_dict):
    """Make prediction using the loaded model."""
    model = model_dict["model"]
    processor = model_dict["processor"]
    
    try:
        # Process image and text inputs
        inputs = processor(
            images=input_data["image"],
            text=input_data["candidate_labels"],
            return_tensors="pt",
            padding=True
        )
        
        if torch.cuda.is_available():
            inputs = {key: val.to("cuda") for key, val in inputs.items()}
        
        # Get the probability for each label
        outputs = model(**inputs)
        logits_per_image = outputs.logits_per_image
        probs = logits_per_image.softmax(dim=1)
        
        # Convert predictions to list and pair with labels
        probs = probs.cpu().detach().numpy()
        predictions = []
        for label, score in zip(input_data["candidate_labels"], probs[0].tolist()):
            predictions.append({
                "label": label,
                "score": score
            })
        
        # Sort predictions by score in descending order
        predictions.sort(key=lambda x: x["score"], reverse=True)
        
        return {
            "predictions": predictions
        }
    
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise

def output_fn(prediction, response_content_type):
    """Format prediction for response."""
    if response_content_type == "application/json":
        return json.dumps(prediction)
    
    raise ValueError(f"Unsupported content type: {response_content_type}")