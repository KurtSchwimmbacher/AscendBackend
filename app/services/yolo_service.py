import logging
import io
import base64
import os
import uuid
import time
from functools import lru_cache
from typing import Any, List, Optional, Tuple

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

from app.core.config import settings


# Set up a logger for this module
logger = logging.getLogger(__name__)


# ------------------------------
# Load YOLO Model (cached)
# ------------------------------
@lru_cache(maxsize=1)  # Only load the model once, subsequent calls return cached instance
def get_model() -> YOLO:
    """
    Download and load the YOLOv8 model from Hugging Face.
    Uses LRU cache to avoid reloading the model multiple times.
    """
    logger.info(
        "Loading YOLOv8 model from Hugging Face repo %s (file: %s)",
        settings.hf_repo_id,
        settings.hf_filename,
    )

    # Download model weights from Hugging Face hub
    weights_path = hf_hub_download(
        repo_id=settings.hf_repo_id,
        filename=settings.hf_filename,
        local_dir=None,  # Default cache dir (~/.cache/huggingface/hub)
    )

    # Load the YOLO model with the downloaded weights
    model = YOLO(weights_path)
    logger.info("YOLOv8 model loaded")
    return model


# ------------------------------
# Helper Functions for Image Visualization
# ------------------------------
def draw_bounding_boxes(
    image: Image.Image, 
    detections: List[dict],
    conf_threshold: float = 0.5
) -> Image.Image:
    """
    Draw bounding boxes and confidence scores on the image.
    
    Args:
        image: PIL Image object
        detections: List of detection dictionaries
        conf_threshold: Minimum confidence to display
        
    Returns:
        PIL Image with drawn bounding boxes
    """
    # Create a copy to avoid modifying the original
    img_with_boxes = image.copy()
    draw = ImageDraw.Draw(img_with_boxes)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    for detection in detections:
        if detection["score"] < conf_threshold:
            continue
            
        bbox = detection["bbox_xyxy"]
        score = detection["score"]
        
        # Convert to integers for drawing
        x1, y1, x2, y2 = [int(coord) for coord in bbox]
        
        # Choose color based on confidence
        if score >= 0.8:
            color = (0, 255, 0)  # Green for high confidence
        elif score >= 0.6:
            color = (255, 255, 0)  # Yellow for medium confidence
        else:
            color = (255, 0, 0)  # Red for low confidence
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        
        # Draw confidence score
        label = f"{score:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background rectangle for text
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=color
        )
        
        # Draw text
        draw.text((x1 + 2, y1 - text_height - 2), label, fill=(0, 0, 0), font=font)
    
    return img_with_boxes


def save_annotated_image(image: Image.Image, format: str = "JPEG") -> str:
    """
    Save annotated image to disk and return the URL path.
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        URL path to the saved image
    """
    # Create static/images directory if it doesn't exist
    static_dir = "static"
    images_dir = os.path.join(static_dir, "images")
    os.makedirs(images_dir, exist_ok=True)
    
    # Generate unique filename with timestamp
    timestamp = int(time.time())
    unique_id = str(uuid.uuid4())[:8]
    filename = f"detection_{timestamp}_{unique_id}.{format.lower()}"
    filepath = os.path.join(images_dir, filename)
    
    # Save the image
    image.save(filepath, format=format, quality=85, optimize=True)
    
    # Return the URL path (relative to the API base)
    url_path = f"/static/images/{filename}"
    logger.info(f"Saved annotated image to: {url_path}")
    
    return url_path


def cleanup_old_images(max_age_hours: int = 24):
    """
    Clean up old annotated images to prevent disk bloat.
    
    Args:
        max_age_hours: Maximum age of images in hours before deletion
    """
    try:
        images_dir = os.path.join("static", "images")
        if not os.path.exists(images_dir):
            return
            
        current_time = time.time()
        max_age_seconds = max_age_hours * 3600
        
        for filename in os.listdir(images_dir):
            if filename.startswith("detection_") and filename.endswith((".jpg", ".jpeg", ".png")):
                filepath = os.path.join(images_dir, filename)
                file_age = current_time - os.path.getmtime(filepath)
                
                if file_age > max_age_seconds:
                    os.remove(filepath)
                    logger.info(f"Cleaned up old image: {filename}")
                    
    except Exception as e:
        logger.warning(f"Failed to cleanup old images: {e}")


def image_to_base64(image: Image.Image, format: str = "JPEG") -> str:
    """
    Convert PIL Image to base64 string (kept for backward compatibility).
    
    Args:
        image: PIL Image object
        format: Image format (JPEG, PNG, etc.)
        
    Returns:
        Base64 encoded string with data URI prefix
    """
    buffer = io.BytesIO()
    image.save(buffer, format=format)
    img_bytes = buffer.getvalue()
    img_base64 = base64.b64encode(img_bytes).decode()
    
    # Return as data URI
    mime_type = f"image/{format.lower()}"
    return f"data:{mime_type};base64,{img_base64}"


# ------------------------------
# Run Detection on Image Bytes
# ------------------------------
def predict_holds(
    image_bytes: bytes,
    conf: Optional[float] = None,
    return_annotated_image: bool = False,
) -> Tuple[List[dict], Optional[str]]:
    """
    Predict climbing holds in an image.
    
    Args:
        image_bytes: Raw image bytes (from uploaded file)
        conf: Optional confidence threshold (0.0–1.0)
        return_annotated_image: Whether to return base64-encoded image with boxes
        
    Returns:
        Tuple of (detections, annotated_image_base64):
        - detections: List of detection dictionaries
        - annotated_image_base64: Base64 string of image with boxes (or None)
    """
    # Get the cached model instance
    model = get_model()

    # Use user-provided confidence or default from settings
    confidence = conf if conf is not None else settings.yolo_conf
    
    try:
        # Convert raw bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to RGB if necessary (handles RGBA, grayscale, etc.)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {image.size[0]}x{image.size[1]} pixels")
        
        # Run YOLO inference
        results = model(
            source=image,
            imgsz=settings.yolo_imgsz,     # Inference image size
            max_det=settings.yolo_max_det,  # Maximum number of detections
            conf=confidence,                # Confidence threshold
            verbose=False,
        )

        # Parse results into a list of dictionaries
        parsed: List[dict] = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()           # Bounding box coords
                    score = float(box.conf[0])            # Confidence score
                    cls_id = int(box.cls[0]) if box.cls is not None else 0
                    parsed.append(
                        {
                            "bbox_xyxy": xyxy,
                            "score": score,
                            "class_id": cls_id,
                            "class_name": "hold",  # Only detecting holds
                        }
                    )
        
        logger.info(f"Found {len(parsed)} climbing holds")
        
        # Generate annotated image if requested
        annotated_image_url = None
        if return_annotated_image and parsed:
            try:
                # Draw bounding boxes on the original image
                annotated_image = draw_bounding_boxes(image, parsed, conf_threshold=confidence)
                # Save to disk and get URL
                annotated_image_url = save_annotated_image(annotated_image)
                logger.info("Generated and saved annotated image with bounding boxes")
                
                # Clean up old images (run occasionally, not every request)
                if len(parsed) % 10 == 0:  # Cleanup every 10th request
                    cleanup_old_images()
                    
            except Exception as viz_error:
                logger.warning(f"Failed to generate annotated image: {str(viz_error)}")
                # Don't fail the whole request if visualization fails
                annotated_image_url = None
        
        return parsed, annotated_image_url
        
    except Exception as e:
        # Log errors and raise a ValueError for the API to handle
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}") from e
