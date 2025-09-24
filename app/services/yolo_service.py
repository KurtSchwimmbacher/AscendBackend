import logging
import io
from functools import lru_cache
from typing import Any, List, Optional

from huggingface_hub import hf_hub_download
from ultralytics import YOLO
from PIL import Image

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
# Run Detection on Image Bytes
# ------------------------------
def predict_holds(
    image_bytes: bytes,
    conf: Optional[float] = None,
) -> List[dict]:
    """
    Predict climbing holds in an image.
    
    Args:
        image_bytes: Raw image bytes (from uploaded file)
        conf: Optional confidence threshold (0.0–1.0)
        
    Returns:
        List of detections, each as a dictionary with:
        - bbox_xyxy: [x_min, y_min, x_max, y_max]
        - score: confidence
        - class_id: numeric class
        - class_name: human-readable class
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
        return parsed
        
    except Exception as e:
        # Log errors and raise a ValueError for the API to handle
        logger.error(f"Error processing image: {str(e)}")
        raise ValueError(f"Failed to process image: {str(e)}") from e
