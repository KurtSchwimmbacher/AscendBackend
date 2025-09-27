import logging
import io
from typing import List, Optional, Tuple

from PIL import Image

from app.core.config import settings
from app.services.yolo_service import get_model
from app.services.color_service import (
    get_color_at_pixel,
    classify_color,
    filter_detections_by_color,
    draw_bounding_boxes,
    save_annotated_image,
    cleanup_old_images
)

# Set up a logger for this module
logger = logging.getLogger(__name__)


# ------------------------------
# Color Filter Detection
# ------------------------------
def predict_holds_by_color(
    image_bytes: bytes,
    tap_x: int,
    tap_y: int,
    color_tolerance: float = 30.0,
    conf: Optional[float] = None,
    return_annotated_image: bool = False,
) -> Tuple[List[dict], str, float, Optional[str]]:
    """
    Predict climbing holds and filter by color at tap point.
    
    Args:
        image_bytes: Raw image bytes (from uploaded file)
        tap_x, tap_y: Coordinates where user tapped
        color_tolerance: Color similarity tolerance (0-100)
        conf: Optional confidence threshold (0.0–1.0)
        return_annotated_image: Whether to return URL to image with boxes
        
    Returns:
        Tuple of (filtered_detections, selected_color, color_confidence, annotated_image_url):
        - filtered_detections: List of detection dictionaries matching the color
        - selected_color: Name of the detected color
        - color_confidence: Confidence in color detection
        - annotated_image_url: URL to image with filtered boxes (or None)
    """
    # Get the cached model instance
    model = get_model()

    # Use user-provided confidence or default from settings
    confidence = conf if conf is not None else settings.yolo_conf
    
    try:
        # Convert raw bytes to a PIL Image
        image = Image.open(io.BytesIO(image_bytes))

        # Convert image to RGB if necessary
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        logger.info(f"Processing image: {image.size[0]}x{image.size[1]} pixels")
        
        # Validate tap coordinates
        if tap_x >= image.size[0] or tap_y >= image.size[1] or tap_x < 0 or tap_y < 0:
            raise ValueError(f"Tap coordinates ({tap_x}, {tap_y}) are outside image bounds")
        
        # Run YOLO inference to get all detections
        results = model(
            source=image,
            imgsz=settings.yolo_imgsz,
            max_det=settings.yolo_max_det,
            conf=confidence,
            verbose=False,
        )

        # Parse results into a list of dictionaries
        all_detections: List[dict] = []
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    xyxy = box.xyxy[0].tolist()
                    score = float(box.conf[0])
                    cls_id = int(box.cls[0]) if box.cls is not None else 0
                    all_detections.append(
                        {
                            "bbox_xyxy": xyxy,
                            "score": score,
                            "class_id": cls_id,
                            "class_name": "hold",
                        }
                    )
        
        logger.info(f"Found {len(all_detections)} total climbing holds")
        
        # Get color at tap point
        target_hsv = get_color_at_pixel(image, tap_x, tap_y)
        selected_color, color_confidence = classify_color(target_hsv)
        
        logger.info(f"Detected color at ({tap_x}, {tap_y}): {selected_color} (confidence: {color_confidence:.2f})")
        
        # Filter detections by color
        filtered_detections = filter_detections_by_color(
            all_detections, image, target_hsv, color_tolerance
        )
        
        logger.info(f"Filtered to {len(filtered_detections)} holds matching color '{selected_color}'")
        
        # Generate annotated image if requested
        annotated_image_url = None
        if return_annotated_image and filtered_detections:
            try:
                # Draw bounding boxes on the original image (only filtered detections)
                annotated_image = draw_bounding_boxes(image, filtered_detections, conf_threshold=confidence)
                # Save to disk and get URL
                annotated_image_url = save_annotated_image(annotated_image)
                logger.info("Generated and saved filtered annotated image")
                
                # Clean up old images (run occasionally)
                if len(filtered_detections) % 10 == 0:
                    cleanup_old_images()
                    
            except Exception as viz_error:
                logger.warning(f"Failed to generate annotated image: {str(viz_error)}")
                annotated_image_url = None
        
        return filtered_detections, selected_color, color_confidence, annotated_image_url
        
    except Exception as e:
        # Log errors and raise a ValueError for the API to handle
        logger.error(f"Error processing image for color filtering: {str(e)}")
        raise ValueError(f"Failed to process image for color filtering: {str(e)}") from e
