import logging
import io
import os
import uuid
import time
from typing import List, Optional, Tuple, Dict

from PIL import Image, ImageDraw, ImageFont
import cv2
import numpy as np

# Set up a logger for this module
logger = logging.getLogger(__name__)


# ------------------------------
# Colour Analysis Constants
# ------------------------------
# HSV colour ranges for common climbing hold colours
COLOUR_RANGES = {
    "red": [(0, 50, 50), (10, 255, 255), (170, 50, 50), (180, 255, 255)],  # Red has two ranges
    "orange": [(10, 50, 50), (25, 255, 255)],
    "yellow": [(25, 50, 50), (35, 255, 255)],
    "green": [(35, 50, 50), (85, 255, 255)],
    "blue": [(85, 50, 50), (130, 255, 255)],
    "purple": [(130, 50, 50), (170, 255, 255)],
    "pink": [(160, 50, 50), (180, 255, 255)],
    "white": [(0, 0, 200), (180, 30, 255)],
    "black": [(0, 0, 0), (180, 255, 50)],
    "gray": [(0, 0, 50), (180, 30, 200)],
}


# ------------------------------
# Colour Analysis Functions
# ------------------------------
def get_colour_at_pixel(image: Image.Image, x: int, y: int, region_size: int = 10) -> Tuple[int, int, int]:
    """
    Get the dominant colour in a small region around the specified pixel.
    
    Args:
        image: PIL Image object
        x, y: Center coordinates
        region_size: Size of the region to analyze (odd number)
        
    Returns:
        HSV colour tuple (h, s, v)
    """
    # Convert PIL to OpenCV format
    # np.array converts the image to a numpy array (which is in RGB because PIL is RGB)
    # cv2.cvtColor converts the image from RGB to BGR - because OpenCV expects BGR for processing
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    # cv2.cvtColor converts the image from BGR to HSV -> HSV is a colour space that is more intuitive for colour detection
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    # Define the square region around the pixel to analyze
    half_size = region_size // 2
    # y1 is the top left y coordinate of the square region and so on
    # max & min ensure that the region stays within the image boundaries
    y1 = max(0, y - half_size)
    y2 = min(hsv_image.shape[0], y + half_size + 1)
    x1 = max(0, x - half_size)
    x2 = min(hsv_image.shape[1], x + half_size + 1)
    
    # Extract region from the HSV image with the coordinate ranges defined above
    region = hsv_image[y1:y2, x1:x2]
    
    # Calculate mean colour
    # np.mean calculates the mean colour of the region
    # so mean_colour will look like this: mean_colour = [12.3, 200.5, 255.0] -> hue, saturation, value
    mean_colour = np.mean(region, axis=(0, 1))
    return tuple(map(int, mean_colour))


def classify_colour(hsv_colour: Tuple[int, int, int]) -> Tuple[str, float]:
    """
    Classify an HSV colour into a named colour category.
    
    Args:
        hsv_colour: HSV colour tuple (h, s, v)
        
    Returns:
        Tuple of (colour_name, confidence)
    """
    h, s, v = hsv_colour
    
    best_match = "unknown"
    best_confidence = 0.0
    
    for colour_name, ranges in COLOUR_RANGES.items():
        confidence = 0.0
        
        # Handle colours with multiple ranges (like red)
        if isinstance(ranges[0], list):
            for range_pair in ranges:
                if len(range_pair) == 2:
                    h_min, s_min, v_min = range_pair[0]
                    h_max, s_max, v_max = range_pair[1]
                    
                    # if the hue, saturation, and value are within the range, set the confidence to 1.0
                    if (h_min <= h <= h_max and 
                        s_min <= s <= s_max and 
                        v_min <= v <= v_max):
                        confidence = 1.0
                        break
        else:
            # Single range
            h_min, s_min, v_min = ranges[0]
            h_max, s_max, v_max = ranges[1]
            
            if (h_min <= h <= h_max and 
                s_min <= s <= s_max and 
                v_min <= v <= v_max):
                confidence = 1.0
        
        if confidence > best_confidence:
            best_confidence = confidence
            best_match = colour_name
    
    return best_match, best_confidence


def create_colour_mask(image: Image.Image, target_hsv: Tuple[int, int, int], tolerance: float) -> np.ndarray:
    """
    Create a binary mask of pixels similar to the target colour.
    
    Args:
        image: PIL Image object
        target_hsv: Target HSV colour
        tolerance: Colour tolerance (0-100)
        
    Returns:
        Binary mask as numpy array
    """
    # Convert PIL to OpenCV format
    # np.array converts the image to a numpy array (which is in RGB because PIL is RGB)
    # cv2.cvtColor converts the image from RGB to BGR - because OpenCV expects BGR for processing
    cv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    hsv_image = cv2.cvtColor(cv_image, cv2.COLOR_BGR2HSV)
    
    h, s, v = target_hsv
    # tolerance is user defined from 0-100, so now we convert it to 0-179 for H and 0-255 for S and V
    # this is for how similar the colour needs to be to the target colour
    tolerance_h = int(tolerance * 1.8)  # Hue tolerance
    tolerance_sv = int(tolerance * 2.55)  # Saturation/Value tolerance
    
    # Define colour range
    # max & min ensure that the range stays within the colour boundaries
    # one calculation for each of H, S, and V
    lower = np.array([
        max(0, h - tolerance_h), #H
        max(0, s - tolerance_sv), #S
        max(0, v - tolerance_sv) #V
        ]) #lower is now the lowest possible colour value -> lower = (0, 0, 0)
    upper = np.array([
        min(179, h + tolerance_h), 
        min(255, s + tolerance_sv), 
        min(255, v + tolerance_sv)
        ])
    
    # Create mask
    # cv2.inRange creates a binary mask of the image where the pixels are within the lower and upper bounds
    # pixels within the bounds are set to 255, otherwise 0
    mask = cv2.inRange(hsv_image, lower, upper)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8) # creates a 3x3 kernel
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel) # closes gaps in the mask
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel) # removes small white noise from the mask
    
    return mask


def filter_detections_by_colour(
    detections: List[dict], 
    image: Image.Image, 
    target_hsv: Tuple[int, int, int], 
    tolerance: float
) -> List[dict]:
    """
    Filter detections to only include holds with colours similar to the target.
    
    Args:
        detections: List of detection dictionaries
        image: PIL Image object
        target_hsv: Target HSV colour
        tolerance: Colour tolerance (0-100)
        
    Returns:
        Filtered list of detections
    """
    # Create colour mask using function above
    colour_mask = create_colour_mask(image, target_hsv, tolerance)
    
    # this will store detections that are within the colour mask
    filtered_detections = []
    
    # iterate through each detection
    for detection in detections:
        bbox = detection["bbox_xyxy"] #get the bounding box coordinates
        x1, y1, x2, y2 = [int(coord) for coord in bbox] #convert to integers
        
        # Ensure coordinates are within image bounds
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(colour_mask.shape[1], x2)
        y2 = min(colour_mask.shape[0], y2)
        
        # this is the region of the image that is within the colour mask
        region_mask = colour_mask[y1:y2, x1:x2] #array of booleans for each pixel in the region
        
        # Calculate colour overlap percentage
        if region_mask.size > 0: #total number of pixels in the region
            colour_pixels = np.sum(region_mask > 0) #counts the number of pixels that match the target colour
            total_pixels = region_mask.size 
            overlap_percentage = (colour_pixels / total_pixels) * 100 #percentage of the region that matches the target colour
            
            # Include detection if it has sufficient colour overlap
            if overlap_percentage >= 20.0:  # At least 20% of the region should match the colour
                filtered_detections.append(detection) #add the detection to the filtered list
    
    return filtered_detections


# ------------------------------
# Image Visualization Functions
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
        
        # Choose colour based on confidence
        if score >= 0.8:
            colour = (0, 255, 0)  # Green for high confidence
        elif score >= 0.6:
            colour = (255, 255, 0)  # Yellow for medium confidence
        else:
            colour = (255, 0, 0)  # Red for low confidence
        
        # Draw bounding box
        draw.rectangle([x1, y1, x2, y2], outline=colour, width=2)
        
        # Draw confidence score
        label = f"{score:.2f}"
        text_bbox = draw.textbbox((0, 0), label, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_height = text_bbox[3] - text_bbox[1]
        
        # Draw background rectangle for text
        draw.rectangle(
            [x1, y1 - text_height - 4, x1 + text_width + 4, y1],
            fill=colour
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
