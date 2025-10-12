import logging
import io
from typing import Tuple, Optional

from PIL import Image

from app.services.color_service import (
    classify_color,
    save_annotated_image,
    cleanup_old_images
)

# Set up a logger for this module
logger = logging.getLogger(__name__)


# ------------------------------
# Colour Detection Service
# ------------------------------

def get_average_colour(image: Image.Image, x: int, y: int, radius: int = 10) -> Tuple[int, int, int]:
    """
    Compute the average RGB color in a square window around (x, y).
    Returns an (R, G, B) tuple with 0-255 ints.
    """
    # store the pixels being sampled
    pixels = []
    width, height = image.size
    # loop through the pixels in the square region
    # so with radius = 10, we sample a 21x21 square and start at (x-10, y-10)
    for dx in range(-radius, radius + 1):
        # this nested part moves y, thinking like a 2D array
        for dy in range(-radius, radius + 1):
            # this line is what gets the actual pixel (x,y) coords
            nx, ny = x + dx, y + dy
            # this line makes sure those coords are within the image bounds
            if 0 <= nx < width and 0 <= ny < height:
                # adds the pixel at (nx,ny) to the array of pixels
                # this method image.getpixel() returns the RGB color of a pixel as a tuple
                pixels.append(image.getpixel((nx, ny)))
    # if pixels is empty return black
    if not pixels:
        return (0, 0, 0)
    
    """ pixels would look like this:
    pixels = [
    (125, 200, 75),
    (130, 210, 80),
    (120, 195, 70),
    ...] """

    # this sums up all the R,G,B values separately and divides by the number of pixels to get the average
    # for example, for red channel: 
    # r = (125 + 130 + 120 +...) // lenght of pixels = total amount of red divided by the number of pixels
    r = sum(p[0] for p in pixels) // len(pixels)
    # // ensures r is an integer
    g = sum(p[1] for p in pixels) // len(pixels)
    b = sum(p[2] for p in pixels) // len(pixels)
    # return an RGB tuple for average colour in square
    return (r, g, b)


def rgb_to_opencv_hsv(rgb: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """
    Convert RGB (0-255) to OpenCV-style HSV where H in [0,179], S,V in [0,255].
    """
    # colorsys provides rgb_to_hsv but with H in [0,1], S,V in [0,1]
    import colorsys
    # unpack the rgb values
    r, g, b = rgb


    # normalise RGB to [0,1] for colorsys
    """ original | normalised
        R: 255   | 1.0
        G: 128   | 0.50
        B: 50    | 0.20 """
    rh, gs, gb = r / 255.0, g / 255.0, b / 255.0
    # now we have (rh, gs, gb) = (1.0, 0.392, 0.196)


    # convert to HSV using colorsys
    h, s, v = colorsys.rgb_to_hsv(rh, gs, gb)  # h in [0,1], s,v in [0,1]
    # convert to OpenCV Scale
    # gets us h in [0,179], s,v in [0,255]
    # return example: (0.0, 0.392, 0.196)
    return (int(h * 179), int(s * 255), int(v * 255))

def predict_holds_by_colour(
    image_bytes: bytes,
    tap_x: int,
    tap_y: int,
    colour_tolerance: float = 30.0,
    return_annotated_image: bool = False,
) -> Tuple[int, int, str, float, Optional[str]]:
    """
    Detect the colour at a specific tap point in an image.
    
    Args:
        image_bytes: Raw image bytes (from uploaded file)
        tap_x, tap_y: Coordinates where user tapped
        colour_tolerance: Colour similarity tolerance (0-100) - not used in simplified version
        return_annotated_image: Whether to return URL to image with colour indicator
        
    Returns:
        Tuple of (tap_x, tap_y, selected_colour, colour_confidence, annotated_image_url):
        - tap_x, tap_y: Original tap coordinates
        - selected_colour: Name of the detected colour
        - colour_confidence: Confidence in colour detection
        - annotated_image_url: URL to image with colour indicator (or None)
    """
    
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
        
        # Get average colour around tap point and convert to HSV for classification
        avg_rgb = get_average_colour(image, tap_x, tap_y, radius=5)
        target_hsv = rgb_to_opencv_hsv(avg_rgb)
        # classifies the colour to a name and confidence score using color_service.py function
        selected_colour, colour_confidence = classify_color(target_hsv)
        
        logger.info(f"Detected colour at ({tap_x}, {tap_y}): {selected_colour} (confidence: {colour_confidence:.2f})")
        
        # Generate annotated image if requested
        # starts with no URL
        annotated_image_url = None
        if return_annotated_image:
            try:
                # Create a simple annotation showing the tap point and detected colour
                annotated_image = create_colour_indicator_image(image, tap_x, tap_y, selected_colour)
                # Save to disk and get URL
                annotated_image_url = save_annotated_image(annotated_image)
                logger.info("Generated and saved colour indicator image")
                
                # Clean up old images (run occasionally)
                cleanup_old_images()
                    
            except Exception as viz_error:
                logger.warning(f"Failed to generate annotated image: {str(viz_error)}")
                annotated_image_url = None
        
        return tap_x, tap_y, selected_colour, colour_confidence, annotated_image_url
        
    except Exception as e:
        # Log errors and raise a ValueError for the API to handle
        logger.error(f"Error processing image for colour detection: {str(e)}")
        raise ValueError(f"Failed to process image for colour detection: {str(e)}") from e


def create_colour_indicator_image(
    image: Image.Image, 
    tap_x: int, 
    tap_y: int, 
    selected_colour: str
) -> Image.Image:
    """
    Create an image with a visual indicator showing the tap point and detected colour.
    
    Args:
        image: PIL Image object
        tap_x, tap_y: Tap coordinates
        selected_colour: Detected colour name
        
    Returns:
        PIL Image with colour indicator
    """
    from PIL import ImageDraw, ImageFont
    
    # Create a copy to avoid modifying the original
    img_with_indicator = image.copy()
    draw = ImageDraw.Draw(img_with_indicator)
    
    # Try to load a font, fall back to default if not available
    try:
        font = ImageFont.truetype("arial.ttf", 20)
    except (OSError, IOError):
        font = ImageFont.load_default()
    
    # Draw a circle at the tap point
    circle_radius = 10
    draw.ellipse(
        [tap_x - circle_radius, tap_y - circle_radius, 
         tap_x + circle_radius, tap_y + circle_radius],
        outline=(255, 255, 255),  # White outline
        width=3
    )
    
    # Draw a crosshair
    crosshair_size = 15
    draw.line([tap_x - crosshair_size, tap_y, tap_x + crosshair_size, tap_y], 
              fill=(255, 255, 255), width=2)
    draw.line([tap_x, tap_y - crosshair_size, tap_x, tap_y + crosshair_size], 
              fill=(255, 255, 255), width=2)
    
    # Draw colour label
    label = f"Colour: {selected_colour}"
    text_bbox = draw.textbbox((0, 0), label, font=font)
    text_width = text_bbox[2] - text_bbox[0]
    text_height = text_bbox[3] - text_bbox[1]
    
    # Position label near tap point
    label_x = tap_x + 20
    label_y = tap_y - text_height - 10
    
    # Ensure label stays within image bounds
    if label_x + text_width > image.size[0]:
        label_x = tap_x - text_width - 20
    if label_y < 0:
        label_y = tap_y + 20
    
    # Draw background rectangle for text
    draw.rectangle(
        [label_x - 5, label_y - 5, label_x + text_width + 5, label_y + text_height + 5],
        fill=(0, 0, 0, 180)  # Semi-transparent black background
    )
    
    # Draw text
    draw.text((label_x, label_y), label, fill=(255, 255, 255), font=font)
    
    return img_with_indicator
