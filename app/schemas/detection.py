from typing import List, Literal, Optional
from pydantic import BaseModel, Field


# ------------------------------
# Health Check Response Schema
# ------------------------------
class HealthResponse(BaseModel):
    # Always returns "ok" — used by the /health endpoint
    status: Literal["ok"] = "ok"


# ------------------------------
# Single Detection Schema
# ------------------------------
class Detection(BaseModel):
    # Bounding box coordinates in [x_min, y_min, x_max, y_max] format
    bbox_xyxy: List[float] = Field(..., min_items=4, max_items=4)

    # Confidence score of the detection (0.0 – 1.0)
    score: float

    # Numeric class ID (e.g., 0, 1, 2 depending on the YOLO model classes)
    class_id: int

    # Human-readable class name 
    class_name: str = "hold"


# ------------------------------
# Query Parameters Schema
# ------------------------------
class DetectRequestQuery(BaseModel):
    # Optional confidence threshold for filtering detections
    # Must be between 0.0 and 1.0 if provided
    conf: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Whether to return annotated image with bounding boxes
    return_annotated_image: bool = Field(False, description="Return URL to image with drawn bounding boxes")


# ------------------------------
# Colour Filter Request Schema
# ------------------------------
class colourFilterRequest(BaseModel):
    # Pixel coordinates where user tapped
    tap_x: int = Field(..., ge=0, description="X coordinate of tap point")
    tap_y: int = Field(..., ge=0, description="Y coordinate of tap point")
    
    # Colour similarity tolerance (0-100)
    colour_tolerance: float = Field(30.0, ge=0.0, le=100.0, description="Colour similarity tolerance percentage")
    
    # Optional confidence threshold for filtering detections
    conf: Optional[float] = Field(None, ge=0.0, le=1.0)
    
    # Whether to return annotated image with filtered bounding boxes
    return_annotated_image: bool = Field(False, description="Return URL to image with filtered bounding boxes")


# ------------------------------
# Full Detection Response Schema
# ------------------------------
class DetectResponse(BaseModel):
    # List of all detected holds in the uploaded image
    detections: List[Detection]
    
    # Optional URL to image with bounding boxes drawn
    image_with_boxes: Optional[str] = Field(None, description="URL to image with drawn bounding boxes")


# ------------------------------
# Colour Filter Response Schema
# ------------------------------
class colourFilterResponse(BaseModel):
    # Tap position coordinates
    tap_x: int = Field(..., description="X coordinate where user tapped")
    tap_y: int = Field(..., description="Y coordinate where user tapped")
    
    # The colour that was detected at the tap point
    selected_colour: str = Field(..., description="Colour name detected at tap point")
    
    # Colour confidence (0.0-1.0)
    colour_confidence: float = Field(..., description="Confidence in colour detection")
    
    # Optional URL to image with filtered bounding boxes
    image_with_boxes: Optional[str] = Field(None, description="URL to image with filtered bounding boxes")


# ------------------------------
# Route By Colour Response Schema
# ------------------------------
class RouteByColourResponse(BaseModel):
    # Tap position coordinates used to pick the target colour
    tap_x: int = Field(..., description="X coordinate where user tapped")
    tap_y: int = Field(..., description="Y coordinate where user tapped")

    # The colour detected at the tap point (name and confidence)
    selected_colour: str = Field(..., description="Detected colour name at tap point")
    colour_confidence: float = Field(..., description="Confidence in colour classification (0.0-1.0)")

    # Detections filtered to match the selected colour
    detections: List[Detection]

    # Optional URL to image with matched bounding boxes drawn
    image_with_boxes: Optional[str] = Field(None, description="URL to image with matched bounding boxes only")