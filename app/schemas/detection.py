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


# ------------------------------
# Full Detection Response Schema
# ------------------------------
class DetectResponse(BaseModel):
    # List of all detected holds in the uploaded image
    detections: List[Detection]
