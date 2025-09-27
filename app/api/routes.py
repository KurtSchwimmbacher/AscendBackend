import logging
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException

# Import Pydantic request/response schemas for type validation and docs
from app.schemas.detection import (
    DetectRequestQuery,
    DetectResponse,
    colourFilterRequest,
    colourFilterResponse,
    HealthResponse,
)

# Import your YOLO prediction service
from app.services.yolo_service import predict_holds
from app.services.colour_detection_service import predict_holds_by_colour

# Set up a logger for this module 
logger = logging.getLogger(__name__)


# ------------------------------
# Router setup
# ------------------------------
# Create an APIRouter instance — this groups related endpoints together.
# Later, this `api_router` will be included in the main FastAPI app.
api_router = APIRouter()


# ------------------------------
# Health Check Endpoint
# ------------------------------
@api_router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    """
    Simple health check endpoint.
    Useful for monitoring and deployment systems
    to verify that the API is alive and responsive.
    """
    return HealthResponse()


# ------------------------------
# Detection Endpoint
# ------------------------------
@api_router.post("/detect", response_model=DetectResponse)
async def detect(
    file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)"),
    query: DetectRequestQuery = Depends(),  # Extracts query parameters like confidence threshold
) -> DetectResponse:
    """
    Detect climbing holds in an uploaded image using YOLO.
    
    Args:
        file: Image file uploaded by the client (JPG/PNG)
        query: Optional query parameters (e.g., confidence threshold)
    
    Returns:
        A DetectResponse object containing bounding boxes and confidence scores
    """

    # 1. Validate file type (only images are allowed)
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    # 2. Validate file size (limit to 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file.size} bytes. Maximum size is {max_size} bytes."
        )
    
    try:
        # Log metadata about the uploaded file
        logger.info(
            f"Processing uploaded file: {file.filename}, "
            f"type: {file.content_type}, size: {file.size}"
        )
        
        # 3. Read the raw bytes of the uploaded image
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # 4. Run YOLO inference using your service layer
        detections, annotated_image_url = predict_holds(
            contents, 
            conf=query.conf,
            return_annotated_image=query.return_annotated_image
        )

        # 5. Return results wrapped in a Pydantic response model
        return DetectResponse(
            detections=detections,
            image_with_boxes=annotated_image_url
        )
        
    except HTTPException:
        # If you already raised an HTTPException above, just re-raise it
        raise
    except ValueError as e:
        # Catch predictable errors (e.g. invalid image format)
        logger.error(f"Image processing error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as exc:
        # Catch *unexpected* errors so the API doesn’t crash silently
        logger.error(f"Unexpected error during detection: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during image processing"
        ) from exc


# ------------------------------
# colour Filter Endpoint
# ------------------------------
@api_router.post("/detect/colour-filter", response_model=colourFilterResponse)
async def detect_by_colour(
    file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)"),
    query: colourFilterRequest = Depends(),  # Extracts query parameters
) -> colourFilterResponse:
    """
    Detect climbing holds and filter by colour at a specific tap point.
    
    Args:
        file: Image file uploaded by the client (JPG/PNG)
        query: Query parameters including tap coordinates and colour tolerance
    
    Returns:
        A colourFilterResponse object containing filtered detections and colour info
    """

    # 1. Validate file type (only images are allowed)
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400, 
            detail=f"Invalid file type: {file.content_type}. Please upload an image file."
        )
    
    # 2. Validate file size (limit to 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file.size} bytes. Maximum size is {max_size} bytes."
        )
    
    try:
        # Log metadata about the uploaded file
        logger.info(
            f"Processing colour filter request: {file.filename}, "
            f"type: {file.content_type}, size: {file.size}, "
            f"tap: ({query.tap_x}, {query.tap_y}), tolerance: {query.colour_tolerance}"
        )
        
        # 3. Read the raw bytes of the uploaded image
        contents = await file.read()
        
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")
        
        # 4. Run colour detection
        tap_x, tap_y, selected_colour, colour_confidence, annotated_image_url = predict_holds_by_colour(
            contents,
            tap_x=query.tap_x,
            tap_y=query.tap_y,
            colour_tolerance=query.colour_tolerance,
            return_annotated_image=query.return_annotated_image
        )

        # 5. Return results wrapped in a Pydantic response model
        return colourFilterResponse(
            tap_x=tap_x,
            tap_y=tap_y,
            selected_colour=selected_colour,
            colour_confidence=colour_confidence,
            image_with_boxes=annotated_image_url
        )
        
    except HTTPException:
        # If you already raised an HTTPException above, just re-raise it
        raise
    except ValueError as e:
        # Catch predictable errors (e.g. invalid image format, coordinates)
        logger.error(f"colour filtering error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as exc:
        # Catch *unexpected* errors so the API doesn't crash silently
        logger.error(f"Unexpected error during colour filtering: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during colour filtering"
        ) from exc
