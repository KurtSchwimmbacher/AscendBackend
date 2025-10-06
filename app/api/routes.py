import logging
from fastapi import APIRouter, File, UploadFile, Depends, HTTPException

# Import Pydantic request/response schemas for type validation and docs
from app.schemas.detection import (
    DetectRequestQuery,
    DetectResponse,
    colourFilterRequest,
    colourFilterResponse,
    HealthResponse,
    RouteByColourResponse,
)

# Import your YOLO prediction service
from app.services.yolo_service import predict_holds
from app.services.colour_detection_service import predict_holds_by_colour
from app.services.color_service import (
    get_color_at_pixel,
    classify_color,
    filter_detections_by_color,
    draw_bounding_boxes,
    save_annotated_image,
)

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
# Detect Route Holds By Tap Colour (orchestrator)
# ------------------------------
@api_router.post("/routes/detect-by-colour", response_model=RouteByColourResponse)
async def detect_route_by_colour(
    file: UploadFile = File(..., description="Image file (JPG, PNG, etc.)"),
    query: colourFilterRequest = Depends(),
) -> RouteByColourResponse:
    """
    Use the user's tap to select a target colour, run YOLO hold detection,
    filter holds by the selected colour, and return an annotated image of matches.
    """

    # 1. Validate file type (only images are allowed)
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid file type: {file.content_type}. Please upload an image file.")

    # 2. Validate file size (limit to 10MB)
    max_size = 10 * 1024 * 1024  # 10MB
    if file.size and file.size > max_size:
        raise HTTPException(
            status_code=400,
            detail=f"File too large: {file.size} bytes. Maximum size is {max_size} bytes.")

    try:
        # Read bytes
        contents = await file.read()
        if not contents:
            raise HTTPException(status_code=400, detail="Empty file uploaded")

        # Run YOLO to get detections (no annotated image yet)
        detections, _ = predict_holds(
            contents,
            conf=query.conf,
            return_annotated_image=False,
        )

        # Load PIL image once for colour ops
        from PIL import Image
        import io as _io
        image = Image.open(_io.BytesIO(contents))
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Determine target colour at tap
        hsv_at_tap = get_color_at_pixel(image, query.tap_x, query.tap_y, region_size=5)
        selected_colour, colour_confidence = classify_color(hsv_at_tap)

        # Filter detections by colour tolerance
        filtered = filter_detections_by_color(
            detections=detections,
            image=image,
            target_hsv=hsv_at_tap,
            tolerance=query.colour_tolerance,
        )

        # Create annotated image and draw a circle at the tap point
        annotated_url = None
        if query.return_annotated_image:
            try:
                # Start from original image, draw boxes for filtered detections if any
                annotated = draw_bounding_boxes(image, filtered, conf_threshold=query.conf or 0.0) if filtered else image.copy()

                # Draw a circle around the selected tap point
                from PIL import ImageDraw as _ImageDraw
                draw = _ImageDraw.Draw(annotated)
                circle_radius = 10
                x, y = int(query.tap_x), int(query.tap_y)
                draw.ellipse([
                    x - circle_radius,
                    y - circle_radius,
                    x + circle_radius,
                    y + circle_radius,
                ], outline=(255, 255, 255), width=3)

                annotated_url = save_annotated_image(annotated)
            except Exception as viz_error:
                logger.warning(f"Failed to create annotated image for route-by-colour: {viz_error}")
                annotated_url = None

        return RouteByColourResponse(
            tap_x=query.tap_x,
            tap_y=query.tap_y,
            selected_colour=selected_colour,
            colour_confidence=colour_confidence,
            detections=filtered,
            image_with_boxes=annotated_url,
        )

    except HTTPException:
        raise
    except ValueError as e:
        logger.error(f"route-by-colour error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e)) from e
    except Exception as exc:
        logger.error(f"Unexpected error during route-by-colour: {str(exc)}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error during route-by-colour",
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
        # If already raised an HTTPException above, just re-raise it
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
