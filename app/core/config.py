from pydantic import Field
from pydantic_settings import BaseSettings


# Define a settings class that loads configuration values
class Settings(BaseSettings):


    # ------------------------------------------------------------------
    # General Settings
    # ------------------------------------------------------------------

    # Which environment the app is running in (local, dev, prod, etc.)
    environment: str = Field("local", description="Environment name")

    # The logging level (e.g., DEBUG, INFO, WARNING, ERROR)
    log_level: str = Field("INFO", description="Logging level")

    # ------------------------------------------------------------------
    # Model Settings
    # ------------------------------------------------------------------

    # The Hugging Face repo where the YOLO model is stored
    hf_repo_id: str = Field(
        "jwlarocque/yolov8n-freeclimbs-detect-2",
        description="Hugging Face repo id for the YOLOv8 model",
    )

    # The specific filename of the model weights within the repo
    hf_filename: str = Field(
        "yolov8n-freeclimbs-detect-2.pt",
        description="Specific model weight file name in the repo",
    )

    # Image size used for YOLO inference (larger = more detail, slower)
    yolo_imgsz: int = Field(2560, description="YOLO inference image size")

    # Maximum number of detections YOLO should return
    # set to a high number for now, filtering can happen later
    yolo_max_det: int = Field(2000, description="YOLO max detections")

    # Minimum confidence threshold for YOLO predictions
    # can also be upped later 
    yolo_conf: float = Field(0.25, description="Default confidence threshold")

    class Config:
        # Tell Pydantic to automatically load variables from a `.env` file
        env_file = ".env"
        # Ensure the file is read using UTF-8 encoding
        env_file_encoding = "utf-8"


# Create a global `settings` object that can be imported and used throughout the app to access config values (settings.environment, etc.)
settings = Settings()
