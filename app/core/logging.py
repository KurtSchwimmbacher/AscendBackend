import logging
import sys
from app.core.config import settings


def configure_logging() -> None:
    
    """
    Configure the root logger for the application.
    Ensures logs are formatted consistently and sent to stdout.
    """

    # Get the root logger (the base logger that all others inherit from)
    root = logging.getLogger()

    # If logging has already been configured (handlers exist), do nothing
    if root.handlers:
        return

    # Convert the configured log level (string from settings) to an actual logging constant.
    # e.g. "INFO" -> logging.INFO, "DEBUG" -> logging.DEBUG
    level = getattr(logging, settings.log_level.upper(), logging.INFO)

    # Create a stream handler that writes logs to standard output (console)
    handler = logging.StreamHandler(sys.stdout)

    # Define how log messages should look (format + timestamp format)
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(name)s | %(message)s",  # Example: 2025-09-24 15:00:00 | INFO | myapp | Starting...
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Attach the formatter to the handler so logs use that style
    handler.setFormatter(formatter)

    # Set the root logger’s level (this determines which messages get through)
    root.setLevel(level)

    # Add the handler so logs are actually output somewhere (stdout)
    root.addHandler(handler)
