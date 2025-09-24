from fastapi import FastAPI
from app.core.logging import configure_logging
from app.api.routes import api_router


def create_app() -> FastAPI:
    """
    Factory function that creates and configures the FastAPI app.
    This pattern keeps things organized and makes testing easier.
    """

    # Set up logging before anything else so all logs are consistent
    configure_logging()

    # Create the FastAPI application instance
    application = FastAPI(
        title="Climb Detection API", 
        version="0.1.0",             
    )

    # Register all API routes from your `api_router`
    application.include_router(api_router)

    # Return the configured application
    return application


# Create the app instance that will be used by Uvicorn/Gunicorn
# e.g. `uvicorn app.main:app --reload`
app = create_app()
