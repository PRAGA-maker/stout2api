from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
from contextlib import asynccontextmanager
import logging
import traceback
import ray
from api.v1.endpoints import stout
from api.stout_service import stout_service
import os

# Suppress TensorFlow warnings about duplicate registrations
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app):
    logger.info("Starting up STOUT API...")
    try:
        # Set Ray memory usage threshold before Ray initialization
        os.environ["RAY_memory_usage_threshold"] = "0.8"
        # Initialize Ray for parallel processing
        if not ray.is_initialized():
            try:
                # For Azure container deployment, use local mode with resource limits
                ray.init(
                    ignore_reinit_error=True,
                    local_mode=False,  # Enable parallel processing
                    num_cpus=None,  # Use all available CPUs
                    object_store_memory=None,  # Use default memory
                    _memory=None,  # Use default memory
                    log_to_driver=True,  # Log to main process
                    include_dashboard=True,  # Enable dashboard for monitoring
                    dashboard_host="0.0.0.0",  # Bind to all interfaces
                    dashboard_port=8265,  # Default Ray dashboard port
                )
                logger.info("Ray initialized successfully with dashboard")
            except Exception as dashboard_error:
                logger.warning(f"Failed to initialize Ray with dashboard: {dashboard_error}")
                # Fallback: Initialize Ray without dashboard
                try:
                    ray.init(
                        ignore_reinit_error=True,
                        local_mode=False,  # Enable parallel processing
                        num_cpus=None,  # Use all available CPUs
                        object_store_memory=None,  # Use default memory
                        _memory=None,  # Use default memory
                        log_to_driver=True,  # Log to main process
                    )
                    logger.info("Ray initialized successfully without dashboard")
                except Exception as fallback_error:
                    logger.error(f"Failed to initialize Ray: {fallback_error}")
                    raise
        else:
            logger.info("Ray already initialized")
        
        # Load models on startup to avoid cold starts
        stout_service.load_models()
        
        # Initialize dynamic scaling system
        from api.utils.resource_manager import resource_manager
        from api.utils.dynamic_actor_factory import actor_factory
        
        # Test dynamic scaling system
        system_status = resource_manager.get_system_status()
        logger.info(f"Dynamic scaling system initialized: {system_status}")
        
        logger.info("STOUT API ready!")
    except Exception as e:
        logger.error(f"Failed to start STOUT API: {e}")
        raise
    yield
    logger.info("Shutting down STOUT API...")
    try:
        # Shutdown Ray gracefully
        if ray.is_initialized():
            ray.shutdown()
            logger.info("Ray shutdown successfully")
    except Exception as e:
        logger.error(f"Error during Ray shutdown: {e}")

app = FastAPI(
    title="STOUT API",
    description="API for STOUT (SMILES to IUPAC and IUPAC to SMILES) translation",
    version="1.0.0",
    lifespan=lifespan
)

# Global exception handler
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Global exception handler for unhandled exceptions"""
    logger.error(f"Unhandled exception: {exc}")
    logger.error(f"Request URL: {request.url}")
    logger.error(f"Request method: {request.method}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": "An unexpected error occurred. Please try again later.",
            "type": "internal_error"
        }
    )

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    """Handler for HTTP exceptions"""
    logger.warning(f"HTTP exception: {exc.status_code} - {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": "HTTP error",
            "detail": exc.detail,
            "status_code": exc.status_code
        }
    )

# Include routers
app.include_router(stout.router)

@app.get("/")
def root():
    """Root endpoint with API information"""
    try:
        return {
            "message": "STOUT API",
            "version": "1.0.0",
            "docs": "/docs",
            "health": "/stout/health",
            "status": "running",
            "models_loaded": stout_service.models_loaded if hasattr(stout_service, 'models_loaded') else False
        }
    except Exception as e:
        logger.error(f"Root endpoint error: {e}")
        return {
            "message": "STOUT API",
            "version": "1.0.0",
            "status": "error",
            "error": str(e)
        }

@app.get("/health")
def health_check():
    """Health check endpoint"""
    try:
        return {
            "status": "healthy",
            "models_loaded": stout_service.models_loaded if hasattr(stout_service, 'models_loaded') else False,
            "cdk_available": stout_service.cdk_available if hasattr(stout_service, 'cdk_available') else False
        }
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return {
            "status": "unhealthy",
            "error": str(e)
        } 