"""
FastAPI application for ground-level visual localization service.
Provides REST API endpoints for real-time image-based geolocation.
"""

from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any
import numpy as np
from PIL import Image
import io
import os
import logging
import asyncio
import time
from datetime import datetime
import uuid

# Import core modules
import sys
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from core.localization_engine import GroundLevelLocalizer, LocalizationResult
from core.streetview_database import StreetViewDatabase, DatabaseBuilder
from models.feature_extractor import MultiModalFeatureExtractor
from utils.image_processing import ImageProcessor
from utils.validation import InputValidator

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Ground-Level Visual Localization API",
    description="AI-powered geolocation from street-level images",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for models (will be loaded on startup)
localizer: Optional[GroundLevelLocalizer] = None
image_processor: Optional[ImageProcessor] = None
validator: Optional[InputValidator] = None

# Pydantic models for API
class LocalizationRequest(BaseModel):
    method: str = Field(default="ensemble", description="Localization method")
    constraints: Optional[Dict[str, Any]] = Field(default=None, description="Geographic constraints")
    include_debug: bool = Field(default=False, description="Include debug information")

class LocationConstraints(BaseModel):
    bounds: Optional[Dict[str, float]] = None
    allowed_countries: Optional[List[str]] = None
    max_distance_km: Optional[float] = None

class LocalizationResponse(BaseModel):
    success: bool
    latitude: Optional[float] = None
    longitude: Optional[float] = None
    confidence: Optional[float] = None
    processing_time: Optional[float] = None
    top_matches: Optional[List[Dict]] = None
    cluster_analysis: Optional[Dict] = None
    method_breakdown: Optional[Dict] = None
    error_message: Optional[str] = None
    request_id: Optional[str] = None

class DatabaseStats(BaseModel):
    total_images: int
    by_source: Dict[str, int]
    feature_index_size: int
    geographic_bounds: Optional[Dict[str, float]]

class HealthResponse(BaseModel):
    status: str
    database_status: str
    model_status: str
    uptime_seconds: float
    version: str

# Startup event
@app.on_event("startup")
async def startup_event():
    """Initialize models and services on startup."""
    global localizer, image_processor, validator
    
    logger.info("Initializing Ground-Level Visual Localization API...")
    
    try:
        # Initialize components
        localizer = GroundLevelLocalizer()
        image_processor = ImageProcessor()
        validator = InputValidator()
        
        logger.info("✅ API initialization complete")
        
        # Log database stats
        stats = localizer.get_database_stats()
        logger.info(f"Database contains {stats['total_images']} images")
        
    except Exception as e:
        logger.error(f"❌ Failed to initialize API: {e}")
        raise

# Health check endpoint
@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    start_time = getattr(app.state, 'start_time', time.time())
    
    database_status = "healthy" if localizer and localizer.database else "unavailable"
    model_status = "healthy" if localizer and localizer.feature_extractor else "unavailable"
    
    return HealthResponse(
        status="healthy" if database_status == "healthy" and model_status == "healthy" else "degraded",
        database_status=database_status,
        model_status=model_status,
        uptime_seconds=time.time() - start_time,
        version="1.0.0"
    )

# Database statistics endpoint
@app.get("/stats", response_model=DatabaseStats)
async def get_database_stats():
    """Get database statistics."""
    if not localizer:
        raise HTTPException(status_code=503, detail="Localizer not initialized")
    
    try:
        stats = localizer.get_database_stats()
        return DatabaseStats(**stats)
    except Exception as e:
        logger.error(f"Failed to get database stats: {e}")
        raise HTTPException(status_code=500, detail="Failed to retrieve database statistics")

# Main localization endpoint
@app.post("/localize", response_model=LocalizationResponse)
async def localize_image(
    file: UploadFile = File(...),
    method: str = "ensemble",
    include_debug: bool = False,
    constraints: Optional[str] = None
):
    """
    Localize an uploaded image to geographic coordinates.
    
    Args:
        file: Image file to localize
        method: Localization method ('ensemble', 'matching', 'direct')
        include_debug: Include debug information in response
        constraints: JSON string of geographic constraints
        
    Returns:
        LocalizationResponse with predicted location and confidence
    """
    if not localizer:
        raise HTTPException(status_code=503, detail="Localizer not initialized")
    
    request_id = str(uuid.uuid4())
    start_time = time.time()
    
    try:
        # Validate file
        if not file.content_type or not file.content_type.startswith('image/'):
            raise HTTPException(status_code=400, detail="File must be an image")
        
        # Read and validate image
        image_data = await file.read()
        if len(image_data) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="Image too large (max 10MB)")
        
        # Convert to PIL Image
        try:
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
        except Exception as e:
            raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")
        
        # Validate image dimensions
        if image.size[0] < 100 or image.size[1] < 100:
            raise HTTPException(status_code=400, detail="Image too small (minimum 100x100 pixels)")
        
        # Process constraints
        constraint_dict = None
        if constraints:
            try:
                import json
                constraint_dict = json.loads(constraints)
            except json.JSONDecodeError:
                raise HTTPException(status_code=400, detail="Invalid constraints JSON")
        
        # Perform localization
        logger.info(f"Processing localization request {request_id} with method '{method}'")
        
        result = localizer.localize_image(
            image=image,
            method=method,
            constraints=constraint_dict
        )
        
        processing_time = time.time() - start_time
        
        # Prepare response
        response = LocalizationResponse(
            success=True,
            latitude=result.predicted_latitude,
            longitude=result.predicted_longitude,
            confidence=result.confidence,
            processing_time=processing_time,
            request_id=request_id
        )
        
        # Include debug information if requested
        if include_debug:
            response.top_matches = [
                {
                    'latitude': match['location'][0],
                    'longitude': match['location'][1],
                    'confidence': match['confidence'],
                    'source': match['image_data'].source,
                    'distance_km': match.get('distance_km')
                }
                for match in result.top_matches[:10]
            ]
            response.cluster_analysis = result.cluster_analysis
            response.method_breakdown = result.method_breakdown
        
        logger.info(f"Localization successful: {result.predicted_latitude:.6f}, {result.predicted_longitude:.6f} "
                   f"(confidence: {result.confidence:.3f}, time: {processing_time:.2f}s)")
        
        return response
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Localization failed for request {request_id}: {e}")
        return LocalizationResponse(
            success=False,
            error_message=str(e),
            processing_time=time.time() - start_time,
            request_id=request_id
        )

# Batch localization endpoint
@app.post("/localize/batch")
async def localize_batch(
    files: List[UploadFile] = File(...),
    method: str = "ensemble",
    background_tasks: BackgroundTasks = BackgroundTasks()
):
    """
    Localize multiple images in batch.
    
    Args:
        files: List of image files to localize
        method: Localization method
        background_tasks: Background task handler
        
    Returns:
        Batch job ID for status tracking
    """
    if not localizer:
        raise HTTPException(status_code=503, detail="Localizer not initialized")
    
    if len(files) > 20:  # Limit batch size
        raise HTTPException(status_code=400, detail="Maximum 20 images per batch")
    
    batch_id = str(uuid.uuid4())
    
    # Store batch job (in production, use proper job queue like Celery)
    batch_job = {
        'id': batch_id,
        'status': 'processing',
        'total_images': len(files),
        'completed': 0,
        'results': [],
        'created_at': datetime.now().isoformat()
    }
    
    # Add to background tasks
    background_tasks.add_task(process_batch_localization, batch_job, files, method)
    
    return {"batch_id": batch_id, "status": "processing", "total_images": len(files)}

async def process_batch_localization(batch_job: Dict, files: List[UploadFile], method: str):
    """Process batch localization in background."""
    try:
        for i, file in enumerate(files):
            try:
                # Read image
                image_data = await file.read()
                image = Image.open(io.BytesIO(image_data)).convert('RGB')
                
                # Localize
                result = localizer.localize_image(image, method=method)
                
                # Store result
                batch_job['results'].append({
                    'filename': file.filename,
                    'success': True,
                    'latitude': result.predicted_latitude,
                    'longitude': result.predicted_longitude,
                    'confidence': result.confidence
                })
                
            except Exception as e:
                batch_job['results'].append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
            
            batch_job['completed'] = i + 1
        
        batch_job['status'] = 'completed'
        
    except Exception as e:
        batch_job['status'] = 'failed'
        batch_job['error'] = str(e)

# Get localization methods endpoint
@app.get("/methods")
async def get_localization_methods():
    """Get available localization methods."""
    return {
        "methods": [
            {
                "name": "ensemble",
                "description": "Combines multiple localization approaches for best accuracy",
                "accuracy": "High",
                "speed": "Medium"
            },
            {
                "name": "matching",
                "description": "Feature matching against street-view database",
                "accuracy": "Medium-High",
                "speed": "Fast"
            },
            {
                "name": "direct",
                "description": "Direct neural network prediction from image features",
                "accuracy": "Medium",
                "speed": "Very Fast"
            }
        ]
    }

# Database management endpoints
@app.post("/database/rebuild")
async def rebuild_database(
    background_tasks: BackgroundTasks,
    image_directory: str,
    metadata_file: Optional[str] = None
):
    """
    Rebuild the street-view database from images.
    
    Args:
        image_directory: Directory containing street-view images
        metadata_file: Optional metadata JSON file
        
    Returns:
        Rebuild job status
    """
    if not os.path.exists(image_directory):
        raise HTTPException(status_code=400, detail="Image directory not found")
    
    rebuild_id = str(uuid.uuid4())
    
    # Add to background tasks
    background_tasks.add_task(
        rebuild_database_task, 
        rebuild_id, 
        image_directory, 
        metadata_file
    )
    
    return {"rebuild_id": rebuild_id, "status": "started"}

async def rebuild_database_task(rebuild_id: str, image_directory: str, metadata_file: Optional[str]):
    """Rebuild database in background."""
    try:
        logger.info(f"Starting database rebuild {rebuild_id}")
        
        # Initialize database builder
        feature_extractor = MultiModalFeatureExtractor()
        database = StreetViewDatabase()
        builder = DatabaseBuilder(database, feature_extractor)
        
        # Build database
        builder.build_database_from_directory(image_directory, metadata_file)
        
        logger.info(f"Database rebuild {rebuild_id} completed successfully")
        
    except Exception as e:
        logger.error(f"Database rebuild {rebuild_id} failed: {e}")

# Utility endpoints
@app.get("/")
async def root():
    """Root endpoint with API information."""
    return {
        "name": "Ground-Level Visual Localization API",
        "version": "1.0.0",
        "description": "AI-powered geolocation from street-level images",
        "endpoints": {
            "localize": "POST /localize - Localize single image",
            "batch": "POST /localize/batch - Batch localization",
            "health": "GET /health - Health check",
            "stats": "GET /stats - Database statistics",
            "methods": "GET /methods - Available methods",
            "docs": "GET /docs - API documentation"
        }
    }

# Error handlers
@app.exception_handler(404)
async def not_found_handler(request, exc):
    return JSONResponse(
        status_code=404,
        content={"error": "Endpoint not found", "message": "The requested endpoint does not exist"}
    )

@app.exception_handler(500)
async def internal_error_handler(request, exc):
    logger.error(f"Internal server error: {exc}")
    return JSONResponse(
        status_code=500,
        content={"error": "Internal server error", "message": "An unexpected error occurred"}
    )

# Set startup time
app.state.start_time = time.time()

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)