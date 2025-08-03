"""
Input validation utilities for the localization system.
"""

import os
import cv2
import numpy as np
from PIL import Image
from typing import Tuple, Optional, List, Union
import logging

logger = logging.getLogger(__name__)

class ValidationResult:
    """Container for validation results."""
    
    def __init__(self, is_valid: bool, message: str = "", details: dict = None):
        self.is_valid = is_valid
        self.message = message
        self.details = details or {}

class InputValidator:
    """Validates various inputs for the localization system."""
    
    @staticmethod
    def validate_coordinates(lat: float, lon: float) -> ValidationResult:
        """Validate latitude and longitude coordinates."""
        if not isinstance(lat, (int, float)) or not isinstance(lon, (int, float)):
            return ValidationResult(False, "Coordinates must be numeric")
        
        if not (-90 <= lat <= 90):
            return ValidationResult(
                False, 
                f"Invalid latitude: {lat}. Must be between -90 and 90 degrees.",
                {"latitude": lat}
            )
        
        if not (-180 <= lon <= 180):
            return ValidationResult(
                False,
                f"Invalid longitude: {lon}. Must be between -180 and 180 degrees.",
                {"longitude": lon}
            )
        
        return ValidationResult(True, "Coordinates are valid")
    
    @staticmethod
    def validate_image_file(image_path: str) -> ValidationResult:
        """Validate image file exists and is readable."""
        if not isinstance(image_path, str):
            return ValidationResult(False, "Image path must be a string")
        
        if not os.path.exists(image_path):
            return ValidationResult(
                False,
                f"Image file not found: {image_path}",
                {"path": image_path}
            )
        
        if not os.path.isfile(image_path):
            return ValidationResult(
                False,
                f"Path is not a file: {image_path}",
                {"path": image_path}
            )
        
        # Check file extension
        valid_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        ext = os.path.splitext(image_path)[1].lower()
        if ext not in valid_extensions:
            return ValidationResult(
                False,
                f"Unsupported image format: {ext}. Supported: {valid_extensions}",
                {"extension": ext, "path": image_path}
            )
        
        # Try to load the image
        try:
            img = cv2.imread(image_path)
            if img is None:
                return ValidationResult(
                    False,
                    f"Cannot read image file: {image_path}. File may be corrupted.",
                    {"path": image_path}
                )
            
            height, width = img.shape[:2]
            if height < 100 or width < 100:
                return ValidationResult(
                    False,
                    f"Image too small: {width}x{height}. Minimum size is 100x100 pixels.",
                    {"width": width, "height": height, "path": image_path}
                )
            
            return ValidationResult(
                True,
                "Image file is valid",
                {
                    "path": image_path,
                    "width": width,
                    "height": height,
                    "channels": img.shape[2] if len(img.shape) > 2 else 1,
                    "size_mb": os.path.getsize(image_path) / (1024 * 1024)
                }
            )
            
        except Exception as e:
            return ValidationResult(
                False,
                f"Error reading image: {e}",
                {"path": image_path, "error": str(e)}
            )
    
    @staticmethod
    def validate_image_array(image: np.ndarray) -> ValidationResult:
        """Validate numpy image array."""
        if not isinstance(image, np.ndarray):
            return ValidationResult(False, "Image must be numpy array")
        
        if len(image.shape) not in [2, 3]:
            return ValidationResult(
                False,
                f"Invalid image dimensions: {image.shape}. Must be 2D or 3D array."
            )
        
        height, width = image.shape[:2]
        if height < 50 or width < 50:
            return ValidationResult(
                False,
                f"Image too small: {width}x{height}. Minimum size is 50x50 pixels."
            )
        
        if len(image.shape) == 3 and image.shape[2] not in [1, 3, 4]:
            return ValidationResult(
                False,
                f"Invalid number of channels: {image.shape[2]}. Must be 1, 3, or 4."
            )
        
        return ValidationResult(
            True,
            "Image array is valid",
            {
                "width": width,
                "height": height,
                "channels": image.shape[2] if len(image.shape) > 2 else 1,
                "dtype": str(image.dtype)
            }
        )