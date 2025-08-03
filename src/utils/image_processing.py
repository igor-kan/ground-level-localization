"""
Image processing utilities for ground-level localization.
"""

import cv2
import numpy as np
from PIL import Image, ImageEnhance, ExifTags
from typing import Tuple, Optional, Dict, Union
import logging

logger = logging.getLogger(__name__)

class ImageProcessor:
    """Utility class for image preprocessing and enhancement."""
    
    def __init__(self):
        """Initialize image processor."""
        pass
    
    def preprocess_image(self, image: Union[np.ndarray, Image.Image], 
                        target_size: Optional[Tuple[int, int]] = None,
                        enhance: bool = True) -> Image.Image:
        """
        Preprocess image for optimal feature extraction.
        
        Args:
            image: Input image
            target_size: Optional target size (width, height)
            enhance: Whether to apply image enhancement
            
        Returns:
            Preprocessed PIL Image
        """
        # Convert to PIL Image if needed
        if isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            pil_image = Image.fromarray(image)
        else:
            pil_image = image.convert('RGB')
        
        # Resize if target size specified
        if target_size:
            pil_image = pil_image.resize(target_size, Image.LANCZOS)
        
        # Apply enhancements
        if enhance:
            pil_image = self._enhance_image(pil_image)
        
        return pil_image
    
    def _enhance_image(self, image: Image.Image) -> Image.Image:
        """Apply image enhancements for better feature extraction."""
        # Enhance contrast
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(1.2)
        
        # Enhance sharpness
        enhancer = ImageEnhance.Sharpness(image)
        image = enhancer.enhance(1.1)
        
        # Enhance color
        enhancer = ImageEnhance.Color(image)
        image = enhancer.enhance(1.1)
        
        return image
    
    def extract_exif_data(self, image_path: str) -> Dict:
        """Extract EXIF data from image."""
        try:
            image = Image.open(image_path)
            exif_data = {}
            
            if hasattr(image, '_getexif') and image._getexif() is not None:
                exif = image._getexif()
                
                for tag_id, value in exif.items():
                    tag = ExifTags.TAGS.get(tag_id, tag_id)
                    exif_data[tag] = value
            
            # Extract GPS coordinates if available
            gps_coords = self._extract_gps_coordinates(exif_data)
            if gps_coords:
                exif_data['gps_coordinates'] = gps_coords
            
            return exif_data
            
        except Exception as e:
            logger.warning(f"Failed to extract EXIF data: {e}")
            return {}
    
    def _extract_gps_coordinates(self, exif_data: Dict) -> Optional[Tuple[float, float]]:
        """Extract GPS coordinates from EXIF data."""
        try:
            gps_info = exif_data.get('GPSInfo', {})
            if not gps_info:
                return None
            
            # Extract latitude
            lat_ref = gps_info.get(1)
            lat_dms = gps_info.get(2)
            
            # Extract longitude  
            lon_ref = gps_info.get(3)
            lon_dms = gps_info.get(4)
            
            if not all([lat_ref, lat_dms, lon_ref, lon_dms]):
                return None
            
            # Convert DMS to decimal degrees
            lat = self._dms_to_decimal(lat_dms)
            lon = self._dms_to_decimal(lon_dms)
            
            # Apply direction
            if lat_ref == 'S':
                lat = -lat
            if lon_ref == 'W':
                lon = -lon
            
            return (lat, lon)
            
        except Exception:
            return None
    
    def _dms_to_decimal(self, dms: Tuple[float, float, float]) -> float:
        """Convert degrees, minutes, seconds to decimal degrees."""
        degrees, minutes, seconds = dms
        return degrees + minutes/60.0 + seconds/3600.0