"""
Deep learning-based feature extraction for ground-level images.
Uses pre-trained models optimized for place recognition and visual localization.
"""

import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision.models import resnet50, efficientnet_b4, vit_b_16
from transformers import CLIPModel, CLIPProcessor
import numpy as np
from typing import Tuple, List, Optional, Dict, Union
import logging
import cv2
from PIL import Image
import io

logger = logging.getLogger(__name__)

class StreetViewFeatureExtractor:
    """Extract deep features from street-level images for place recognition."""
    
    def __init__(self, model_name: str = "clip", device: str = "auto"):
        """
        Initialize feature extractor.
        
        Args:
            model_name: Model to use ('clip', 'resnet50', 'efficientnet', 'vit')
            device: Device to use ('cpu', 'cuda', 'auto')
        """
        self.model_name = model_name
        self.device = self._get_device(device)
        self.model = None
        self.processor = None
        self.transform = None
        self.feature_dim = None
        
        self._load_model()
        
    def _get_device(self, device: str) -> torch.device:
        """Get appropriate device for inference."""
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)
    
    def _load_model(self):
        """Load the specified model."""
        logger.info(f"Loading {self.model_name} model on {self.device}")
        
        if self.model_name == "clip":
            self._load_clip_model()
        elif self.model_name == "resnet50":
            self._load_resnet_model()
        elif self.model_name == "efficientnet":
            self._load_efficientnet_model()
        elif self.model_name == "vit":
            self._load_vit_model()
        else:
            raise ValueError(f"Unsupported model: {self.model_name}")
    
    def _load_clip_model(self):
        """Load CLIP model for multimodal embeddings."""
        model_id = "openai/clip-vit-base-patch32"
        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)
        self.feature_dim = 512
        self.model.eval()
    
    def _load_resnet_model(self):
        """Load ResNet-50 model for image features."""
        self.model = resnet50(pretrained=True)
        # Remove final classification layer
        self.model = nn.Sequential(*list(self.model.children())[:-1])
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = 2048
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_efficientnet_model(self):
        """Load EfficientNet model for efficient feature extraction."""
        self.model = efficientnet_b4(pretrained=True)
        # Remove final classification layer
        self.model.classifier = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = 1792
        self.transform = transforms.Compose([
            transforms.Resize((380, 380)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def _load_vit_model(self):
        """Load Vision Transformer model."""
        self.model = vit_b_16(pretrained=True)
        # Remove final classification layer
        self.model.heads = nn.Identity()
        self.model.to(self.device)
        self.model.eval()
        
        self.feature_dim = 768
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    
    def extract_features(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Extract features from an image.
        
        Args:
            image: Input image (numpy array, PIL Image, or file path)
            
        Returns:
            Feature vector as numpy array
        """
        # Convert input to PIL Image
        pil_image = self._to_pil_image(image)
        
        if self.model_name == "clip":
            return self._extract_clip_features(pil_image)
        else:
            return self._extract_cnn_features(pil_image)
    
    def _to_pil_image(self, image: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        """Convert various image formats to PIL Image."""
        if isinstance(image, str):
            return Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            if image.shape[2] == 3:  # BGR to RGB
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            return Image.fromarray(image)
        elif isinstance(image, Image.Image):
            return image.convert('RGB')
        else:
            raise ValueError(f"Unsupported image type: {type(image)}")
    
    def _extract_clip_features(self, image: Image.Image) -> np.ndarray:
        """Extract features using CLIP model."""
        with torch.no_grad():
            inputs = self.processor(images=image, return_tensors="pt").to(self.device)
            image_features = self.model.get_image_features(**inputs)
            
            # Normalize features
            image_features = image_features / image_features.norm(p=2, dim=-1, keepdim=True)
            
            return image_features.cpu().numpy().flatten()
    
    def _extract_cnn_features(self, image: Image.Image) -> np.ndarray:
        """Extract features using CNN models."""
        with torch.no_grad():
            # Apply transforms
            tensor_image = self.transform(image).unsqueeze(0).to(self.device)
            
            # Extract features
            features = self.model(tensor_image)
            
            # Flatten and normalize
            features = features.view(features.size(0), -1)
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            
            return features.cpu().numpy().flatten()
    
    def extract_batch_features(self, images: List[Union[np.ndarray, Image.Image, str]], 
                              batch_size: int = 32) -> np.ndarray:
        """
        Extract features from multiple images in batches.
        
        Args:
            images: List of images
            batch_size: Batch size for processing
            
        Returns:
            Array of feature vectors
        """
        all_features = []
        
        for i in range(0, len(images), batch_size):
            batch = images[i:i + batch_size]
            batch_features = []
            
            for image in batch:
                features = self.extract_features(image)
                batch_features.append(features)
            
            all_features.extend(batch_features)
        
        return np.array(all_features)

class GeolocationFeatureExtractor:
    """Extract location-specific features from images."""
    
    def __init__(self):
        """Initialize geolocation feature extractor."""
        self.feature_extractors = {
            'architectural': ArchitecturalFeatureExtractor(),
            'signage': SignageFeatureExtractor(),
            'vegetation': VegetationFeatureExtractor(),
            'vehicles': VehicleFeatureExtractor()
        }
    
    def extract_geolocation_features(self, image: Union[np.ndarray, Image.Image]) -> Dict[str, np.ndarray]:
        """
        Extract location-specific features from image.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary of feature types and their vectors
        """
        features = {}
        
        for feature_type, extractor in self.feature_extractors.items():
            try:
                features[feature_type] = extractor.extract(image)
            except Exception as e:
                logger.warning(f"Failed to extract {feature_type} features: {e}")
                features[feature_type] = np.array([])
        
        return features

class ArchitecturalFeatureExtractor:
    """Extract architectural features that indicate geographic location."""
    
    def __init__(self):
        """Initialize architectural feature extractor."""
        # This would use a model trained on architectural styles
        # For demo, we'll use a simple approach
        pass
    
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract architectural features."""
        # Convert to numpy array
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Simple architectural features (edges, textures)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Edge features
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255.0
        
        # Texture features using LBP-like approach
        texture_features = self._extract_texture_features(gray)
        
        # Combine features
        features = np.concatenate([
            [edge_density],
            texture_features
        ])
        
        return features
    
    def _extract_texture_features(self, gray_image: np.ndarray) -> np.ndarray:
        """Extract simple texture features."""
        # Calculate local variance as texture measure
        kernel = np.ones((5, 5), np.float32) / 25
        local_mean = cv2.filter2D(gray_image.astype(np.float32), -1, kernel)
        local_variance = cv2.filter2D((gray_image.astype(np.float32) - local_mean) ** 2, -1, kernel)
        
        # Statistical measures
        variance_mean = np.mean(local_variance)
        variance_std = np.std(local_variance)
        
        return np.array([variance_mean, variance_std])

class SignageFeatureExtractor:
    """Extract features from text and signage in images."""
    
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract signage features."""
        # This would use OCR and text analysis
        # For demo, we'll extract basic text-like regions
        
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Find text-like regions using MSER
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        # Basic statistics about text regions
        num_regions = len(regions)
        if num_regions > 0:
            region_sizes = [len(region) for region in regions]
            avg_region_size = np.mean(region_sizes)
            region_density = num_regions / (gray.shape[0] * gray.shape[1])
        else:
            avg_region_size = 0
            region_density = 0
        
        return np.array([num_regions, avg_region_size, region_density])

class VegetationFeatureExtractor:
    """Extract vegetation features that may indicate geographic region."""
    
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract vegetation features."""
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Convert to HSV for better vegetation detection
        hsv = cv2.cvtColor(img_array, cv2.COLOR_RGB2HSV)
        
        # Green vegetation mask
        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Vegetation statistics
        vegetation_ratio = np.sum(green_mask > 0) / (green_mask.shape[0] * green_mask.shape[1])
        
        # Green color statistics
        green_pixels = img_array[green_mask > 0]
        if len(green_pixels) > 0:
            green_mean = np.mean(green_pixels, axis=0)
            green_std = np.std(green_pixels, axis=0)
        else:
            green_mean = np.array([0, 0, 0])
            green_std = np.array([0, 0, 0])
        
        features = np.concatenate([
            [vegetation_ratio],
            green_mean,
            green_std
        ])
        
        return features

class VehicleFeatureExtractor:
    """Extract vehicle features that may indicate geographic region."""
    
    def extract(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Extract vehicle features."""
        # This would use a vehicle detection model
        # For demo, we'll use simple color-based features
        
        if isinstance(image, Image.Image):
            img_array = np.array(image)
        else:
            img_array = image
        
        # Simple vehicle-like color detection (cars are often dark)
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        
        # Dark regions that might be vehicles
        dark_mask = gray < 80
        dark_ratio = np.sum(dark_mask) / (gray.shape[0] * gray.shape[1])
        
        # Horizontal edge features (cars have many horizontal lines)
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        horizontal_edges = np.sum(np.abs(sobelx) > np.abs(sobely))
        edge_ratio = horizontal_edges / (gray.shape[0] * gray.shape[1])
        
        return np.array([dark_ratio, edge_ratio])

class MultiModalFeatureExtractor:
    """Combine multiple feature extraction approaches."""
    
    def __init__(self, primary_model: str = "clip"):
        """
        Initialize multi-modal feature extractor.
        
        Args:
            primary_model: Primary deep learning model to use
        """
        self.primary_extractor = StreetViewFeatureExtractor(primary_model)
        self.geo_extractor = GeolocationFeatureExtractor()
    
    def extract_comprehensive_features(self, image: Union[np.ndarray, Image.Image, str]) -> Dict[str, np.ndarray]:
        """
        Extract comprehensive features for place recognition.
        
        Args:
            image: Input image
            
        Returns:
            Dictionary with different feature types
        """
        # Convert to PIL for consistent processing
        if isinstance(image, str):
            pil_image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            pil_image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        else:
            pil_image = image
        
        features = {}
        
        # Primary deep features
        features['deep'] = self.primary_extractor.extract_features(pil_image)
        
        # Geographic features
        geo_features = self.geo_extractor.extract_geolocation_features(pil_image)
        features.update(geo_features)
        
        return features
    
    def get_combined_feature_vector(self, image: Union[np.ndarray, Image.Image, str]) -> np.ndarray:
        """
        Get a single combined feature vector.
        
        Args:
            image: Input image
            
        Returns:
            Combined feature vector
        """
        all_features = self.extract_comprehensive_features(image)
        
        # Concatenate all non-empty features
        feature_vectors = []
        for feature_type, features in all_features.items():
            if len(features) > 0:
                feature_vectors.append(features)
        
        if feature_vectors:
            combined = np.concatenate(feature_vectors)
            # Normalize combined vector
            combined = combined / np.linalg.norm(combined)
            return combined
        else:
            # Return zero vector if no features extracted
            return np.zeros(self.primary_extractor.feature_dim)