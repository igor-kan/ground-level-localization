"""
Main localization engine that combines feature matching with geolocation prediction.
Uses ensemble of neural networks and geometric constraints for accurate location estimation.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Tuple, Optional, Union
import logging
from dataclasses import dataclass
from sklearn.cluster import DBSCAN
from geopy.distance import geodesic
import cv2
from PIL import Image

from .streetview_database import StreetViewDatabase, StreetViewMatcher, StreetViewImage
from ..models.feature_extractor import MultiModalFeatureExtractor

logger = logging.getLogger(__name__)

@dataclass
class LocalizationResult:
    """Container for localization results."""
    predicted_latitude: float
    predicted_longitude: float
    confidence: float
    top_matches: List[Dict]
    cluster_analysis: Optional[Dict] = None
    processing_time: float = 0.0
    method_breakdown: Optional[Dict] = None

class GeolocationPredictor(nn.Module):
    """Neural network for direct geolocation prediction from features."""
    
    def __init__(self, feature_dim: int, hidden_dims: List[int] = [1024, 512, 256]):
        """
        Initialize geolocation predictor.
        
        Args:
            feature_dim: Dimension of input features
            hidden_dims: Hidden layer dimensions
        """
        super().__init__()
        
        layers = []
        input_dim = feature_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.3)
            ])
            input_dim = hidden_dim
        
        # Output layer for lat/lon
        layers.append(nn.Linear(input_dim, 2))
        
        self.network = nn.Sequential(*layers)
        
        # Coordinate normalization parameters (will be set during training)
        self.register_buffer('lat_mean', torch.tensor(0.0))
        self.register_buffer('lat_std', torch.tensor(1.0))
        self.register_buffer('lon_mean', torch.tensor(0.0))
        self.register_buffer('lon_std', torch.tensor(1.0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        normalized_coords = self.network(x)
        
        # Denormalize coordinates
        lat = normalized_coords[:, 0] * self.lat_std + self.lat_mean
        lon = normalized_coords[:, 1] * self.lon_std + self.lon_mean
        
        return torch.stack([lat, lon], dim=1)
    
    def set_normalization_params(self, lat_stats: Tuple[float, float], 
                                lon_stats: Tuple[float, float]):
        """Set coordinate normalization parameters."""
        self.lat_mean = torch.tensor(lat_stats[0])
        self.lat_std = torch.tensor(lat_stats[1])
        self.lon_mean = torch.tensor(lon_stats[0])
        self.lon_std = torch.tensor(lon_stats[1])

class EnsembleLocalizer:
    """Ensemble localization using multiple methods."""
    
    def __init__(self, database: StreetViewDatabase, feature_extractor: MultiModalFeatureExtractor,
                 geolocation_model_path: Optional[str] = None):
        """
        Initialize ensemble localizer.
        
        Args:
            database: Street-view database
            feature_extractor: Feature extraction model
            geolocation_model_path: Path to trained geolocation model
        """
        self.database = database
        self.feature_extractor = feature_extractor
        self.matcher = StreetViewMatcher(database, feature_extractor)
        
        # Load geolocation predictor if available
        self.geolocation_predictor = None
        if geolocation_model_path:
            self._load_geolocation_model(geolocation_model_path)
    
    def _load_geolocation_model(self, model_path: str):
        """Load trained geolocation prediction model."""
        try:
            checkpoint = torch.load(model_path, map_location='cpu')
            
            feature_dim = checkpoint.get('feature_dim', 512)
            self.geolocation_predictor = GeolocationPredictor(feature_dim)
            self.geolocation_predictor.load_state_dict(checkpoint['model_state_dict'])
            self.geolocation_predictor.eval()
            
            logger.info(f"Loaded geolocation model from {model_path}")
        except Exception as e:
            logger.warning(f"Failed to load geolocation model: {e}")
            self.geolocation_predictor = None
    
    def localize(self, query_image: Union[str, np.ndarray, Image.Image],
                method: str = "ensemble", top_k: int = 20) -> LocalizationResult:
        """
        Localize query image using specified method.
        
        Args:
            query_image: Input image
            method: Localization method ('ensemble', 'matching', 'direct')
            top_k: Number of top matches to consider
            
        Returns:
            LocalizationResult with predicted location and confidence
        """
        import time
        start_time = time.time()
        
        if method == "ensemble":
            result = self._ensemble_localization(query_image, top_k)
        elif method == "matching":
            result = self._matching_based_localization(query_image, top_k)
        elif method == "direct":
            result = self._direct_localization(query_image)
        else:
            raise ValueError(f"Unknown localization method: {method}")
        
        result.processing_time = time.time() - start_time
        return result
    
    def _ensemble_localization(self, query_image: Union[str, np.ndarray, Image.Image],
                              top_k: int = 20) -> LocalizationResult:
        """Ensemble localization combining multiple methods."""
        # Method 1: Feature matching
        matching_result = self._matching_based_localization(query_image, top_k)
        
        # Method 2: Direct prediction (if available)
        direct_result = None
        if self.geolocation_predictor:
            direct_result = self._direct_localization(query_image)
        
        # Method 3: Clustering analysis
        cluster_result = self._cluster_based_localization(matching_result.top_matches)
        
        # Combine results
        final_lat, final_lon, confidence = self._combine_predictions(
            matching_result, direct_result, cluster_result
        )
        
        method_breakdown = {
            'matching_weight': 0.4,
            'direct_weight': 0.3 if direct_result else 0.0,
            'cluster_weight': 0.3 if direct_result else 0.6,
            'matching_confidence': matching_result.confidence,
            'direct_confidence': direct_result.confidence if direct_result else 0.0,
            'cluster_confidence': cluster_result['confidence']
        }
        
        return LocalizationResult(
            predicted_latitude=final_lat,
            predicted_longitude=final_lon,
            confidence=confidence,
            top_matches=matching_result.top_matches,
            cluster_analysis=cluster_result,
            method_breakdown=method_breakdown
        )
    
    def _matching_based_localization(self, query_image: Union[str, np.ndarray, Image.Image],
                                   top_k: int = 20) -> LocalizationResult:
        """Localization based on feature matching."""
        matches = self.matcher.find_location(query_image, top_k=top_k)
        
        if not matches:
            return LocalizationResult(
                predicted_latitude=0.0,
                predicted_longitude=0.0,
                confidence=0.0,
                top_matches=[]
            )
        
        # Weighted average of top matches
        total_weight = 0.0
        weighted_lat = 0.0
        weighted_lon = 0.0
        
        for i, match in enumerate(matches):
            # Weight decreases with rank, increases with similarity
            rank_weight = 1.0 / (i + 1)
            similarity_weight = match['confidence']
            weight = rank_weight * similarity_weight
            
            weighted_lat += match['location'][0] * weight
            weighted_lon += match['location'][1] * weight
            total_weight += weight
        
        if total_weight > 0:
            predicted_lat = weighted_lat / total_weight
            predicted_lon = weighted_lon / total_weight
        else:
            predicted_lat = matches[0]['location'][0]
            predicted_lon = matches[0]['location'][1]
        
        # Calculate confidence based on top matches consistency
        confidence = self._calculate_matching_confidence(matches)
        
        return LocalizationResult(
            predicted_latitude=predicted_lat,
            predicted_longitude=predicted_lon,
            confidence=confidence,
            top_matches=matches
        )
    
    def _direct_localization(self, query_image: Union[str, np.ndarray, Image.Image]) -> LocalizationResult:
        """Direct geolocation prediction using neural network."""
        if not self.geolocation_predictor:
            return LocalizationResult(
                predicted_latitude=0.0,
                predicted_longitude=0.0,
                confidence=0.0,
                top_matches=[]
            )
        
        # Extract features
        features = self.feature_extractor.get_combined_feature_vector(query_image)
        features_tensor = torch.tensor(features).unsqueeze(0).float()
        
        # Predict coordinates
        with torch.no_grad():
            predictions = self.geolocation_predictor(features_tensor)
            lat, lon = predictions[0].tolist()
        
        # Calculate confidence (this would need to be learned during training)
        # For now, use a simple heuristic
        confidence = 0.7  # Placeholder
        
        return LocalizationResult(
            predicted_latitude=lat,
            predicted_longitude=lon,
            confidence=confidence,
            top_matches=[]
        )
    
    def _cluster_based_localization(self, matches: List[Dict]) -> Dict:
        """Analyze spatial clustering of matches."""
        if len(matches) < 3:
            return {
                'predicted_location': None,
                'confidence': 0.0,
                'cluster_info': {'num_clusters': 0}
            }
        
        # Extract coordinates and similarities
        coordinates = np.array([match['location'] for match in matches])
        similarities = np.array([match['confidence'] for match in matches])
        
        # Perform DBSCAN clustering in geographic space
        # Convert to approximate meters for clustering
        coords_rad = np.radians(coordinates)
        coords_cart = np.column_stack([
            coords_rad[:, 0] * 6371000,  # lat to meters
            coords_rad[:, 1] * 6371000 * np.cos(coords_rad[:, 0])  # lon to meters
        ])
        
        # Cluster with 100m epsilon
        clustering = DBSCAN(eps=100, min_samples=2).fit(coords_cart)
        labels = clustering.labels_
        
        # Analyze clusters
        unique_labels = set(labels)
        clusters = []
        
        for label in unique_labels:
            if label == -1:  # Noise points
                continue
            
            cluster_mask = labels == label
            cluster_coords = coordinates[cluster_mask]
            cluster_similarities = similarities[cluster_mask]
            
            # Weighted centroid
            weights = cluster_similarities / np.sum(cluster_similarities)
            centroid_lat = np.sum(cluster_coords[:, 0] * weights)
            centroid_lon = np.sum(cluster_coords[:, 1] * weights)
            
            cluster_info = {
                'centroid': (centroid_lat, centroid_lon),
                'size': len(cluster_coords),
                'avg_similarity': np.mean(cluster_similarities),
                'weight': np.sum(cluster_similarities)
            }
            clusters.append(cluster_info)
        
        # Select best cluster
        if clusters:
            best_cluster = max(clusters, key=lambda x: x['weight'])
            predicted_location = best_cluster['centroid']
            
            # Confidence based on cluster quality
            confidence = min(1.0, best_cluster['weight'] / len(matches))
        else:
            predicted_location = None
            confidence = 0.0
        
        return {
            'predicted_location': predicted_location,
            'confidence': confidence,
            'cluster_info': {
                'num_clusters': len(clusters),
                'clusters': clusters,
                'total_matches': len(matches)
            }
        }
    
    def _combine_predictions(self, matching_result: LocalizationResult,
                           direct_result: Optional[LocalizationResult],
                           cluster_result: Dict) -> Tuple[float, float, float]:
        """Combine predictions from different methods."""
        predictions = []
        weights = []
        
        # Matching-based prediction
        predictions.append((matching_result.predicted_latitude, matching_result.predicted_longitude))
        weights.append(matching_result.confidence * 0.4)
        
        # Direct prediction
        if direct_result and direct_result.confidence > 0.1:
            predictions.append((direct_result.predicted_latitude, direct_result.predicted_longitude))
            weights.append(direct_result.confidence * 0.3)
        
        # Cluster-based prediction
        if cluster_result['predicted_location'] and cluster_result['confidence'] > 0.1:
            predictions.append(cluster_result['predicted_location'])
            weights.append(cluster_result['confidence'] * (0.3 if direct_result else 0.6))
        
        if not predictions:
            return 0.0, 0.0, 0.0
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            return predictions[0][0], predictions[0][1], 0.0
        
        weighted_lat = sum(pred[0] * weight for pred, weight in zip(predictions, weights)) / total_weight
        weighted_lon = sum(pred[1] * weight for pred, weight in zip(predictions, weights)) / total_weight
        
        # Overall confidence
        confidence = min(1.0, total_weight)
        
        return weighted_lat, weighted_lon, confidence
    
    def _calculate_matching_confidence(self, matches: List[Dict]) -> float:
        """Calculate confidence from matching results."""
        if not matches:
            return 0.0
        
        # Base confidence from top match
        base_confidence = matches[0]['confidence']
        
        # Consistency bonus: check if top matches are geographically consistent
        if len(matches) >= 3:
            top_locations = [match['location'] for match in matches[:3]]
            distances = []
            
            for i in range(len(top_locations)):
                for j in range(i + 1, len(top_locations)):
                    dist = geodesic(top_locations[i], top_locations[j]).kilometers
                    distances.append(dist)
            
            avg_distance = np.mean(distances)
            consistency_bonus = max(0, 1.0 - avg_distance / 10.0)  # Bonus if matches are within 10km
            
            base_confidence += consistency_bonus * 0.2
        
        return min(1.0, base_confidence)

class LocationValidator:
    """Validate and refine localization results."""
    
    def __init__(self):
        """Initialize location validator."""
        pass
    
    def validate_location(self, result: LocalizationResult, 
                         constraints: Optional[Dict] = None) -> LocalizationResult:
        """
        Validate and potentially refine localization result.
        
        Args:
            result: Initial localization result
            constraints: Optional constraints (e.g., country bounds)
            
        Returns:
            Validated/refined result
        """
        # Check geographic bounds
        if constraints:
            result = self._apply_geographic_constraints(result, constraints)
        
        # Check consistency with nearby matches
        result = self._check_spatial_consistency(result)
        
        # Adjust confidence based on validation
        result = self._adjust_confidence(result)
        
        return result
    
    def _apply_geographic_constraints(self, result: LocalizationResult, 
                                    constraints: Dict) -> LocalizationResult:
        """Apply geographic constraints to results."""
        lat, lon = result.predicted_latitude, result.predicted_longitude
        
        # Check bounds
        if 'bounds' in constraints:
            bounds = constraints['bounds']
            if not (bounds['min_lat'] <= lat <= bounds['max_lat'] and
                   bounds['min_lon'] <= lon <= bounds['max_lon']):
                # Reduce confidence for out-of-bounds predictions
                result.confidence *= 0.5
        
        # Check country constraints
        if 'allowed_countries' in constraints:
            # This would require reverse geocoding
            # For now, just apply a general penalty for low-confidence results
            if result.confidence < 0.3:
                result.confidence *= 0.8
        
        return result
    
    def _check_spatial_consistency(self, result: LocalizationResult) -> LocalizationResult:
        """Check spatial consistency of prediction with top matches."""
        if not result.top_matches:
            return result
        
        predicted_location = (result.predicted_latitude, result.predicted_longitude)
        
        # Calculate distances to top matches
        distances = []
        for match in result.top_matches[:5]:  # Check top 5 matches
            distance = geodesic(predicted_location, match['location']).kilometers
            distances.append(distance)
        
        avg_distance = np.mean(distances)
        
        # Penalize if prediction is far from top matches
        if avg_distance > 5.0:  # More than 5km from top matches
            distance_penalty = min(0.5, avg_distance / 20.0)
            result.confidence *= (1.0 - distance_penalty)
        
        return result
    
    def _adjust_confidence(self, result: LocalizationResult) -> LocalizationResult:
        """Final confidence adjustment based on various factors."""
        # Minimum confidence threshold
        result.confidence = max(0.0, result.confidence)
        
        # Maximum confidence cap
        result.confidence = min(1.0, result.confidence)
        
        # If very few matches, reduce confidence
        if len(result.top_matches) < 3:
            result.confidence *= 0.8
        
        return result

class GroundLevelLocalizer:
    """Main interface for ground-level visual localization."""
    
    def __init__(self, database_path: str = "data/streetview.db",
                 feature_index_path: str = "data/embeddings/features.index",
                 model_path: Optional[str] = None):
        """
        Initialize ground-level localizer.
        
        Args:
            database_path: Path to street-view database
            feature_index_path: Path to feature index
            model_path: Optional path to trained geolocation model
        """
        # Initialize components
        self.feature_extractor = MultiModalFeatureExtractor(primary_model="clip")
        self.database = StreetViewDatabase(database_path, feature_index_path)
        self.localizer = EnsembleLocalizer(self.database, self.feature_extractor, model_path)
        self.validator = LocationValidator()
        
        logger.info("Ground-level localizer initialized")
        stats = self.database.get_statistics()
        logger.info(f"Database contains {stats['total_images']} images")
    
    def localize_image(self, image: Union[str, np.ndarray, Image.Image],
                      method: str = "ensemble", constraints: Optional[Dict] = None) -> LocalizationResult:
        """
        Localize an image to geographic coordinates.
        
        Args:
            image: Input image (file path, numpy array, or PIL Image)
            method: Localization method ('ensemble', 'matching', 'direct')
            constraints: Optional geographic constraints
            
        Returns:
            LocalizationResult with predicted location and confidence
        """
        # Perform localization
        result = self.localizer.localize(image, method=method)
        
        # Validate result
        if constraints:
            result = self.validator.validate_location(result, constraints)
        
        return result
    
    def get_database_stats(self) -> Dict:
        """Get database statistics."""
        return self.database.get_statistics()