"""
Street-view image database management and indexing system.
Handles storage, indexing, and retrieval of street-view imagery with geolocation data.
"""

import os
import json
import numpy as np
import faiss
import h5py
from typing import Dict, List, Tuple, Optional, Union
import logging
import pickle
from datetime import datetime
import requests
from PIL import Image
import io
import hashlib
from dataclasses import dataclass, asdict
from tqdm import tqdm
import sqlite3

logger = logging.getLogger(__name__)

@dataclass
class StreetViewImage:
    """Container for street-view image metadata."""
    image_id: str
    latitude: float
    longitude: float
    heading: float  # Camera direction in degrees
    pitch: float    # Camera tilt
    fov: float      # Field of view
    image_path: str
    source: str     # 'google', 'mapillary', 'manual', etc.
    timestamp: str
    address: Optional[str] = None
    city: Optional[str] = None
    country: Optional[str] = None
    feature_vector: Optional[np.ndarray] = None

class StreetViewDatabase:
    """Manages street-view image database with spatial indexing."""
    
    def __init__(self, db_path: str = "data/streetview.db", 
                 feature_index_path: str = "data/embeddings/features.index"):
        """
        Initialize street-view database.
        
        Args:
            db_path: Path to SQLite database file
            feature_index_path: Path to FAISS feature index
        """
        self.db_path = db_path
        self.feature_index_path = feature_index_path
        self.feature_dim = None
        self.faiss_index = None
        
        # Initialize database
        self._init_database()
        
        # Load or create feature index
        self._load_feature_index()
    
    def _init_database(self):
        """Initialize SQLite database for metadata storage."""
        os.makedirs(os.path.dirname(self.db_path), exist_ok=True)
        
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS streetview_images (
                    image_id TEXT PRIMARY KEY,
                    latitude REAL NOT NULL,
                    longitude REAL NOT NULL,
                    heading REAL NOT NULL,
                    pitch REAL NOT NULL,
                    fov REAL NOT NULL,
                    image_path TEXT NOT NULL,
                    source TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    address TEXT,
                    city TEXT,
                    country TEXT
                )
            """)
            
            # Create spatial index
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_location 
                ON streetview_images (latitude, longitude)
            """)
            
            # Create source index
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_source 
                ON streetview_images (source)
            """)
    
    def _load_feature_index(self):
        """Load or create FAISS feature index."""
        os.makedirs(os.path.dirname(self.feature_index_path), exist_ok=True)
        
        if os.path.exists(self.feature_index_path):
            try:
                self.faiss_index = faiss.read_index(self.feature_index_path)
                self.feature_dim = self.faiss_index.d
                logger.info(f"Loaded FAISS index with {self.faiss_index.ntotal} vectors")
            except Exception as e:
                logger.error(f"Failed to load FAISS index: {e}")
                self.faiss_index = None
    
    def add_image(self, image_data: StreetViewImage, feature_vector: Optional[np.ndarray] = None):
        """
        Add a street-view image to the database.
        
        Args:
            image_data: Street-view image metadata
            feature_vector: Optional pre-computed feature vector
        """
        # Add to SQLite database
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO streetview_images 
                (image_id, latitude, longitude, heading, pitch, fov, 
                 image_path, source, timestamp, address, city, country)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                image_data.image_id, image_data.latitude, image_data.longitude,
                image_data.heading, image_data.pitch, image_data.fov,
                image_data.image_path, image_data.source, image_data.timestamp,
                image_data.address, image_data.city, image_data.country
            ))
        
        # Add to feature index if provided
        if feature_vector is not None:
            self._add_to_feature_index(image_data.image_id, feature_vector)
    
    def _add_to_feature_index(self, image_id: str, feature_vector: np.ndarray):
        """Add feature vector to FAISS index."""
        if self.faiss_index is None:
            # Create new index
            self.feature_dim = len(feature_vector)
            self.faiss_index = faiss.IndexFlatIP(self.feature_dim)  # Inner product for cosine similarity
        
        # Ensure feature vector is normalized
        feature_vector = feature_vector / np.linalg.norm(feature_vector)
        
        # Add to index
        self.faiss_index.add(feature_vector.reshape(1, -1).astype(np.float32))
        
        # Save index periodically
        if self.faiss_index.ntotal % 1000 == 0:
            self._save_feature_index()
    
    def _save_feature_index(self):
        """Save FAISS index to disk."""
        if self.faiss_index is not None:
            faiss.write_index(self.faiss_index, self.feature_index_path)
    
    def search_by_location(self, latitude: float, longitude: float, 
                          radius_km: float = 1.0, limit: int = 100) -> List[StreetViewImage]:
        """
        Search for images within a geographic radius.
        
        Args:
            latitude: Center latitude
            longitude: Center longitude
            radius_km: Search radius in kilometers
            limit: Maximum number of results
            
        Returns:
            List of StreetViewImage objects
        """
        # Approximate degree conversion (varies by latitude)
        lat_deg_per_km = 1.0 / 111.0
        lon_deg_per_km = 1.0 / (111.0 * np.cos(np.radians(latitude)))
        
        lat_delta = radius_km * lat_deg_per_km
        lon_delta = radius_km * lon_deg_per_km
        
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM streetview_images 
                WHERE latitude BETWEEN ? AND ? 
                AND longitude BETWEEN ? AND ?
                ORDER BY ABS(latitude - ?) + ABS(longitude - ?)
                LIMIT ?
            """, (
                latitude - lat_delta, latitude + lat_delta,
                longitude - lon_delta, longitude + lon_delta,
                latitude, longitude, limit
            ))
            
            results = []
            for row in cursor.fetchall():
                image_data = StreetViewImage(
                    image_id=row[0], latitude=row[1], longitude=row[2],
                    heading=row[3], pitch=row[4], fov=row[5],
                    image_path=row[6], source=row[7], timestamp=row[8],
                    address=row[9], city=row[10], country=row[11]
                )
                results.append(image_data)
        
        return results
    
    def search_by_features(self, query_features: np.ndarray, 
                          k: int = 50) -> List[Tuple[float, int]]:
        """
        Search for similar images using feature vectors.
        
        Args:
            query_features: Query feature vector
            k: Number of results to return
            
        Returns:
            List of (similarity_score, index) tuples
        """
        if self.faiss_index is None or self.faiss_index.ntotal == 0:
            return []
        
        # Normalize query features
        query_features = query_features / np.linalg.norm(query_features)
        query_features = query_features.reshape(1, -1).astype(np.float32)
        
        # Search
        similarities, indices = self.faiss_index.search(query_features, min(k, self.faiss_index.ntotal))
        
        results = []
        for sim, idx in zip(similarities[0], indices[0]):
            if idx != -1:  # Valid index
                results.append((float(sim), int(idx)))
        
        return results
    
    def get_image_by_index(self, index: int) -> Optional[StreetViewImage]:
        """Get image metadata by FAISS index."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                SELECT * FROM streetview_images 
                ORDER BY rowid 
                LIMIT 1 OFFSET ?
            """, (index,))
            
            row = cursor.fetchone()
            if row:
                return StreetViewImage(
                    image_id=row[0], latitude=row[1], longitude=row[2],
                    heading=row[3], pitch=row[4], fov=row[5],
                    image_path=row[6], source=row[7], timestamp=row[8],
                    address=row[9], city=row[10], country=row[11]
                )
        
        return None
    
    def get_statistics(self) -> Dict:
        """Get database statistics."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("SELECT COUNT(*) FROM streetview_images")
            total_images = cursor.fetchone()[0]
            
            cursor = conn.execute("""
                SELECT source, COUNT(*) 
                FROM streetview_images 
                GROUP BY source
            """)
            by_source = dict(cursor.fetchall())
            
            cursor = conn.execute("""
                SELECT MIN(latitude), MAX(latitude), MIN(longitude), MAX(longitude)
                FROM streetview_images
            """)
            bounds = cursor.fetchone()
        
        return {
            'total_images': total_images,
            'by_source': by_source,
            'feature_index_size': self.faiss_index.ntotal if self.faiss_index else 0,
            'geographic_bounds': {
                'min_lat': bounds[0], 'max_lat': bounds[1],
                'min_lon': bounds[2], 'max_lon': bounds[3]
            } if bounds[0] is not None else None
        }

class GoogleStreetViewCollector:
    """Collect street-view images from Google Street View API."""
    
    def __init__(self, api_key: str):
        """
        Initialize Google Street View collector.
        
        Args:
            api_key: Google Maps API key
        """
        self.api_key = api_key
        self.base_url = "https://maps.googleapis.com/maps/api/streetview"
    
    def collect_images_for_area(self, center_lat: float, center_lon: float,
                               radius_km: float = 1.0, grid_spacing_m: int = 100,
                               save_dir: str = "data/streetview") -> List[StreetViewImage]:
        """
        Collect street-view images for a geographic area.
        
        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_km: Collection radius in kilometers
            grid_spacing_m: Spacing between collection points in meters
            save_dir: Directory to save images
            
        Returns:
            List of collected StreetViewImage objects
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Generate grid of collection points
        collection_points = self._generate_grid_points(
            center_lat, center_lon, radius_km, grid_spacing_m
        )
        
        collected_images = []
        
        for i, (lat, lon) in enumerate(tqdm(collection_points, desc="Collecting street view images")):
            # Multiple viewpoints per location
            headings = [0, 90, 180, 270]  # North, East, South, West
            
            for heading in headings:
                try:
                    image_data = self._download_streetview_image(
                        lat, lon, heading, save_dir, f"{i}_{heading}"
                    )
                    if image_data:
                        collected_images.append(image_data)
                except Exception as e:
                    logger.warning(f"Failed to download image at {lat}, {lon}, heading {heading}: {e}")
        
        return collected_images
    
    def _generate_grid_points(self, center_lat: float, center_lon: float,
                             radius_km: float, spacing_m: int) -> List[Tuple[float, float]]:
        """Generate grid of collection points."""
        points = []
        
        # Convert to approximate degrees
        lat_deg_per_m = 1.0 / 111000.0
        lon_deg_per_m = 1.0 / (111000.0 * np.cos(np.radians(center_lat)))
        
        spacing_lat = spacing_m * lat_deg_per_m
        spacing_lon = spacing_m * lon_deg_per_m
        
        radius_deg_lat = radius_km * 1000 * lat_deg_per_m
        radius_deg_lon = radius_km * 1000 * lon_deg_per_m
        
        # Generate grid
        lat = center_lat - radius_deg_lat
        while lat <= center_lat + radius_deg_lat:
            lon = center_lon - radius_deg_lon
            while lon <= center_lon + radius_deg_lon:
                # Check if point is within radius
                distance = np.sqrt((lat - center_lat)**2 + (lon - center_lon)**2)
                if distance <= max(radius_deg_lat, radius_deg_lon):
                    points.append((lat, lon))
                lon += spacing_lon
            lat += spacing_lat
        
        return points
    
    def _download_streetview_image(self, lat: float, lon: float, heading: float,
                                  save_dir: str, image_id: str) -> Optional[StreetViewImage]:
        """Download a single street-view image."""
        params = {
            'size': '640x640',
            'location': f"{lat},{lon}",
            'heading': heading,
            'pitch': 0,
            'fov': 90,
            'key': self.api_key
        }
        
        response = requests.get(self.base_url, params=params)
        
        if response.status_code == 200 and 'image' in response.headers.get('content-type', ''):
            # Save image
            image_filename = f"{image_id}.jpg"
            image_path = os.path.join(save_dir, image_filename)
            
            with open(image_path, 'wb') as f:
                f.write(response.content)
            
            # Create metadata
            image_data = StreetViewImage(
                image_id=image_id,
                latitude=lat,
                longitude=lon,
                heading=heading,
                pitch=0,
                fov=90,
                image_path=image_path,
                source='google',
                timestamp=datetime.now().isoformat()
            )
            
            return image_data
        
        return None

class MapillaryCollector:
    """Collect street-view images from Mapillary API."""
    
    def __init__(self, access_token: str):
        """
        Initialize Mapillary collector.
        
        Args:
            access_token: Mapillary API access token
        """
        self.access_token = access_token
        self.base_url = "https://graph.mapillary.com"
    
    def collect_images_for_area(self, bbox: Tuple[float, float, float, float],
                               save_dir: str = "data/streetview") -> List[StreetViewImage]:
        """
        Collect images from Mapillary for a bounding box.
        
        Args:
            bbox: Bounding box (min_lon, min_lat, max_lon, max_lat)
            save_dir: Directory to save images
            
        Returns:
            List of collected StreetViewImage objects
        """
        os.makedirs(save_dir, exist_ok=True)
        
        # Search for images in bounding box
        search_url = f"{self.base_url}/images"
        params = {
            'access_token': self.access_token,
            'bbox': ','.join(map(str, bbox)),
            'limit': 1000,
            'fields': 'id,computed_geometry,compass_angle,camera_type,captured_at'
        }
        
        response = requests.get(search_url, params=params)
        if response.status_code != 200:
            logger.error(f"Failed to search Mapillary images: {response.status_code}")
            return []
        
        data = response.json()
        images = data.get('data', [])
        
        collected_images = []
        
        for image_info in tqdm(images, desc="Downloading Mapillary images"):
            try:
                image_data = self._download_mapillary_image(image_info, save_dir)
                if image_data:
                    collected_images.append(image_data)
            except Exception as e:
                logger.warning(f"Failed to download Mapillary image {image_info.get('id')}: {e}")
        
        return collected_images
    
    def _download_mapillary_image(self, image_info: Dict, save_dir: str) -> Optional[StreetViewImage]:
        """Download a single Mapillary image."""
        image_id = image_info['id']
        
        # Get image download URL
        download_url = f"{self.base_url}/{image_id}?fields=thumb_1024_url"
        params = {'access_token': self.access_token}
        
        response = requests.get(download_url, params=params)
        if response.status_code != 200:
            return None
        
        download_data = response.json()
        image_url = download_data.get('thumb_1024_url')
        
        if not image_url:
            return None
        
        # Download image
        img_response = requests.get(image_url)
        if img_response.status_code != 200:
            return None
        
        # Save image
        image_filename = f"mapillary_{image_id}.jpg"
        image_path = os.path.join(save_dir, image_filename)
        
        with open(image_path, 'wb') as f:
            f.write(img_response.content)
        
        # Extract metadata
        geometry = image_info.get('computed_geometry', {})
        coordinates = geometry.get('coordinates', [0, 0])
        
        image_data = StreetViewImage(
            image_id=f"mapillary_{image_id}",
            latitude=coordinates[1],
            longitude=coordinates[0],
            heading=image_info.get('compass_angle', 0),
            pitch=0,
            fov=90,
            image_path=image_path,
            source='mapillary',
            timestamp=image_info.get('captured_at', datetime.now().isoformat())
        )
        
        return image_data

class DatabaseBuilder:
    """Build and populate street-view database."""
    
    def __init__(self, database: StreetViewDatabase, feature_extractor):
        """
        Initialize database builder.
        
        Args:
            database: StreetViewDatabase instance
            feature_extractor: Feature extraction model
        """
        self.database = database
        self.feature_extractor = feature_extractor
    
    def build_database_from_directory(self, image_dir: str, 
                                     metadata_file: Optional[str] = None):
        """
        Build database from a directory of images.
        
        Args:
            image_dir: Directory containing street-view images
            metadata_file: Optional JSON file with image metadata
        """
        # Load metadata if available
        metadata = {}
        if metadata_file and os.path.exists(metadata_file):
            with open(metadata_file, 'r') as f:
                metadata = json.load(f)
        
        # Process all images in directory
        image_files = [f for f in os.listdir(image_dir) 
                      if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        for image_file in tqdm(image_files, desc="Processing images"):
            image_path = os.path.join(image_dir, image_file)
            image_id = os.path.splitext(image_file)[0]
            
            try:
                # Extract features
                features = self.feature_extractor.extract_features(image_path)
                
                # Get metadata or use defaults
                img_metadata = metadata.get(image_id, {})
                
                image_data = StreetViewImage(
                    image_id=image_id,
                    latitude=img_metadata.get('latitude', 0.0),
                    longitude=img_metadata.get('longitude', 0.0),
                    heading=img_metadata.get('heading', 0.0),
                    pitch=img_metadata.get('pitch', 0.0),
                    fov=img_metadata.get('fov', 90.0),
                    image_path=image_path,
                    source=img_metadata.get('source', 'manual'),
                    timestamp=img_metadata.get('timestamp', datetime.now().isoformat()),
                    address=img_metadata.get('address'),
                    city=img_metadata.get('city'),
                    country=img_metadata.get('country')
                )
                
                # Add to database
                self.database.add_image(image_data, features)
                
            except Exception as e:
                logger.error(f"Failed to process {image_file}: {e}")
        
        # Save feature index
        self.database._save_feature_index()
        
        logger.info("Database building complete")
        stats = self.database.get_statistics()
        logger.info(f"Database stats: {stats}")

class StreetViewMatcher:
    """Match query images against street-view database."""
    
    def __init__(self, database: StreetViewDatabase, feature_extractor):
        """
        Initialize street-view matcher.
        
        Args:
            database: StreetViewDatabase instance
            feature_extractor: Feature extraction model
        """
        self.database = database
        self.feature_extractor = feature_extractor
    
    def find_location(self, query_image: Union[str, np.ndarray, Image.Image],
                     top_k: int = 10, use_geographic_filter: bool = True,
                     approx_location: Optional[Tuple[float, float]] = None) -> List[Dict]:
        """
        Find location of query image.
        
        Args:
            query_image: Query image
            top_k: Number of top matches to return
            use_geographic_filter: Whether to use geographic filtering
            approx_location: Approximate location for filtering (lat, lon)
            
        Returns:
            List of match results with confidence scores
        """
        # Extract features from query image
        query_features = self.feature_extractor.extract_features(query_image)
        
        # Search by features
        feature_matches = self.database.search_by_features(query_features, k=top_k * 2)
        
        results = []
        for similarity, index in feature_matches:
            image_data = self.database.get_image_by_index(index)
            if image_data:
                result = {
                    'image_data': image_data,
                    'similarity': similarity,
                    'confidence': float(similarity),
                    'location': (image_data.latitude, image_data.longitude)
                }
                results.append(result)
        
        # Geographic filtering if requested
        if use_geographic_filter and approx_location:
            results = self._apply_geographic_filter(results, approx_location)
        
        # Sort by confidence and return top_k
        results.sort(key=lambda x: x['confidence'], reverse=True)
        return results[:top_k]
    
    def _apply_geographic_filter(self, results: List[Dict], 
                                approx_location: Tuple[float, float]) -> List[Dict]:
        """Apply geographic filtering to results."""
        from geopy.distance import geodesic
        
        filtered_results = []
        for result in results:
            distance_km = geodesic(
                approx_location, 
                result['location']
            ).kilometers
            
            # Boost confidence for closer matches
            distance_factor = max(0.1, 1.0 / (1.0 + distance_km / 10.0))
            result['confidence'] *= distance_factor
            result['distance_km'] = distance_km
            
            filtered_results.append(result)
        
        return filtered_results