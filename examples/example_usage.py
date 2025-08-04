#!/usr/bin/env python3
"""
Example usage of the ground-level visual localization system.
Demonstrates various ways to use the localization engine and API.
"""

import sys
import os
import time
import requests
import json
from PIL import Image
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def example_direct_api():
    """Example: Using the localization engine directly."""
    print("=== Direct API Usage Example ===")
    
    try:
        from core.localization_engine import GroundLevelLocalizer
        
        # Initialize localizer
        print("Initializing localizer...")
        localizer = GroundLevelLocalizer()
        
        # Check database status
        stats = localizer.get_database_stats()
        print(f"Database contains {stats['total_images']} images")
        
        if stats['total_images'] == 0:
            print("⚠️  Database is empty. Add images first using the database builder.")
            return
        
        # Create a dummy image for testing
        dummy_image = Image.new('RGB', (640, 480), color='blue')
        
        # Perform localization
        print("Performing localization...")
        result = localizer.localize_image(dummy_image, method="ensemble")
        
        print(f"Predicted location: {result.predicted_latitude:.6f}, {result.predicted_longitude:.6f}")
        print(f"Confidence: {result.confidence:.3f}")
        print(f"Processing time: {result.processing_time:.2f}s")
        
        if result.top_matches:
            print(f"Found {len(result.top_matches)} matches")
            
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have run the API server at least once to initialize the database.")

def example_rest_api():
    """Example: Using the REST API."""
    print("\n=== REST API Usage Example ===")
    
    api_url = "http://localhost:8000"
    
    # Check API health
    try:
        response = requests.get(f"{api_url}/health", timeout=5)
        if response.status_code == 200:
            health = response.json()
            print(f"✅ API is {health['status']}")
            print(f"   Database: {health['database_status']}")
            print(f"   Model: {health['model_status']}")
        else:
            print("❌ API is not healthy")
            return
    except requests.exceptions.RequestException:
        print("❌ API is not running. Start it with: ./run_system.sh --api")
        return
    
    # Get database stats
    try:
        response = requests.get(f"{api_url}/stats")
        if response.status_code == 200:
            stats = response.json()
            print(f"📊 Database: {stats['total_images']} images, {stats['feature_index_size']} feature vectors")
        else:
            print("Failed to get database stats")
    except Exception as e:
        print(f"Error getting stats: {e}")
    
    # Create a test image
    test_image = Image.new('RGB', (640, 480), color=(100, 150, 200))
    test_image_path = "/tmp/test_image.jpg"
    test_image.save(test_image_path)
    
    try:
        # Test localization
        print("Testing image localization...")
        
        with open(test_image_path, 'rb') as f:
            files = {'file': f}
            data = {
                'method': 'ensemble',
                'include_debug': True
            }
            
            response = requests.post(f"{api_url}/localize", files=files, data=data)
        
        if response.status_code == 200:
            result = response.json()
            if result['success']:
                print(f"✅ Localization successful!")
                print(f"   Location: {result['latitude']:.6f}, {result['longitude']:.6f}")
                print(f"   Confidence: {result['confidence']:.3f}")
                print(f"   Processing time: {result['processing_time']:.2f}s")
                
                if result.get('top_matches'):
                    print(f"   Top matches: {len(result['top_matches'])}")
            else:
                print(f"❌ Localization failed: {result.get('error_message')}")
        else:
            print(f"❌ API request failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error testing API: {e}")
    finally:
        # Cleanup
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def example_batch_processing():
    """Example: Batch processing multiple images."""
    print("\n=== Batch Processing Example ===")
    
    api_url = "http://localhost:8000"
    
    # Create multiple test images
    test_images = []
    for i in range(3):
        # Create images with different colors
        color = (50 + i * 50, 100 + i * 30, 150 + i * 20)
        image = Image.new('RGB', (640, 480), color=color)
        image_path = f"/tmp/test_batch_{i}.jpg"
        image.save(image_path)
        test_images.append(image_path)
    
    try:
        # Prepare files for batch upload
        files = []
        for i, image_path in enumerate(test_images):
            files.append(('files', (f'image_{i}.jpg', open(image_path, 'rb'), 'image/jpeg')))
        
        # Submit batch job
        print("Submitting batch localization job...")
        response = requests.post(f"{api_url}/localize/batch", files=files)
        
        # Close file handles
        for _, (_, file_handle, _) in files:
            file_handle.close()
        
        if response.status_code == 200:
            batch_info = response.json()
            print(f"✅ Batch job submitted: {batch_info['batch_id']}")
            print(f"   Processing {batch_info['total_images']} images")
            print("   Note: This is a demo - in production, you'd poll for results")
        else:
            print(f"❌ Batch submission failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error in batch processing: {e}")
    finally:
        # Cleanup
        for image_path in test_images:
            if os.path.exists(image_path):
                os.remove(image_path)

def example_database_operations():
    """Example: Database operations."""
    print("\n=== Database Operations Example ===")
    
    try:
        from core.streetview_database import StreetViewDatabase, StreetViewImage
        from models.feature_extractor import MultiModalFeatureExtractor
        from datetime import datetime
        
        # Initialize components
        print("Initializing database...")
        database = StreetViewDatabase("data/example.db", "data/example.index")
        feature_extractor = MultiModalFeatureExtractor()
        
        # Create example street-view image metadata
        example_image = StreetViewImage(
            image_id="example_001",
            latitude=40.7589,
            longitude=-73.9851,
            heading=90.0,
            pitch=0.0,
            fov=90.0,
            image_path="data/example_street.jpg",
            source="manual",
            timestamp=datetime.now().isoformat(),
            address="Central Park, New York, NY",
            city="New York",
            country="USA"
        )
        
        # Create a dummy feature vector
        dummy_features = np.random.random(512).astype(np.float32)
        dummy_features = dummy_features / np.linalg.norm(dummy_features)
        
        # Add to database
        print("Adding example image to database...")
        database.add_image(example_image, dummy_features)
        
        # Search by location
        print("Searching by location...")
        nearby_images = database.search_by_location(40.7589, -73.9851, radius_km=1.0)
        print(f"Found {len(nearby_images)} images within 1km")
        
        # Search by features
        print("Searching by features...")
        query_features = np.random.random(512).astype(np.float32)
        query_features = query_features / np.linalg.norm(query_features)
        
        similar_images = database.search_by_features(query_features, k=5)
        print(f"Found {len(similar_images)} similar images")
        
        # Get statistics
        stats = database.get_statistics()
        print(f"Database statistics: {stats}")
        
    except Exception as e:
        print(f"Error in database operations: {e}")

def example_feature_extraction():
    """Example: Feature extraction from images."""
    print("\n=== Feature Extraction Example ===")
    
    try:
        from models.feature_extractor import MultiModalFeatureExtractor
        
        # Initialize feature extractor
        print("Initializing feature extractor...")
        extractor = MultiModalFeatureExtractor(primary_model="clip")
        
        # Create test images
        test_images = [
            Image.new('RGB', (640, 480), color='red'),
            Image.new('RGB', (640, 480), color='green'),
            Image.new('RGB', (640, 480), color='blue')
        ]
        
        print("Extracting features from test images...")
        
        for i, image in enumerate(test_images):
            # Extract comprehensive features
            features = extractor.extract_comprehensive_features(image)
            
            print(f"\nImage {i+1}:")
            for feature_type, feature_vector in features.items():
                if len(feature_vector) > 0:
                    print(f"  {feature_type}: {len(feature_vector)} dimensions")
                    print(f"    Mean: {np.mean(feature_vector):.4f}")
                    print(f"    Std:  {np.std(feature_vector):.4f}")
            
            # Get combined feature vector
            combined = extractor.get_combined_feature_vector(image)
            print(f"  Combined: {len(combined)} dimensions")
        
        print("\n✅ Feature extraction completed successfully")
        
    except Exception as e:
        print(f"Error in feature extraction: {e}")

def example_api_methods():
    """Example: Testing different API methods."""
    print("\n=== API Methods Comparison ===")
    
    api_url = "http://localhost:8000"
    
    # Check if API is running
    try:
        requests.get(f"{api_url}/health", timeout=5)
    except:
        print("❌ API is not running. Start it with: ./run_system.sh --api")
        return
    
    # Get available methods
    try:
        response = requests.get(f"{api_url}/methods")
        if response.status_code == 200:
            methods_info = response.json()
            print("Available localization methods:")
            for method in methods_info['methods']:
                print(f"  {method['name']}: {method['description']}")
                print(f"    Accuracy: {method['accuracy']}, Speed: {method['speed']}")
        else:
            print("Failed to get methods information")
            return
    except Exception as e:
        print(f"Error getting methods: {e}")
        return
    
    # Create test image
    test_image = Image.new('RGB', (640, 480), color=(128, 128, 128))
    test_image_path = "/tmp/method_test.jpg"
    test_image.save(test_image_path)
    
    methods = ['ensemble', 'matching', 'direct']
    
    try:
        print("\nTesting different methods:")
        
        for method in methods:
            print(f"\n--- Testing {method} method ---")
            
            start_time = time.time()
            
            with open(test_image_path, 'rb') as f:
                files = {'file': f}
                data = {'method': method}
                
                response = requests.post(f"{api_url}/localize", files=files, data=data)
            
            request_time = time.time() - start_time
            
            if response.status_code == 200:
                result = response.json()
                if result['success']:
                    print(f"  ✅ Success: {result['latitude']:.6f}, {result['longitude']:.6f}")
                    print(f"  Confidence: {result['confidence']:.3f}")
                    print(f"  Server time: {result['processing_time']:.2f}s")
                    print(f"  Total time: {request_time:.2f}s")
                else:
                    print(f"  ❌ Failed: {result.get('error_message')}")
            else:
                print(f"  ❌ HTTP Error: {response.status_code}")
                
    except Exception as e:
        print(f"Error testing methods: {e}")
    finally:
        if os.path.exists(test_image_path):
            os.remove(test_image_path)

def main():
    """Run all examples."""
    print("🌍 Ground-Level Visual Localization - Examples")
    print("=" * 50)
    
    # Run examples
    example_feature_extraction()
    example_database_operations()
    example_direct_api()
    example_rest_api()
    example_batch_processing()
    example_api_methods()
    
    print("\n" + "=" * 50)
    print("✅ All examples completed!")
    print("\nNext steps:")
    print("1. Start the full system: ./run_system.sh --all")
    print("2. Open web interface: http://localhost:8501")
    print("3. View API docs: http://localhost:8000/docs")
    print("4. Add real street-view images to improve accuracy")

if __name__ == "__main__":
    main()