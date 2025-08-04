"""
Streamlit web interface for ground-level visual localization.
Provides an easy-to-use web interface for testing the localization system.
"""

import streamlit as st
import requests
import json
import folium
from streamlit_folium import st_folium
import plotly.express as px
import plotly.graph_objects as go
from PIL import Image
import io
import time
import pandas as pd
from typing import Optional, Dict, List

# Page configuration
st.set_page_config(
    page_title="Ground-Level Visual Localization",
    page_icon="📍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Constants
API_BASE_URL = "http://localhost:8000"
DEFAULT_LOCATION = [40.7589, -73.9851]  # NYC

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .success-message {
        background-color: #d4edda;
        color: #155724;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .error-message {
        background-color: #f8d7da;
        color: #721c24;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def get_api_status():
    """Get API health status."""
    try:
        response = requests.get(f"{API_BASE_URL}/health", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

@st.cache_data
def get_database_stats():
    """Get database statistics."""
    try:
        response = requests.get(f"{API_BASE_URL}/stats", timeout=10)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

@st.cache_data
def get_localization_methods():
    """Get available localization methods."""
    try:
        response = requests.get(f"{API_BASE_URL}/methods", timeout=5)
        if response.status_code == 200:
            return response.json()
        return None
    except:
        return None

def localize_image(image_file, method: str, include_debug: bool = False) -> Optional[Dict]:
    """Send image to API for localization."""
    try:
        files = {"file": image_file}
        data = {
            "method": method,
            "include_debug": include_debug
        }
        
        response = requests.post(
            f"{API_BASE_URL}/localize",
            files=files,
            data=data,
            timeout=60
        )
        
        if response.status_code == 200:
            return response.json()
        else:
            st.error(f"API Error: {response.status_code} - {response.text}")
            return None
            
    except requests.exceptions.Timeout:
        st.error("Request timed out. The image may be too large or the server is busy.")
        return None
    except Exception as e:
        st.error(f"Error connecting to API: {e}")
        return None

def create_result_map(lat: float, lon: float, confidence: float, top_matches: Optional[List] = None):
    """Create a folium map showing the localization result."""
    # Create base map
    m = folium.Map(location=[lat, lon], zoom_start=15)
    
    # Add main prediction marker
    folium.Marker(
        [lat, lon],
        popup=f"Predicted Location<br>Confidence: {confidence:.3f}",
        icon=folium.Icon(color='red', icon='star'),
        tooltip=f"Predicted: {lat:.6f}, {lon:.6f}"
    ).add_to(m)
    
    # Add top matches if available
    if top_matches:
        for i, match in enumerate(top_matches[:10]):
            folium.CircleMarker(
                [match['latitude'], match['longitude']],
                radius=8,
                popup=f"Match #{i+1}<br>Confidence: {match['confidence']:.3f}<br>Source: {match['source']}",
                color='blue',
                fillColor='lightblue',
                fillOpacity=0.7,
                tooltip=f"Match #{i+1}: {match['confidence']:.3f}"
            ).add_to(m)
    
    return m

def display_confidence_chart(confidence: float):
    """Display confidence as a gauge chart."""
    fig = go.Figure(go.Indicator(
        mode = "gauge+number",
        value = confidence * 100,
        domain = {'x': [0, 1], 'y': [0, 1]},
        title = {'text': "Confidence (%)"},
        gauge = {
            'axis': {'range': [None, 100]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 80], 'color': "gray"},
                {'range': [80, 100], 'color': "lightgreen"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig

def main():
    """Main Streamlit application."""
    
    # Header
    st.markdown('<h1 class="main-header">📍 Ground-Level Visual Localization</h1>', unsafe_allow_html=True)
    st.markdown("**AI-powered geolocation from street-level images**")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Configuration")
        
        # API Status
        st.subheader("API Status")
        api_status = get_api_status()
        if api_status:
            if api_status['status'] == 'healthy':
                st.success("✅ API is healthy")
            else:
                st.warning("⚠️ API is degraded")
            
            st.write(f"**Database:** {api_status['database_status']}")
            st.write(f"**Model:** {api_status['model_status']}")
            st.write(f"**Uptime:** {api_status['uptime_seconds']:.1f}s")
        else:
            st.error("❌ API is unavailable")
            st.stop()
        
        # Method selection
        st.subheader("Localization Method")
        methods_data = get_localization_methods()
        if methods_data:
            method_names = [m['name'] for m in methods_data['methods']]
            selected_method = st.selectbox(
                "Choose method:",
                method_names,
                index=0,
                help="Select the localization algorithm to use"
            )
            
            # Show method info
            method_info = next(m for m in methods_data['methods'] if m['name'] == selected_method)
            st.info(f"**{method_info['description']}**\n\n"
                   f"Accuracy: {method_info['accuracy']}\n\n"
                   f"Speed: {method_info['speed']}")
        else:
            selected_method = "ensemble"
            st.warning("Could not load method information")
        
        # Debug options
        st.subheader("Debug Options")
        include_debug = st.checkbox("Include debug information", value=False)
        show_matches = st.checkbox("Show top matches on map", value=True)
    
    # Main content
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.header("📤 Upload Image")
        
        # File uploader
        uploaded_file = st.file_uploader(
            "Choose a street-level image...",
            type=['jpg', 'jpeg', 'png'],
            help="Upload a photo taken at street level for geolocation"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_column_width=True)
            
            # Image info
            st.write(f"**Size:** {image.size[0]} x {image.size[1]} pixels")
            st.write(f"**Format:** {image.format}")
            st.write(f"**File size:** {len(uploaded_file.getvalue()) / 1024:.1f} KB")
            
            # Localize button
            if st.button("🎯 Localize Image", type="primary"):
                with st.spinner("Analyzing image and matching against database..."):
                    # Reset file pointer
                    uploaded_file.seek(0)
                    
                    # Call API
                    result = localize_image(uploaded_file, selected_method, include_debug)
                    
                    if result and result['success']:
                        st.session_state['localization_result'] = result
                        st.success("✅ Localization successful!")
                    else:
                        if result:
                            st.error(f"❌ Localization failed: {result.get('error_message', 'Unknown error')}")
                        else:
                            st.error("❌ Failed to get response from API")
    
    with col2:
        st.header("📍 Results")
        
        # Display results if available
        if 'localization_result' in st.session_state:
            result = st.session_state['localization_result']
            
            # Main metrics
            col2a, col2b, col2c = st.columns(3)
            
            with col2a:
                st.metric(
                    "Latitude",
                    f"{result['latitude']:.6f}°",
                    help="Predicted latitude coordinate"
                )
            
            with col2b:
                st.metric(
                    "Longitude", 
                    f"{result['longitude']:.6f}°",
                    help="Predicted longitude coordinate"
                )
            
            with col2c:
                st.metric(
                    "Processing Time",
                    f"{result['processing_time']:.2f}s",
                    help="Time taken to process the image"
                )
            
            # Confidence gauge
            st.subheader("Confidence Score")
            confidence_fig = display_confidence_chart(result['confidence'])
            st.plotly_chart(confidence_fig, use_container_width=True)
            
            # Map
            st.subheader("📍 Location Map")
            top_matches = result.get('top_matches', []) if show_matches else None
            result_map = create_result_map(
                result['latitude'], 
                result['longitude'], 
                result['confidence'],
                top_matches
            )
            st_folium(result_map, width=700, height=400)
            
            # Copy coordinates button
            coordinates_text = f"{result['latitude']:.6f}, {result['longitude']:.6f}"
            st.text_input("Coordinates (copy to clipboard):", value=coordinates_text)
            
            # Debug information
            if include_debug and result.get('top_matches'):
                st.subheader("🔍 Debug Information")
                
                # Top matches table
                matches_data = []
                for i, match in enumerate(result['top_matches'][:10]):
                    matches_data.append({
                        'Rank': i + 1,
                        'Latitude': f"{match['latitude']:.6f}",
                        'Longitude': f"{match['longitude']:.6f}",
                        'Confidence': f"{match['confidence']:.3f}",
                        'Source': match['source'],
                        'Distance (km)': f"{match.get('distance_km', 'N/A')}"
                    })
                
                df = pd.DataFrame(matches_data)
                st.dataframe(df, use_container_width=True)
                
                # Method breakdown
                if result.get('method_breakdown'):
                    st.subheader("Method Breakdown")
                    breakdown = result['method_breakdown']
                    
                    breakdown_data = {
                        'Method': ['Matching', 'Direct', 'Cluster'],
                        'Weight': [
                            breakdown.get('matching_weight', 0),
                            breakdown.get('direct_weight', 0),
                            breakdown.get('cluster_weight', 0)
                        ],
                        'Confidence': [
                            breakdown.get('matching_confidence', 0),
                            breakdown.get('direct_confidence', 0),
                            breakdown.get('cluster_confidence', 0)
                        ]
                    }
                    
                    breakdown_df = pd.DataFrame(breakdown_data)
                    st.dataframe(breakdown_df, use_container_width=True)
        
        else:
            st.info("👆 Upload an image to see localization results here")
    
    # Database statistics
    st.header("📊 Database Statistics")
    db_stats = get_database_stats()
    
    if db_stats:
        col3a, col3b, col3c, col3d = st.columns(4)
        
        with col3a:
            st.metric("Total Images", f"{db_stats['total_images']:,}")
        
        with col3b:
            st.metric("Feature Index Size", f"{db_stats['feature_index_size']:,}")
        
        with col3c:
            if db_stats['geographic_bounds']:
                bounds = db_stats['geographic_bounds']
                lat_range = bounds['max_lat'] - bounds['min_lat']
                st.metric("Latitude Range", f"{lat_range:.2f}°")
            else:
                st.metric("Latitude Range", "N/A")
        
        with col3d:
            if db_stats['geographic_bounds']:
                bounds = db_stats['geographic_bounds']
                lon_range = bounds['max_lon'] - bounds['min_lon']
                st.metric("Longitude Range", f"{lon_range:.2f}°")
            else:
                st.metric("Longitude Range", "N/A")
        
        # Source breakdown chart
        if db_stats['by_source']:
            st.subheader("Images by Source")
            source_data = pd.DataFrame(
                list(db_stats['by_source'].items()),
                columns=['Source', 'Count']
            )
            
            fig = px.pie(
                source_data, 
                values='Count', 
                names='Source',
                title="Distribution of Images by Source"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    else:
        st.warning("Could not load database statistics")
    
    # Footer
    st.markdown("---")
    st.markdown(
        "Built with ❤️ using Streamlit, FastAPI, and PyTorch. "
        "For API documentation, visit `/docs`"
    )

if __name__ == "__main__":
    main()