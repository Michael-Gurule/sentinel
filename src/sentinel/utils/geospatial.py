"""
Geospatial utility functions
"""

import numpy as np
from typing import Tuple

def haversine_distance(
    lat1: float,
    lon1: float,
    lat2: float,
    lon2: float
) -> float:
    """
    Calculate great circle distance between two points
    
    Args:
        lat1, lon1: First point (degrees)
        lat2, lon2: Second point (degrees)
        
    Returns:
        Distance in kilometers
    """
    R = 6371  # Earth radius in km
    
    lat1_rad, lon1_rad = np.radians(lat1), np.radians(lon1)
    lat2_rad, lon2_rad = np.radians(lat2), np.radians(lon2)
    
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    
    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))
    
    return R * c


def add_gaussian_noise_to_position(
    lat: float,
    lon: float,
    std_km: float
) -> Tuple[float, float]:
    """
    Add Gaussian noise to geographic position
    
    Useful for simulating sensor position uncertainty
    
    Args:
        lat, lon: True position (degrees)
        std_km: Standard deviation in kilometers
        
    Returns:
        Noisy position (lat, lon) in degrees
    """
    # Convert km to degrees (approximate)
    std_deg = std_km / 111.0  # 1 degree ≈ 111 km at equator
    
    lat_noise = np.random.normal(0, std_deg)
    lon_noise = np.random.normal(0, std_deg / np.cos(np.radians(lat)))
    
    return lat + lat_noise, lon + lon_noise


def random_position_in_radius(
    center_lat: float,
    center_lon: float,
    radius_km: float
) -> Tuple[float, float]:
    """
    Generate random position within a radius of center point
    
    Args:
        center_lat, center_lon: Center point (degrees)
        radius_km: Maximum radius in kilometers
        
    Returns:
        Random position (lat, lon) in degrees
    """
    # Random distance and bearing
    distance = np.sqrt(np.random.uniform(0, 1)) * radius_km
    bearing = np.random.uniform(0, 2*np.pi)
    
    # Convert to lat/lon offset
    R = 6371
    lat_offset = (distance / R) * np.cos(bearing) * (180 / np.pi)
    lon_offset = (distance / R) * np.sin(bearing) * (180 / np.pi) / np.cos(np.radians(center_lat))
    
    return center_lat + lat_offset, center_lon + lon_offset