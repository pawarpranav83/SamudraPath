import pandas as pd
import numpy as np
import math

def read_csv(file_path):
    """Reads the CSV file containing lon, lat coordinates."""
    return pd.read_csv(file_path)

def haversine(lat1, lon1, lat2, lon2):
    """Calculates the Haversine distance between two lat/lon points."""
    R = 6371  # Earth radius in kilometers
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    return R * c  # Distance in kilometers

def rdp_simplify(points, epsilon=0.00000001):
    """Ramer-Douglas-Peucker Algorithm for path simplification."""
    if len(points) < 2:
        return points

    # Always keep the first and last points
    start, end = points[0], points[-1]
    
    # Find the point with the maximum distance
    max_distance = 0
    index = 0
    for i in range(1, len(points) - 1):
        distance = perpendicular_distance(points[i], start, end)
        if distance > max_distance:
            max_distance = distance
            index = i
    
    # If the max distance is larger than epsilon, recursively simplify the path
    if max_distance > epsilon:
        left = rdp_simplify(points[:index + 1], epsilon)
        right = rdp_simplify(points[index:], epsilon)
        return left[:-1] + right
    else:
        return [start, end]

def perpendicular_distance(point, start, end):
    """Calculate the perpendicular distance from a point to a line segment."""
    x1, y1 = start
    x2, y2 = end
    x0, y0 = point
    
    if (x1, y1) == (x2, y2):  # Start and end points are the same
        return haversine(y1, x1, y0, x0)
    
    num = abs((y2 - y1) * x0 - (x2 - x1) * y0 + x2 * y1 - y2 * x1)
    denom = math.sqrt((y2 - y1)**2 + (x2 - x1)**2)
    
    return num / denom if denom != 0 else 0  # Avoid division by zero

def smooth_route(coordinates, window_size=2):
    """Smooths the route using a moving average filter."""
    smoothed_coords = []
    for i in range(len(coordinates)):
        start_idx = max(0, i - window_size)
        end_idx = min(len(coordinates), i + window_size + 1)
        window = coordinates[start_idx:end_idx]
        
        avg_lon = np.mean([coord[0] for coord in window])
        avg_lat = np.mean([coord[1] for coord in window])
        
        smoothed_coords.append((avg_lon, avg_lat))
    
    return smoothed_coords

def process_route(file_path, epsilon=0.0000001, window_size=2):
    """Process the route by reading the CSV, simplifying, and smoothing."""
    # Step 1: Read coordinates from the CSV file
    data = read_csv(file_path)
    
    # Ensure that data contains 'Longitude' and 'Latitude' columns
    coordinates = list(zip(data['Longitude'], data['Latitude']))
    
    # Step 2: Simplify the path using the RDP algorithm
    simplified_route = rdp_simplify(coordinates, epsilon)
    
    # Step 3: Smooth the route with a moving average filter
    smoothed_route = smooth_route(simplified_route, window_size)

    # Ensure that the first and last point from the original input data are included
    if smoothed_route[0] != coordinates[0]:
        smoothed_route.insert(0, coordinates[0])  # Add the first point if missing
    
    if smoothed_route[-1] != coordinates[-1]:
        smoothed_route.append(coordinates[-1])  # Add the last point if missing
    
    return smoothed_route

def save_to_csv(coordinates, output_file):
    """Saves the smoothed coordinates to a new CSV file."""
    df = pd.DataFrame(coordinates, columns=['Longitude', 'Latitude'])
    df.to_csv(output_file, index=False)
    print(f"Smoothed route saved to {output_file}")
