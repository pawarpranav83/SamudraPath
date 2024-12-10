import numpy as np
import heapq
import matplotlib.pyplot as plt
import rasterio
from rasterio.enums import Resampling
import math
import pandas as pd
import csv
import os
import argparse

# ---------------------- Constants and Parameters ---------------------- #

# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371  

rho = 1.225
# Define global variables at the top of your script
D = None  # Ship displacement (tonnes)
Cp = None  # Wind pressure coefficient
Af = None  # Frontal area of the ship (m²)
Z = None  # Measurement height above sea surface (meters)
TE = None  # Ship's resonant period (seconds)
n_h = None  # Hull efficiency
n_s = None  # Propeller efficiency
n_e = None  # Engine shaft efficiency
csfoc = None  # Specific Fuel Oil Consumption (g/kWh)
a1 = None  # Weight for wind risk
a2 = None  # Weight for wave risk
pirate_risk_factor = None  # Weight for pirate risk
ship_speed_global = None  # Ship's hydrostatic speed in km/h

# Ship parameters (Adjust based on actual ship specifications)
ship_params = {
    'D': D,                # Ship displacement (tonnes)
    'Cp': Cp,                # Wind pressure coefficient
    'Af': Af,                 # Frontal area of the ship (m²)
    'Z': Z,                  # Measurement height above sea surface (meters)
    'TE': TE,                 # Ship's resonant period (seconds)
    'n_h': n_h,               # Hull efficiency
    'n_s': n_s,              # Propeller efficiency
    'n_e': n_e,              # Engine shaft efficiency
    'csfoc': csfoc,             # Specific Fuel Oil Consumption (g/kWh)
    'a1': 1/3,                # Weight for wind risk
    'a2': 1/3,                # Weight for wave risk
    'pirate_risk_factor': 0.3,  # Weight for pirate risk
    'ship_speed': 40           # Ship's hydrostatic speed in km/h
}

# Risk threshold and weighting factor
RISK_THRESHOLD = 0.6
WEIGHTING_FACTOR = 10  

# ---------------------- Utility Functions ---------------------- #

def haversine(lat1, lon1, lat2, lon2):
    """
    Calculate the Haversine distance between two points on the Earth.
    """
    phi1, phi2 = np.radians(lat1), np.radians(lat2)
    delta_phi = np.radians(lat2 - lat1)
    delta_lambda = np.radians(lon2 - lon1)
    
    a = np.sin(delta_phi / 2) ** 2 + \
        np.cos(phi1) * np.cos(phi2) * np.sin(delta_lambda / 2) ** 2
    c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
    
    return EARTH_RADIUS_KM * c  # Distance in kilometers

def latlon_to_index(lat, lon, lat_min, lon_min, lat_res, lon_res, grid_size):
    """
    Convert latitude and longitude to grid indices.
    """
    x = int((lat - lat_min) / lat_res)
    y = int((lon - lon_min) / lon_res)
    return (grid_size - 1 - x, y)  # Adjusted for 0-based indexing

def index_to_latlon(x, y, lat_min, lon_min, lat_res, lon_res, grid_size):
    """
    Convert grid indices to latitude and longitude.
    """
    lat = lat_min + (grid_size - 1 - x) * lat_res
    lon = lon_min + y * lon_res
    return (lat, lon)

def angle_difference(angle1, angle2):
    """
    Calculate the smallest difference between two angles.
    Result is between -pi and pi.
    """
    diff = angle1 - angle2
    while diff > math.pi:
        diff -= 2 * math.pi
    while diff < -math.pi:
        diff += 2 * math.pi
    return diff

def calculate_u10max(Cp, Af, Z):
    """
    Calculate the maximum crosswind the ship can withstand (u10max).
    Based on provided formula (Equation 8).
    """
    # Placeholder calculation:
    # Replace with the accurate formula as needed.
    u10max = 40 * (10 / Z) ** (1/8) * (Cp * Af * Z)
    return u10max

def calculate_risk_wind(ucross, u10max):
    """
    Calculate the wind risk for a segment.
    """
    if ucross < u10max:
        return ucross / u10max
    else:
        return 1.0

def calculate_risk_wave(T_theta, TE):
    """
    Calculate the wave risk for a segment based on resonance theory.
    """
    ratio = T_theta / TE
    if 0 <= ratio < 1:
        return ratio
    elif 1 <= ratio < 2:
        return 2 - ratio
    else:
        return 0.0

def calculate_risk(risk_wind, risk_wave, a1=1/3, a2=1/3):
    """
    Combine wind and wave risks.
    """
    return a1 * risk_wind + a2 * risk_wave

def calculate_actual_speed(V0, h, q, alpha, F, wind_dir_rad, usurf, vsurf, theta_ship):
    """
    Calculate the actual speed of the ship (Va) under wind and wave effects.
    Placeholder formula, adjust as needed.
    """
    Va = V0 - (1.08 * h - 0.126 * q * h + 2.77e-3 * F * math.cos(alpha)) * (1 - 2.33e-7 * ship_params['D'] * V0)
    
    # Water current component in the direction of ship's heading
    water_current_speed = usurf * math.cos(theta_ship) + vsurf * math.sin(theta_ship)
    
    # Effective speed is Va adjusted by water current
    effective_speed = Va + water_current_speed
    
    # Ensure effective speed is non-negative
    return max(0.1, effective_speed)

def valid_move(x, y, binary_map):
    """
    Check if a move is valid (within bounds and not an obstacle).
    """
    return 0 <= x < binary_map.shape[0] and 0 <= y < binary_map.shape[1] and binary_map[x, y] == 0

# ---------------------- Risk Calculation Functions ---------------------- #

def calculate_risk_values(F, wind_dir_rad, h, usurf, vsurf, theta_ship, pirate_risk):
    """
    Calculate combined risk based on wind, wave, and pirate attacks.
    """
    # Lateral wind speed (ucross)
    alpha = angle_difference(theta_ship, wind_dir_rad)
    ucross = F * abs(math.sin(alpha))
    
    # Calculate u10max
    u10max = calculate_u10max(ship_params['Cp'], ship_params['Af'], ship_params['Z'])
    
    # Wind risk
    risk_wind = calculate_risk_wind(ucross, u10max)
    
    # Wave period (T_theta)
    g = 9.81  # Acceleration due to gravity (m/s^2)
    T_theta = math.sqrt(h / g) if h > 0 else 0.0

    # Wave risk
    risk_wave = calculate_risk_wave(T_theta, ship_params['TE'])
    
    # Combined wind and wave risk
    risk_i = calculate_risk(risk_wind, risk_wave, ship_params['a1'], ship_params['a2'])

    # Incorporate pirate attack risk
    pirate_risk_factor = ship_params['pirate_risk_factor']  # Weight for pirate risk
    combined_risk = risk_i + pirate_risk_factor * pirate_risk
    # Ensure combined risk does not exceed 1
    combined_risk = min(combined_risk, 1.0)
    
    return combined_risk

# ---------------------- Placeholder Functions ---------------------- #

def holtrop_mennen(R, V, D):
    """
    Placeholder for the Holtrop-Mennen method to calculate calm water resistance (R_t).
    Implement the actual Holtrop-Mennen method or integrate with a hydrodynamic model.
    """
    # For demonstration, we'll use a simple empirical formula
    # Replace with actual Holtrop-Mennen implementation
    # R_t = a * V^2 + b * V + c
    a = 0.02
    b = 1.5
    c = 100
    return a * V**2 + b * V + c   

def calculate_added_resistance_waves(h):
    """
    Calculate the added resistance due to waves (R_aw).
    Implement based on relevant hydrodynamic formulas or standards.
    """
    # Placeholder formula; replace with actual calculation
    return 50 * h  # Example: linear relation with wave height

def calculate_added_resistance_wind(F, Cp, Af):
    """
    Calculate the added resistance due to wind (R_aa) based on ISO 15016:2015(E).
    """
    # ISO 15016:2015(E) formula implementation
    # R_aa = 0.5 * rho * Cp * Af * F^2
    rho = 1.225  # Air density at sea level (kg/m^3)
    return 0.5 * rho * Cp * Af * F**2

# ---------------------- Modified Theta* Algorithm ---------------------- #

def theta_star_weighted_path(
    start, goal, binary_map, wind_speed_map, wind_angle_map_rad, wave_height_map,
    usurf_map, vsurf_map, ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size,
    pirate_risk_map,
    weight_shortest=0.5, weight_safest=0.3, weight_fuel=0.2,
    a=0.1, b=0.05,
    eta_h=1.0, eta_s=1.0, eta_e=1.0, c_sfoc=180
):
    """
    Optimized Theta* pathfinding algorithm to find a path based on user-defined weights for
    shortest path, safest path, and fuel consumption.

    Returns:
    - path: List of grid indices representing the path.
    - total_weighted_cost: Combined cost based on the weights.
    - normalized_total_time: Total travel time for the path, normalized.
    - normalized_total_fuel: Total fuel consumption for the path, normalized.
    - normalized_total_risk: Total cumulative risk for the path, normalized.
    """

    # Initialize open list, cost tracking and came_from dictionary
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: start}
    g_score = {start: 0}
    total_time = {start: 0}
    total_fuel = {start: 0}
    total_risk = {start: 0}
    
    # Precompute start and goal lat-lon for heuristic
    start_lat, start_lon = index_to_latlon(*start, lat_min, lon_min, lat_res, lon_res, grid_size)
    goal_lat, goal_lon = index_to_latlon(*goal, lat_min, lon_min, lat_res, lon_res, grid_size)
    
    # Precompute heuristic based on Haversine distance
    heuristic = haversine(start_lat, start_lon, goal_lat, goal_lon) / ship_speed / 24  # Normalize with an estimated max time
    f_score = {start: heuristic}

    c = 0  # Debugging counter

    while open_list:
        current_f, current = heapq.heappop(open_list)
        
        if current == goal:
            # Reconstruct path
            path = []
            while current != came_from[current]:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, g_score[goal], total_time[goal], total_fuel[goal], total_risk[goal]
        
        # Define possible movements (8-connected grid)
        neighbors = [
            (current[0] - 1, current[1]),     # North
            (current[0] + 1, current[1]),     # South
            (current[0], current[1] - 1),     # West
            (current[0], current[1] + 1),     # East
            (current[0] - 1, current[1] - 1), # Northwest
            (current[0] - 1, current[1] + 1), # Northeast
            (current[0] + 1, current[1] - 1), # Southwest
            (current[0] + 1, current[1] + 1), # Southeast
        ]
        
        for neighbor in neighbors:
            if not valid_move(neighbor[0], neighbor[1], binary_map):
                continue
            
            pirate_risk = pirate_risk_map[neighbor[0], neighbor[1]]
            
            lat1, lon1 = index_to_latlon(*current, lat_min, lon_min, lat_res, lon_res, grid_size)
            lat2, lon2 = index_to_latlon(*neighbor, lat_min, lon_min, lat_res, lon_res, grid_size)
            
            distance = haversine(lat1, lon1, lat2, lon2)
            
            # Get environmental data at neighbor
            F = wind_speed_map[neighbor[0], neighbor[1]]
            wind_dir = wind_angle_map_rad[neighbor[0], neighbor[1]]
            h = wave_height_map[neighbor[0], neighbor[1]]
            usurf = usurf_map[neighbor[0], neighbor[1]]
            vsurf = vsurf_map[neighbor[0], neighbor[1]]
            
            # Assume wave direction is the direction of water current
            wave_dir = math.atan2(vsurf, usurf) if usurf != 0 or vsurf != 0 else 0.0
            
            # Compute ship's heading direction from current to neighbor
            dx = neighbor[1] - current[1]
            dy = neighbor[0] - current[0]
            theta_ship = math.atan2(dy, dx)
            
            # Relative angles
            q = angle_difference(theta_ship, wave_dir)
            alpha = angle_difference(theta_ship, wind_dir)
            
            Va = calculate_actual_speed(ship_speed, h, q, alpha, F, wind_dir, usurf, vsurf, theta_ship)
            if Va <= 0:
                Va = 0.1  # Assign a minimal speed to avoid division by zero
            
            # Resistance and fuel cost calculations
            R_t = holtrop_mennen(R=0, V=Va, D=ship_params['D'])
            R_aw = calculate_added_resistance_waves(h)
            R_aa = calculate_added_resistance_wind(F, ship_params['Cp'], ship_params['Af'])
            R_tot = max(R_t + R_aw + R_aa, 1e-3)
            
            p_b = (R_tot * Va) / (eta_e * eta_h * eta_s)
            p_b = max(p_b, 1e-3)  # Ensure p_b > 0
            fuel_consumption = p_b * c_sfoc
            fuel_cost = fuel_consumption * (distance / Va)
            
            # Risk calculations
            risk_i = calculate_risk_values(F, wind_dir, h, usurf, vsurf, theta_ship, pirate_risk)
            
            # Combined cost calculations
            cost_shortest = distance / Va
            cost_fuel = fuel_cost
            cost_safest = risk_i * WEIGHTING_FACTOR
            
            # Normalize costs
            norm_cost_shortest = cost_shortest / 24  # Assuming max time as 24 hours
            norm_cost_fuel = (cost_fuel / 1000)  # Assuming max fuel as 1000 units
            norm_cost_safest = (cost_safest / WEIGHTING_FACTOR)  # Normalize to 1
            
            # Compute weighted cost
            weighted_cost = (weight_shortest * norm_cost_shortest) + \
                            (weight_fuel * norm_cost_fuel) + \
                            (weight_safest * norm_cost_safest)
            
            tentative_g = g_score[current] + weighted_cost
            
            new_total_time = total_time[current] + cost_shortest
            new_total_fuel = total_fuel[current] + cost_fuel
            new_total_risk = max(total_risk[current], cost_safest)
            
            # Update if this path yields a lower weighted cost
            if neighbor not in g_score or tentative_g < g_score.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                total_time[neighbor] = new_total_time
                total_fuel[neighbor] = new_total_fuel
                total_risk[neighbor] = new_total_risk
                heuristic = haversine(lat2, lon2, goal_lat, goal_lon) / ship_speed / 24  # Normalize heuristic
                f = tentative_g + heuristic
                heapq.heappush(open_list, (f, neighbor))

    # ---------------------- Data Loading and Preparation ---------------------- #

def load_data():
    """
    Load and prepare all necessary data for pathfinding.

    Returns:
    - binary_map: 2D numpy array representing obstacles
    - wind_speed_map, wind_angle_map_rad, wave_height_map, usurf_map, vsurf_map: Environmental data
    - lat_min, lon_min, lat_res, lon_res: Map parameters
    - grid_size: Size of the grid (assumed square)
    """
    # Check if files exist
    required_files = [
        "indian_ocean_binary.tif",
        "wind_speed_data.npy",
        "wind_dir_data.npy",
        "wave_height_data.npy",
        "usurf_data.npy",
        "vsurf_data.npy",
        "filtered_coordinates.csv"
    ]
    
    for file in required_files:
        if not os.path.isfile(file):
            raise FileNotFoundError(f"Required file '{file}' not found in the current directory.")
    
    # Load and resize binary map
    binary_file = "indian_ocean_binary.tif"
    with rasterio.open(binary_file) as src:
        original_shape = src.shape
        binary_map_original = src.read(1)
        
        # Define target shape
        target_shape = (900, 900)
        grid_size = target_shape[0]  # Assuming square grid
        
        # Resample using nearest neighbor to preserve binary values
        binary_map = src.read(
            1,
            out_shape=target_shape,
            resampling=Resampling.nearest
        )
        
        # Update transform to match the new shape
        transform = src.transform * src.transform.scale(
            (original_shape[1] / target_shape[1]),
            (original_shape[0] / target_shape[0])
        )
    
    # Define map bounds (Assumed values; adjust based on actual data)
    lat_min, lat_max = -60, 30
    lon_min, lon_max = 20, 120
    lat_res = (lat_max - lat_min) / target_shape[0]
    lon_res = (lon_max - lon_min) / target_shape[1]
    
    # Load additional data
    wind_speed_map = np.load('wind_speed_data.npy')       # Wind speed (F) in m/s
    wind_angle_map_deg = np.load('wind_dir_data.npy')    # Wind direction in degrees
    wave_height_map = np.load('wave_height_data.npy')    # Wave height (h) in meters
    usurf_map = np.load('usurf_data.npy')                # Water current east-west component (m/s)
    vsurf_map = np.load('vsurf_data.npy')                # Water current north-south component (m/s)
    
    # Convert wind angles from degrees to radians
    wind_angle_map_rad = np.radians(wind_angle_map_deg)
    
    # Ensure all loaded maps have the correct shape
    assert wind_speed_map.shape == target_shape, "Wind speed map shape mismatch."
    assert wind_angle_map_rad.shape == target_shape, "Wind angle map shape mismatch."
    assert wave_height_map.shape == target_shape, "Wave height map shape mismatch."
    assert usurf_map.shape == target_shape, "usurf map shape mismatch."
    assert vsurf_map.shape == target_shape, "vsurf map shape mismatch."
    
    return binary_map, wind_speed_map, wind_angle_map_rad, wave_height_map, usurf_map, vsurf_map, lat_min, lon_min, lat_res, lon_res, grid_size

# ---------------------- Pirate Attack Processing ---------------------- #

def load_pirate_attacks(csv_file, lat_min, lon_min, lat_res, lon_res, grid_size, buffer_degree=0.5):
    """
    Load pirate attack coordinates and mark buffer zones on the risk map.
    """
    pirate_risk_map = np.zeros((grid_size, grid_size))
    
    # Load pirate attack coordinates
    attacks = pd.read_csv(csv_file)
    
    for index, row in attacks.iterrows():
        attack_lat = row['latitude']
        attack_lon = row['longitude']
        
        # Determine grid indices within the buffer zone
        lat_start = attack_lat - buffer_degree
        lat_end = attack_lat + buffer_degree
        lon_start = attack_lon - buffer_degree
        lon_end = attack_lon + buffer_degree
        
        # Convert lat/lon to grid indices
        i_start, j_start = latlon_to_index(lat_start, lon_start, lat_min, lon_min, lat_res, lon_res, grid_size)
        i_end, j_end = latlon_to_index(lat_end, lon_end, lat_min, lon_min, lat_res, lon_res, grid_size)
        
        # Ensure indices are within bounds
        i_start = max(i_start, 0)
        j_start = max(j_start, 0)
        i_end = min(i_end, grid_size - 1)
        j_end = min(j_end, grid_size - 1)
        
        # Increase risk in the buffer zone
        for i in range(i_start, i_end + 1):
            for j in range(j_start, j_end + 1):
                distance = math.sqrt((i - i_start)*2 + (j - j_start)*2)
                if distance <= buffer_degree / lat_res:  # Adjust buffer based on resolution
                    pirate_risk_map[i, j] += 1 - (distance / (buffer_degree / lat_res))
    
    # Normalize pirate risk map to 0-1
    if np.max(pirate_risk_map) > 0:
        pirate_risk_map = pirate_risk_map / np.max(pirate_risk_map)
    
    return pirate_risk_map

# ---------------------- Visualization Functions ---------------------- #

def plot_paths(binary_map, path_shortest, path_safest, path_fuel, path_weighted, lat_min, lon_min, lat_res, lon_res, grid_size, new_position=None):
    """
    Plot the shortest, safest, fuel-efficient, and weighted paths on the map.
    Optionally, plot the new ship position.
    """
    plt.figure(figsize=(12, 10))
    plt.imshow(binary_map, cmap='gray', origin='upper')
    
    if path_shortest:
        path_x_short, path_y_short = zip(*path_shortest)
        plt.plot(path_y_short, path_x_short, color='red', linewidth=2, label='Route 1: Shortest Path')
    
    if path_safest:
        path_x_saf, path_y_saf = zip(*path_safest)
        plt.plot(path_y_saf, path_x_saf, color='blue', linewidth=2, label='Route 2: Safest Path')
    
    if path_fuel:
        path_x_fuel, path_y_fuel = zip(*path_fuel)
        plt.plot(path_y_fuel, path_x_fuel, color='green', linewidth=2, label='Route 3: Fuel-Efficient Path')
    
    if path_weighted:
        path_x_weighted, path_y_weighted = zip(*path_weighted)
        plt.plot(path_y_weighted, path_x_weighted, color='pink', linewidth=2, label='Route 4: Weighted Path')
    
    # Mark start and goal
    if path_weighted:
        plt.scatter([path_weighted[0][1], path_weighted[-1][1]],
                    [path_weighted[0][0], path_weighted[-1][0]],
                    c=['green', 'yellow'], marker='o', label='Start/Goal')
    
    # Plot new ship position if provided
    if new_position:
        new_lat, new_lon = new_position
        new_x, new_y = latlon_to_index(new_lat, new_lon, lat_min, lon_min, lat_res, lon_res, grid_size)
        plt.scatter(new_y, new_x, c='cyan', marker='x', s=100, label='New Position (After Travel)')
    
    plt.legend()
    plt.title("Theta* Pathfinding: Routes 1-4")
    plt.xlabel("Longitude Index")
    plt.ylabel("Latitude Index")
    plt.grid(False)
    plt.show()

# ---------------------- CSV Saving Function ---------------------- #

def save_path_as_latlon_csv(path, lat_min, lon_min, lat_res, lon_res, grid_size, csv_file):
    """
    Convert a path of grid indices to latitude and longitude and save it directly to a CSV file.
    """
    def index_to_latlon_inner(x, y, lat_min, lon_min, lat_res, lon_res, grid_size):
        lat = lat_min + (grid_size - 1 - x) * lat_res
        lon = lon_min + y * lon_res
        return (lat, lon)

    with open(csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Latitude', 'Longitude'])  # Write header

        for point in path:
            i, j = point
            lat, lon = index_to_latlon_inner(i, j, lat_min, lon_min, lat_res, lon_res, grid_size)
            writer.writerow([lat, lon])

    print(f"Latitude and longitude data saved to {csv_file}")

# ---------------------- Simulate Travel Function ---------------------- #

def simulate_travel(path, wind_speed_map, wind_angle_map_rad, wave_height_map, usurf_map, vsurf_map, pirate_risk_map, lat_min, lon_min, lat_res, lon_res, grid_size, ship_params, travel_time=3.0):
    """
    Simulate the ship traveling along the selected path for a specified amount of time (in hours).
    
    Parameters:
    - path (list of tuples): List of grid coordinates (lat, lon) for the path.
    - wind_speed_map, wind_angle_map_rad, wave_height_map, usurf_map, vsurf_map: Environmental data.
    - pirate_risk_map (2D array): Pirate risk at each grid cell.
    - lat_min, lon_min, lat_res, lon_res (floats): Map parameters.
    - grid_size (int): Size of the grid.
    - ship_params (dict): Ship parameters for calculations.
    - travel_time (float): Time to travel in hours (default is 3.0).
    
    Returns:
    - new_position (tuple): New latitude and longitude after traveling.
    """
    total_time = 0.0

    for idx in range(len(path) - 1):
        current = path[idx]
        neighbor = path[idx + 1]

        # Convert grid indices to latitude and longitude
        lat1, lon1 = index_to_latlon(*current, lat_min, lon_min, lat_res, lon_res, grid_size)
        lat2, lon2 = index_to_latlon(*neighbor, lat_min, lon_min, lat_res, lon_res, grid_size)

        # Calculate Haversine distance in kilometers
        distance_km = haversine(lat1, lon1, lat2, lon2)

        # Retrieve environmental data for the neighbor cell
        F = wind_speed_map[neighbor[0], neighbor[1]]
        wind_dir = wind_angle_map_rad[neighbor[0], neighbor[1]]
        h = wave_height_map[neighbor[0], neighbor[1]]
        usurf = usurf_map[neighbor[0], neighbor[1]]
        vsurf = vsurf_map[neighbor[0], neighbor[1]]
        pirate_risk = pirate_risk_map[neighbor[0], neighbor[1]]

        # Calculate wave direction
        wave_dir = math.atan2(vsurf, usurf) if usurf != 0 or vsurf != 0 else 0.0

        # Compute ship's heading direction from current to neighbor
        dx = neighbor[1] - current[1]
        dy = neighbor[0] - current[0]
        theta_ship = math.atan2(dy, dx)

        # Calculate relative angles
        q = angle_difference(theta_ship, wave_dir)
        alpha = angle_difference(theta_ship, wind_dir)

        # Calculate actual speed Va
        Va = calculate_actual_speed(
            V0=ship_params.get('ship_speed', 40),  # Assuming ship_speed is provided
            h=h,
            q=q,
            alpha=alpha,
            F=F,
            wind_dir_rad=wind_dir,
            usurf=usurf,
            vsurf=vsurf,
            theta_ship=theta_ship
        )

        # Calculate time for this segment (hours)
        time_hours = distance_km / Va if Va > 0 else float('inf')

        if total_time + time_hours >= travel_time:
            # Calculate the remaining time fraction
            remaining_time = travel_time - total_time
            fraction = remaining_time / time_hours
            # Calculate new latitude and longitude based on the fraction
            new_lat = lat1 + (lat2 - lat1) * fraction
            new_lon = lon1 + (lon2 - lon1) * fraction
            return (new_lat, new_lon)
        else:
            total_time += time_hours

    # If the entire path is traversed within travel_time, return the goal position
    return index_to_latlon(*path[-1], lat_min, lon_min, lat_res, lon_res, grid_size)

# ---------------------- Calculate Path Metrics ---------------------- #

def calculate_path_metrics(
    path,
    wind_speed_map,
    wind_angle_map_rad,
    wave_height_map,
    usurf_map,
    vsurf_map,
    pirate_risk_map,
    lat_min,
    lon_min,
    lat_res,
    lon_res,
    grid_size,
    ship_params
):
    """
    Calculate total time, fuel, and risk for a given path based on environmental and ship parameters.
    
    Parameters:
    path (list of tuples): List of grid coordinates (lat, lon) for the path.
    wind_speed_map (2D array): Wind speed at each grid cell.
    wind_angle_map_rad (2D array): Wind angle (in radians) at each grid cell.
    wave_height_map (2D array): Wave height at each grid cell.
    usurf_map (2D array): East-west water current component.
    vsurf_map (2D array): North-south water current component.
    pirate_risk_map (2D array): Pirate risk at each grid cell.
    lat_min (float): Minimum latitude of the grid.
    lon_min (float): Minimum longitude of the grid.
    lat_res (float): Latitude resolution of the grid.
    lon_res (float): Longitude resolution of the grid.
    grid_size (int): Size of the grid (assumed square).
    ship_params (dict): Ship parameters for risk and fuel calculations.
    
    Returns:
    total_time (float): Total time in hours for the entire path.
    total_fuel (float): Total fuel consumption in gallons for the entire path.
    total_risk (float): Total cumulative risk for the path.
    """
    total_time = 0.0
    total_fuel = 0.0
    total_risk = 0.0  # Assuming cumulative risk along the path

    # Unpack ship parameters
    D = ship_params.get('D', 1000)    # Ship displacement (tons)
    Cp = ship_params.get('Cp', 0.5)   # Wind pressure coefficient
    Af = ship_params.get('Af', 50)    # Wind area (m^2)
    Z = ship_params.get('Z', 10)      # Height from center of wind area (m)
    TE = ship_params.get('TE', 10)    # Encounter period (seconds)
    n_h = ship_params.get('n_h', 0.7) # Hull efficiency factor
    n_s = ship_params.get('n_s', 0.75) # Shaft efficiency factor
    n_e = ship_params.get('n_e', 0.85) # Engine efficiency factor
    csfoc = ship_params.get('csfoc', 180) # Fuel consumption factor (g/kWh)
    a1 = ship_params.get('a1', 1/3)   # Risk weighting factor for wind
    a2 = ship_params.get('a2', 1/3)   # Risk weighting factor for waves
    pirate_risk_factor = ship_params.get('pirate_risk_factor', 0.3)  # Risk factor for piracy

    # Precompute u10max (assuming constant across the map; adjust if variable)
    u10max = calculate_u10max(Cp, Af, Z)

    # Iterate through each segment of the path
    for idx in range(len(path) - 1):
        current = path[idx]
        neighbor = path[idx + 1]

        # Convert grid indices to latitude and longitude
        lat1, lon1 = index_to_latlon(*current, lat_min, lon_min, lat_res, lon_res, grid_size)
        lat2, lon2 = index_to_latlon(*neighbor, lat_min, lon_min, lat_res, lon_res, grid_size)

        # Calculate Haversine distance in kilometers
        distance_km = haversine(lat1, lon1, lat2, lon2)

        # Retrieve environmental data for the neighbor cell
        F = wind_speed_map[neighbor[0], neighbor[1]]
        wind_dir = wind_angle_map_rad[neighbor[0], neighbor[1]]
        h = wave_height_map[neighbor[0], neighbor[1]]
        usurf = usurf_map[neighbor[0], neighbor[1]]
        vsurf = vsurf_map[neighbor[0], neighbor[1]]
        pirate_risk = pirate_risk_map[neighbor[0], neighbor[1]]

        # Calculate wave direction
        wave_dir = math.atan2(vsurf, usurf) if usurf != 0 or vsurf != 0 else 0.0

        # Compute ship's heading direction from current to neighbor
        dx = neighbor[1] - current[1]
        dy = neighbor[0] - current[0]
        theta_ship = math.atan2(dy, dx)

        # Calculate relative angles
        q = angle_difference(theta_ship, wave_dir)
        alpha = angle_difference(theta_ship, wind_dir)

        # Calculate actual speed Va
        Va = calculate_actual_speed(
            V0=ship_params.get('ship_speed', 40),  # Assuming ship_speed is provided
            h=h,
            q=q,
            alpha=alpha,
            F=F,
            wind_dir_rad=wind_dir,
            usurf=usurf,
            vsurf=vsurf,
            theta_ship=theta_ship
        )

        # Calculate time for this segment (hours)
        time_hours = distance_km / Va if Va > 0 else float('inf')
        total_time += time_hours

        # Resistance Calculations
        R_t = holtrop_mennen(R=0, V=Va, D=D)
        R_aw = calculate_added_resistance_waves(h)
        R_aa = calculate_added_resistance_wind(F, Cp, Af)
        R_tot = R_t + R_aw + R_aa
        R_tot = max(R_tot, 1e-3)  # Prevent division by zero

        # Fuel Consumption Calculations
        p_b = (R_tot * Va) / (n_e * n_h * n_s)
        p_b = max(p_b, 1e-3)  # Prevent division by zero
        fuel_consumption = p_b * csfoc  # Units: g/kWh * (kWh) = grams
        fuel_cost = fuel_consumption * (distance_km / Va)  # Adjust units as needed
        total_fuel += fuel_cost

        # Risk Calculations
        risk_i = calculate_risk_values(F, wind_dir, h, usurf, vsurf, theta_ship, pirate_risk)
        combined_risk = min(risk_i + pirate_risk_factor * pirate_risk, 1.0)
        total_risk += combined_risk  # Cumulative risk along the path

    # Convert fuel consumption from grams to gallons (assuming density ~850 g/L and 1 gallon ≈ 3.78541 liters)
    total_fuel_gallons = (total_fuel / 850) * 0.264172
    
    # Calculate average risk
    average_risk = (total_risk / len(path)) * 100  # Percentage

    return total_time, total_fuel_gallons, average_risk

# ---------------------- Main Execution ---------------------- #

def main(start_lat, start_lon, goal_lat, goal_lon, ship_speed, ship_dis, area_front, ship_height, ship_reso, hull_eff, prop_eff, engine_eff, c_sfoc, weight_shortest, weight_safest, weight_fuel):
    # Load data

    global D, Cp, Af, Z, TE, n_h, n_s, n_e, a1, a2, pirate_risk_factor, ship_speed_global, csfoc
    
    # Assign values to global variables
    D = ship_dis                # Ship displacement (tonnes)
    Cp = 0.5                   # Wind pressure coefficient
    Af = area_front             # Frontal area of the ship (m²)
    Z = ship_height              # Measurement height above sea surface (meters)
    TE = ship_reso                     # Ship's resonant period (seconds)
    n_h = hull_eff              # Hull efficiency
    n_s = prop_eff              # Propeller efficiency
    n_e = engine_eff            # Engine shaft efficiency
    csfoc = c_sfoc               # Specific Fuel Oil Consumption (g/kWh)
    a1 = 1 / 3                  # Weight for wind risk
    a2 = 1 / 3                  # Weight for wave risk
    pirate_risk_factor = 0.3    # Weight for pirate risk
    ship_speed_global = ship_speed  # Ship's hydrostatic speed in km/h

    try:
        binary_map, wind_speed_map, wind_angle_map_rad, wave_height_map, usurf_map, vsurf_map, lat_min, lon_min, lat_res, lon_res, grid_size = load_data()
    except FileNotFoundError as e:
        print(e)
        return
    except AssertionError as e:
        print(e)
        return
    
    # Load and process pirate attacks
    pirate_risk_map = load_pirate_attacks(
        csv_file='filtered_coordinates.csv',
        lat_min=lat_min,
        lon_min=lon_min,
        lat_res=lat_res,
        lon_res=lon_res,
        grid_size=grid_size,
        buffer_degree=0.5
    )    
       
    # Convert to grid indices
    start = latlon_to_index(start_lat, start_lon, lat_min, lon_min, lat_res, lon_res, grid_size)
    goal = latlon_to_index(goal_lat, goal_lon, lat_min, lon_min, lat_res, lon_res, grid_size)
    # goal=(450,450)
    
    # Ensure start and goal are within bounds and not on obstacles
    if not valid_move(*start, binary_map):
        raise ValueError("Start position is invalid or on an obstacle.")
    if not valid_move(*goal, binary_map):
        raise ValueError("Goal position is invalid or on an obstacle.")
    
    # Ship's hydrostatic speed in km/h
    ship_speed = 40  # km/h
    
    # Run Theta* Weighted algorithm
    print("Calculating the weighted path (Route 4)...")
    path_weighted, total_weighted_cost, normalized_total_time, normalized_total_fuel, normalized_total_risk = theta_star_weighted_path(
        start, goal, binary_map,
        wind_speed_map, wind_angle_map_rad,
        wave_height_map, usurf_map, vsurf_map,
        ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size,        
        pirate_risk_map,weight_shortest, weight_safest, weight_fuel,        
        a=0.1, b=0.05,
        eta_h=n_h, eta_s=n_s, eta_e=n_e,
        c_sfoc=csfoc
    )
    
    if path_weighted:
        print("Weighted path found successfully!")
        print(f"Total Weighted Cost: {total_weighted_cost:.4f}")
        print(f"Normalized Total Time: {normalized_total_time:.4f} hours")
        print(f"Normalized Total Fuel: {normalized_total_fuel:.4f} gallons")
        print(f"Normalized Total Risk: {normalized_total_risk:.2f}%")
        
        # Save path to CSV
        save_path_as_latlon_csv(path_weighted, lat_min=lat_min, lon_min=lon_min, lat_res=lat_res, lon_res=lon_res, grid_size=grid_size, csv_file='path_weighted.csv')
        
        # Plot the weighted path
        plot_paths(binary_map, path_shortest=None, path_safest=None, path_fuel=None, path_weighted=path_weighted, 
                   lat_min=lat_min, lon_min=lon_min, lat_res=lat_res, lon_res=lon_res, grid_size=grid_size)
        
        
        # Calculate and display path metrics
        total_time, total_fuel, total_risk = calculate_path_metrics(
            path_weighted,
            wind_speed_map,
            wind_angle_map_rad,
            wave_height_map,
            usurf_map,
            vsurf_map,
            pirate_risk_map,
            lat_min,
            lon_min,
            lat_res,
            lon_res,
            grid_size,
            ship_params
        )
        print("\n----- Route 4: Weighted Path Metrics -----")
        print(f"Total travel time : {total_time:.2f} hours")
        print(f"Total cumulative risk: {total_risk:.2f}")
        print(f"Total fuel consumption: {total_fuel:.2f} gallons")
    else:
        print("No weighted path could be found.")

# ---------------------- Execute Main Function ---------------------- #

if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Ship Routing Algorithm")
        parser.add_argument("start_lat", type=float)
        parser.add_argument("start_lon", type=float)
        parser.add_argument("goal_lat", type=float)
        parser.add_argument("goal_lon", type=float)
        parser.add_argument("ship_speed", type=float)
        parser.add_argument("ship_dis", type=float)
        parser.add_argument("ship_height", type=float)
        parser.add_argument("area_front", type=float)
        parser.add_argument("ship_reso", type=int)
        parser.add_argument("hull_eff", type=float)
        parser.add_argument("prop_eff", type=float)
        parser.add_argument("engine_eff", type=float)
        parser.add_argument("c_sfoc", type=float)
        parser.add_argument("weight_fuel", type=float)
        parser.add_argument("weight_safest", type=float)
        parser.add_argument("weight_shortest", type=float)

        args = parser.parse_args()

        main(start_lat=args.start_lat, start_lon=args.start_lon,
            goal_lat=args.goal_lat, goal_lon=args.goal_lon,
            ship_speed=args.ship_speed, ship_dis=args.ship_dis, 
            ship_height=args.ship_height,
            area_front=args.area_front, ship_reso=args.ship_reso,
            hull_eff=args.hull_eff, prop_eff=args.prop_eff,
            engine_eff=args.engine_eff, c_sfoc=args.c_sfoc,
            weight_fuel=args.weight_fuel, weight_safest=args.weight_safest, weight_shortest=args.weight_shortest)
        

    except ValueError as e:
        print(f"ValueError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")