import numpy as np
import heapq
import rasterio
from rasterio.enums import Resampling
import math
import pandas as pd
import csv
import argparse
import matplotlib
matplotlib.use('Agg')

# ---------------------- Constants and Parameters ---------------------- #
total_path_points=0
MAXT_time = 1e-3
MAXT_fuel = 1e-3
MAXT_safe = 1e-3
# Earth's radius in kilometers
EARTH_RADIUS_KM = 6371  

WIND_SPEED_THRESHOLD = 15.0       # Example threshold in m/s
WAVE_HEIGHT_THRESHOLD = 5.0       # Example threshold in meters
CURRENT_SPEED_THRESHOLD = 2.0  
E_speed=0

rho = 1.225
# Ship parameters (Adjust based on actual ship specifications)
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

ecological_area=[[]]

# Risk threshold
RISK_THRESHOLD = 0.6

# Weighting factor to prioritize safety over time (Adjust as needed)
WEIGHTING_FACTOR = 10  

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

def calculate_risk(wind_speed, wind_direction_deg, wave_height, current_speed_ns, current_speed_ew):
    """
    Calculate combined risk based on wind, wave, and pirate attacks.
    
    Parameters:
    - wind_speed (float): Wind speed in m/s.
    - wind_direction_deg (float): Wind direction in degrees.
    - wave_height (float): Wave height in meters.
    - current_speed_ns (float): North-South component of current speed in m/s.
    - current_speed_ew (float): East-West component of current speed in m/s.
    
    Returns:
    - risk (float): Combined risk value between 0 and 1.
    """
    risk = 0
    
    # Wind risk (higher speed = higher risk)
    if wind_speed > WIND_SPEED_THRESHOLD:
        risk += 0.3  # 30% weight for wind risk
    
    # Wave risk (higher waves = higher risk)
    if wave_height > WAVE_HEIGHT_THRESHOLD:
        risk += 0.3  # 30% weight for wave risk
    
    # Current risk (stronger currents = higher risk)
    # Corrected the calculation to use squares for accurate current speed magnitude
    current_speed = np.sqrt(current_speed_ns**2 + current_speed_ew**2)  # Calculate total current speed
    if current_speed > CURRENT_SPEED_THRESHOLD:
        risk += 0.2  # 20% weight for current risk
    
    # Wind direction risk (example: headwinds can increase risk)
    # Assuming wind_direction_deg is in [0, 360), and headwinds are between 0 and 180 degrees
    if 0 <= wind_direction_deg <= 180:
        risk += 0.2  # 20% weight for headwind risk
    
    return risk
def calculate_actual_speed(V0, h, q, alpha, F, wind_dir_deg, usurf, vsurf, theta_ship):
    """
    Calculate the actual speed of the ship (Va) under wind and wave effects.
    Placeholder formula, adjust as needed.
    """
    Va = V0 - (1.08 * h - 0.126 * q * h + 2.77e-3 * F * math.cos(alpha)) * (1 - 2.33e-7 * D * V0)
    
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
    global ecological_area
    
    return 0 <= x < binary_map.shape[0] and 0 <= y < binary_map.shape[1] and binary_map[x, y] == 0 and ecological_area[x,y]==0

def line_of_sight(map_array, start, end):
    """
    Check if there is a clear line of sight between start and end using Bresenham's Line Algorithm.
    """
    x0, y0 = start
    x1, y1 = end
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    n = 1 + dx + dy
    x_inc = 1 if x1 > x0 else -1
    y_inc = 1 if y1 > y0 else -1
    error = dx - dy
    dx *= 2
    dy *= 2
    
    for _ in range(n):
        if map_array[x, y] == 1:
            return False
        if error > 0:
            x += x_inc
            error -= dy
        else:
            y += y_inc
            error += dx
    return True


# ---------------------- Modified Theta* Algorithm Implementations (No line-of-sight shortcuts) ---------------------- #

def theta_star_shortest_path(start, goal, binary_map, wind_speed_map, wind_angle_map_deg, wave_height_map,
                             usurf_map, vsurf_map, ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size):
    """
    Theta* pathfinding algorithm to find the shortest path (minimum travel time) ignoring risks.
    Modified to avoid line-of-sight shortcutting.
    """
    global MAXT_time
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: start}
    g_score = {start: 0}
    
    # Heuristic based on Haversine distance
    start_lat, start_lon = index_to_latlon(*start, lat_min, lon_min, lat_res, lon_res, grid_size)
    goal_lat, goal_lon = index_to_latlon(*goal, lat_min, lon_min, lat_res, lon_res, grid_size)
    f_score = {start: haversine(start_lat, start_lon, goal_lat, goal_lon)}
    
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
            return path, g_score[goal]
        
        # Define possible movements
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
            
            lat1, lon1 = index_to_latlon(*current, lat_min, lon_min, lat_res, lon_res, grid_size)
            lat2, lon2 = index_to_latlon(*neighbor, lat_min, lon_min, lat_res, lon_res, grid_size)
            
            distance = haversine(lat1, lon1, lat2, lon2)
            
            # Get environmental data at neighbor
            F = max(wind_speed_map[neighbor[0], neighbor[1]], 0.1)
            wind_dir = wind_angle_map_deg[neighbor[0], neighbor[1]]
            h = max(wave_height_map[neighbor[0], neighbor[1]], 0.1)  # Ensure h > 0
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
            Va = max(Va, 0.1)

            
            # Time cost (hours)
            time_cost = distance / Va if Va > 0 else float('inf')
            MAXT_time = max(MAXT_time, time_cost)
            
            tentative_g = g_score[current] + time_cost
            
            if neighbor not in g_score or tentative_g < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                neighbor_lat, neighbor_lon = index_to_latlon(neighbor[0], neighbor[1], lat_min, lon_min, lat_res, lon_res, grid_size)
                heuristic = haversine(neighbor_lat, neighbor_lon, goal_lat, goal_lon) / ship_speed
                f = tentative_g + heuristic
                heapq.heappush(open_list, (f, neighbor))

def theta_star_safest_path(start, goal, binary_map, wind_speed_map, wind_angle_map_deg, wave_height_map,
                           usurf_map, vsurf_map, ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size, pirate_risk_map):
    """
    Theta* pathfinding algorithm to find the safest path (minimize max risk) ensuring no segment exceeds RISK_THRESHOLD.
    Modified to avoid line-of-sight shortcutting.
    """
    global MAXT_safe, E_speed
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: start}
    g_score = {start: 0}
    total_risk = {start: 0}
    
    # Heuristic based on Haversine distance
    start_lat, start_lon = index_to_latlon(*start, lat_min, lon_min, lat_res, lon_res, grid_size)
    goal_lat, goal_lon = index_to_latlon(*goal, lat_min, lon_min, lat_res, lon_res, grid_size)
    f_score = {start: haversine(start_lat, start_lon, goal_lat, goal_lon)}
    
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
            return path, g_score[goal], total_risk[goal]
        
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
            
            F = max(wind_speed_map[neighbor[0], neighbor[1]], 0.1)
            wind_dir = wind_angle_map_deg[neighbor[0], neighbor[1]]
            h = max(wave_height_map[neighbor[0], neighbor[1]], 0.1)  # Ensure h > 0
            usurf = usurf_map[neighbor[0], neighbor[1]]
            vsurf = vsurf_map[neighbor[0], neighbor[1]]
            
            wave_dir = math.atan2(vsurf, usurf) if usurf != 0 or vsurf != 0 else 0.0
            dx = neighbor[1] - current[1]
            dy = neighbor[0] - current[0]
            theta_ship = math.atan2(dy, dx)
            
            q = angle_difference(theta_ship, wave_dir)
            alpha = angle_difference(theta_ship, wind_dir)
            
            Va = calculate_actual_speed(ship_speed, h, q, alpha, F, wind_dir, usurf, vsurf, theta_ship)
            time_cost = distance / Va if Va > 0 else float('inf')
            E_speed=Va
            risk_i = calculate_risk(F,wind_dir,h,vsurf,usurf)+pirate_risk
            Va = max(Va, 0.1)
            
            combined_cost =  risk_i*WEIGHTING_FACTOR+time_cost+pirate_risk*WEIGHTING_FACTOR
            MAXT_safe = max(MAXT_safe, combined_cost)
            new_max_risk = max(total_risk[current], risk_i)
            
            tentative_g = g_score[current] + combined_cost
            
            # Update if this path yields a lower maximum risk or lower cost
            if neighbor not in g_score or new_max_risk < total_risk.get(neighbor, float('inf')):
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                total_risk[neighbor] = new_max_risk
                neighbor_lat, neighbor_lon = index_to_latlon(neighbor[0], neighbor[1], lat_min, lon_min, lat_res, lon_res, grid_size)
                heuristic = haversine(neighbor_lat, neighbor_lon, goal_lat, goal_lon) / ship_speed
                f = tentative_g + heuristic
                heapq.heappush(open_list, (f, neighbor))

def theta_star_min_fuel_path(
    start, goal, binary_map, wind_speed_map, wind_angle_map_deg, wave_height_map,
    usurf_map, vsurf_map, ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size,
    pirate_risk_map,
    a=0.1, b=0.05,
    eta_h=n_h, eta_s=n_s, eta_e=n_e, c_sfoc=csfoc
):
    """
    Theta* pathfinding algorithm to find the path with minimum fuel consumption.
    Modified to avoid line-of-sight shortcutting.
    Also returns the total time of the path.
    """
    global MAXT_fuel
    open_list = []
    heapq.heappush(open_list, (0, start))
    came_from = {start: start}
    fuel_score = {start: 0}
    total_time = {start: 0}  # Dictionary to track the total time taken for each point
    
    # Heuristic based on minimal possible fuel consumption
    start_lat, start_lon = index_to_latlon(*start, lat_min, lon_min, lat_res, lon_res, grid_size)
    goal_lat, goal_lon = index_to_latlon(*goal, lat_min, lon_min, lat_res, lon_res, grid_size)
    heuristic_fuel =  haversine(start_lat, start_lon, goal_lat, goal_lon) / ship_speed
    f_score = {start: heuristic_fuel}
    
    while open_list:
        current_f, current = heapq.heappop(open_list)
        
        if current == goal:
            # Reconstruct path
            path = []
            total_time_taken = total_time[goal]  # Capture the total time when goal is reached
            while current != came_from[current]:
                path.append(current)
                current = came_from[current]
            path.append(start)
            path.reverse()
            return path, fuel_score[goal], total_time_taken  # Return path, fuel score, and total time
        
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
            
            F = max(wind_speed_map[neighbor[0], neighbor[1]], 0.1)
            wind_dir = wind_angle_map_deg[neighbor[0], neighbor[1]]
            h = max(wave_height_map[neighbor[0], neighbor[1]], 0.1)  # Ensure h > 0
            usurf = usurf_map[neighbor[0], neighbor[1]]
            vsurf = vsurf_map[neighbor[0], neighbor[1]]
            
            wave_dir = math.atan2(vsurf, usurf) if usurf != 0 or vsurf != 0 else 0.0
            dx = neighbor[1] - current[1]
            dy = neighbor[0] - current[0]
            theta_ship = math.atan2(dy, dx)
            
            q = angle_difference(theta_ship, wave_dir)
            alpha = angle_difference(theta_ship, wind_dir)
            
            Va = calculate_actual_speed(ship_speed, h, q, alpha, F, wind_dir, usurf, vsurf, theta_ship)
            Va = max(Va, 0.1)

            
            # ------------------ Resistance Calculations ------------------ #
            
            # Calculate R_t using Holtrop-Mennen method
            # Placeholder: Replace with actual Holtrop-Mennen calculation or model integration
            R_t = holtrop_mennen(R=0, V=Va, D=D)  # You need to implement this function
            
            # Calculate Added Resistance due to Waves (Raw)
            # Placeholder formula; replace with actual calculation based on wave height
            R_aw = calculate_added_resistance_waves(h)
            
            # Calculate Added Resistance due to Wind (Raa)
            R_aa = calculate_added_resistance_wind(F, Cp, Af)
            
            # Total Resistance
            R_tot = R_t + R_aw + R_aa
            R_tot = max(R_tot, 1e-3)  # Ensure R_tot > 0
            
            # ------------------ Fuel Consumption Calculations ------------------ #
            
            # Calculate p_b
            p_b = (R_tot * ship_speed) / (n_e * n_h * n_s)
            p_b = max(p_b, 1e-3)  # Ensure p_b > 0
            
            # Fuel Consumption Estimate
            fuel_consumption = p_b * csfoc  # Units depend on csfoc
            
            # Fuel Cost (Fuel consumption over the distance segment)
            fuel_cost = fuel_consumption * (distance / Va)

            # fc=100*(1+0.1*(h/Va+F/Va))
            # fuel_consumption=fc*(distance/Va)
            MAXT_fuel = max(fuel_cost, MAXT_fuel)
            
            # Incorporate pirate risk into fuel consumption
            # fuel_cost *= (1 + pirate_risk)
            
            tentative_fuel = fuel_score[current] + fuel_cost
            tentative_time = total_time[current] + (distance / Va)  # Calculate time for this segment
            tentative_cost=tentative_fuel+tentative_time
            if neighbor not in fuel_score or tentative_fuel < fuel_score[neighbor]:
                came_from[neighbor] = current
                fuel_score[neighbor] = tentative_fuel
                total_time[neighbor] = tentative_time  # Update the total time for the neighbor
                neighbor_lat, neighbor_lon = index_to_latlon(neighbor[0], neighbor[1], lat_min, lon_min, lat_res, lon_res, grid_size)
                heuristic = haversine(neighbor_lat, neighbor_lon, goal_lat, goal_lon) / ship_speed
                f = tentative_cost + heuristic
                heapq.heappush(open_list, (f, neighbor))



def load_data():
    """
    Load and prepare all necessary data for pathfinding.
    weighted
    Returns:
    - binary_map: 2D numpy array representing obstacles
    - wind_speed_map, wind_angle_map_deg, wave_height_map, usurf_map, vsurf_map: Environmental data
    - lat_min, lon_min, lat_res, lon_res: Map parameters
    - grid_size: Size of the grid (assumed square)
    """
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
    
    # Define map bounds
    lat_min, lat_max = -60, 30
    lon_min, lon_max = 30, 120
    lat_res = (lat_max - lat_min) / target_shape[0]
    lon_res = (lon_max - lon_min) / target_shape[1]
    
    # Load additional data
    wind_speed_map = np.load('wind_speed_data.npy')       # Wind speed (F) in m/s
    wind_angle_map_deg = np.load('wind_dir_data.npy')    # Wind direction in degrees
    wave_height_map = np.load('wave_height_data.npy')    # Wave height (h) in meters
    usurf_map = np.load('usurf_data.npy')                # Water current east-west component (m/s)
    vsurf_map = np.load('vsurf_data.npy')                # Water current north-south component (m/s)
    
    # Convert wind angles from degrees to radians
    # wind_angle_map_rad = np.radians(wind_angle_map_deg)
    
    # Ensure all loaded maps have the correct shape
    assert wind_speed_map.shape == target_shape, "Wind speed map shape mismatch."
    assert wind_angle_map_deg.shape == target_shape, "Wind angle map shape mismatch."
    assert wave_height_map.shape == target_shape, "Wave height map shape mismatch."
    assert usurf_map.shape == target_shape, "usurf map shape mismatch."
    assert vsurf_map.shape == target_shape, "vsurf map shape mismatch."
    
    return binary_map, wind_speed_map, wind_angle_map_deg, wave_height_map, usurf_map, vsurf_map, lat_min, lon_min, lat_res, lon_res, grid_size

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
        pirate_risk_map[i_start:i_end+1, j_start:j_end+1] += 1  # Increment risk by 1
    
    # Normalize pirate risk map to 0-1
    if np.max(pirate_risk_map) > 0:
        pirate_risk_map = pirate_risk_map / np.max(pirate_risk_map)
    
    return pirate_risk_map



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

def simulate_travel(path, wind_speed_map, wind_angle_map_deg, wave_height_map, usurf_map, vsurf_map, pirate_risk_map, lat_min, lon_min, lat_res, lon_res, grid_size, ship_params, travel_time=3.0):
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
        wind_dir = wind_angle_map_deg[neighbor[0], neighbor[1]]
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
            wind_dir_deg=wind_dir,
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

# ---------------------- Main Execution ---------------------- #

def calculate_path_metrics(
    path,
    wind_speed_map,
    wind_angle_map_deg,
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

    global total_path_points
    # Unpack ship parameters

    print("total_path_points ",total_path_points)
    # Precompute u10max (assuming constant across the map; adjust if variable)
    # u10max = calculate_u10max(Cp, Af, Z)
    max_risk=0
    # Iterate through each segment of the path
    for idx in range(len(path) - 1):
        current = path[idx]
        neighbor = path[idx + 1]

        # Convert grid indices to latitude and longitude
        lat1, lon1 = index_to_latlon(*current, lat_min, lon_min, lat_res, lon_res, grid_size)
        lat2, lon2 = index_to_latlon(*neighbor, lat_min, lon_min, lat_res, lon_res, grid_size)

        # Calculate Haversine distance in kilometers
        distance_km = haversine(lat1, lon1, lat2, lon2)

        # Convert distance to meters for consistency (optional)
        # distance_m = distance_km * 1000

        # Retrieve environmental data for the neighbor cell
        F = max(wind_speed_map[neighbor[0], neighbor[1]], 0.1)
        wind_dir = wind_angle_map_deg[neighbor[0], neighbor[1]]
        h = max(wave_height_map[neighbor[0], neighbor[1]], 0.1)  # Ensure h > 0
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
            wind_dir_deg=wind_dir,
            usurf=usurf,
            vsurf=vsurf,
            theta_ship=theta_ship
        )
        Va = max(Va, 0.1)

        # Calculate time for this segment (hours)
        time_hours = distance_km / Va if Va > 0 else float('inf')
        total_time += time_hours

        # Resistance Calculations
        R_t = holtrop_mennen(R=0, V=Va, D=D)
        R_aw = calculate_added_resistance_waves(h)
        R_aa = calculate_added_resistance_wind(F, Cp=Cp, Af=Af)
        R_tot = R_t + R_aw + R_aa
        R_tot = max(R_tot, 1e-3)  # Prevent division by zero

        # Fuel Consumption Calculations
        p_b = (R_tot * ship_speed_global) / (n_e * n_h * n_s)
        p_b = max(p_b, 1e-3)  # Prevent division by zero
        fuel_consumption = p_b * csfoc  # Units: g/kWh * (kWh) = grams
        fuel_cost = fuel_consumption * (distance_km / Va)  # Adjust units as needed
        total_fuel += fuel_cost

        risk_i = calculate_risk(F,wind_dir,h,vsurf,usurf)*WEIGHTING_FACTOR+pirate_risk*WEIGHTING_FACTOR
        # combined_cost =  risk_i*WEIGHTING_FACTOR+time_cost+pirate_risk
        max_risk=max_risk+risk_i

    return total_time, ((total_fuel)/850)*0.264172, ((max_risk)/len(path))*100

# ---------------------- Main Function ---------------------- #
# def ecologicalarea()

def main(start_lat, start_lon, goal_lat, goal_lon, ship_speed, ship_dis, area_front, ship_height, ship_reso, hull_eff, prop_eff, engine_eff, c_sfoc):
    # Load data
    binary_map, wind_speed_map, wind_angle_map_deg, wave_height_map, usurf_map, vsurf_map, lat_min, lon_min, lat_res, lon_res, grid_size = load_data()
    global ecological_area

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

    print("Global variables initialized:")
    print(f"D = {D}, Cp = {Cp}, Af = {Af}, Z = {Z}, TE = {TE}, n_h = {n_h}, n_s = {n_s}, n_e = {n_e}, csfoc = {csfoc}")

    # Load and process pirate attacks
    pirate_risk_map = np.zeros((grid_size, grid_size))
    for i in range (100,200):
        for j in range(200,380):
            pirate_risk_map[i][j]=0.6
    for i in range (200,400):
        for j in range(180,380):
            pirate_risk_map[i][j]=0.3
    for i in range (200,400):
        for j in range(700,900):
            pirate_risk_map[i][j]=0.3

    depth_data=np.load('depth_data.npy')
    # print(len(depth_data))
    ecological_area=np.zeros((grid_size,grid_size))
    # for i in range(grid_size):
    #     for j in range(grid_size):
    #         # print(depth_data)
    #         if(depth_data[i][j]<15):
    #             print(i,j)
    #             ecological_area[i][j]=1
    data=[[[-5,-8],[70,74]],[[-8,-10],[45,48]],[[5,8],[92,95]],[[-2,-6],[55,58]]]
    for i in data:
        # print("HELLO ",i)
        i1,j1=latlon_to_index(i[0][0],i[1][0],lat_min,lon_min,lat_res,lon_res,grid_size)
        i2,j2=latlon_to_index(i[0][1],i[1][1],lat_min,lon_min,lat_res,lon_res,grid_size)
        # print(i1,j1,i2,j2)
        for k in range(min(i1,i2),max(i1,i2)):
            for p in range(min(j1,j2),max(j1,j2)):
                ecological_area[k][p]=1


    global total_path_points
    # print(len(wind_speed_map))
    # for i in range(300,350):
    #     for j in range(350,600):
    #         usurf_map[i][j]=2
    # for i in range(300,350):
    #     for j in range(350,600):
    #         vsurf_map[i][j]=3
    # for i in range(300,350):
    #     for j in range(350,600):
    #         wind_speed_map[i][j]=70
    # for i in range(300,350):
    #     for j in range(350,600):
    #         wave_height_map[i][j]=15
    


    # for i in range(300,350):
    #     for j in range(400,450):
    #         wind_speed_map[i][j]=100
    # Visualization of Wind Speed Map
    
    def save_plot(data, title, colorbar_label, filename, cmap='cool'):
        import matplotlib.pyplot as plt
        plt.figure(figsize=(10, 8))
        plt.imshow(data, cmap=cmap, origin='upper')
        plt.colorbar(label=colorbar_label)
        plt.title(title)
        plt.xlabel("Longitude Index")
        plt.ylabel("Latitude Index")
        plt.grid(False)
        plt.savefig(filename, format='svg')
        plt.close()

    save_plot(wind_speed_map, "Wind Speed Map", "Wind Speed (m/s)", "wind_speed_map.svg")
    save_plot(wave_height_map, "Wave Height Map", "Wave Height (m)", "wave_height_map.svg")
    save_plot(usurf_map, "East-West Water Current (USurf) Map", "U Surface Current (m/s)", "usurf_map.svg")
    save_plot(vsurf_map, "North-South Water Current (VSurf) Map", "V Surface Current (m/s)", "vsurf_map.svg")

    
    
    
    start = latlon_to_index(start_lat, start_lon, lat_min, lon_min, lat_res, lon_res, grid_size)

    goal = latlon_to_index(goal_lat, goal_lon, lat_min, lon_min, lat_res, lon_res, grid_size)
    if not valid_move(*start, binary_map):
        raise ValueError("Start position is invalid or on an obstacle.")
    if not valid_move(*goal, binary_map):
        raise ValueError("Goal position is invalid or on an obstacle.")
    
    # Ship's hydrostatic speed in km/h
    ship_speed = 40  # km/h
    
    # Run Theta* algorithm for shortest path
    print("Calculating the shortest path (Route 1)...")
    path_shortest, total_time_shortest = theta_star_shortest_path(
        start, goal, binary_map,
        wind_speed_map, wind_angle_map_deg,
        wave_height_map, usurf_map, vsurf_map,
        ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size
    )
    
    # Run Theta* algorithm for safest path
    print("Calculating the safest path (Route 2)...")
    path_safest, total_time_safest, total_risk_safest = theta_star_safest_path(
        start, goal, binary_map,
        wind_speed_map, wind_angle_map_deg,
        wave_height_map, usurf_map, vsurf_map,
        ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size,
        pirate_risk_map=pirate_risk_map
    )
    
    # Run Theta* algorithm for fuel-efficient path
    print("Calculating the fuel-efficient path (Route 3)...")
    path_fuel, total_fuel, total_fuel_time = theta_star_min_fuel_path(
        start, goal, binary_map,
        wind_speed_map, wind_angle_map_deg,
        wave_height_map, usurf_map, vsurf_map,
        ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size,
        pirate_risk_map=pirate_risk_map,
        a=0.1, b=0.05,  # Example parameters; adjust as needed
        eta_h=n_h, eta_s=n_s, eta_e=n_e, c_sfoc=csfoc
    )
    
    # # Run Theta* algorithm for weighted path
    # print("Calculating the weighted path based on user-defined weights (Route 4)...")
    
    # path_weighted_k = k_paths_dijkstra_constrained_path_2(
    #     start, goal, binary_map,
    #     wind_speed_map, wind_angle_map_deg,
    #     wave_height_map, usurf_map, vsurf_map,
    #     ship_speed, lat_min, lon_min, lat_res, lon_res, grid_size,
    #     pirate_risk_map=pirate_risk_map,
    #     k = 5,
    #     constraints={'fuel': 41300}, # Example constraint; adjust as needed
    #     optimize_for='time'
    # )
    # path_weighted, path_weighted_time, path_weighted_fuel, path_weighted_risk = None, None, None, None
    # if path_weighted_k is None:
    #     print("Weighted path not found.")
    # else:
    #     path_weighted = path_weighted_k[0][0]
    #     path_weighted_time = path_weighted_k[0][1]
    #     path_weighted_fuel = path_weighted_k[0][2]
    #     path_weighted_risk = path_weighted_k[0][3]
    
    # Save paths to CSV
    csv_file_fuel = 'path_fuel.csv'
    csv_file_safe = 'path_safe.csv'
    csv_file_short = 'path_short.csv'
    
    if path_safest:
        save_path_as_latlon_csv(path_safest, lat_min, lon_min, lat_res, lon_res, grid_size, csv_file_safe)
    if path_shortest:
        save_path_as_latlon_csv(path_shortest, lat_min, lon_min, lat_res, lon_res, grid_size, csv_file_short)
    if path_fuel:
        save_path_as_latlon_csv(path_fuel, lat_min, lon_min, lat_res, lon_res, grid_size, csv_file_fuel)
    
    
    # Visualization of all paths
    # plot_paths(binary_map, path_shortest, path_safest, path_fuel, path_weighted, lat_min, lon_min, lat_res, lon_res, grid_size)
    total_path_points=len(path_safest)
    # User Selection of Route
    # print("\nAvailable Routes:")
    # print("1. Route 1: Shortest Path")
    # print("2. Route 2: Safest Path")
    # print("3. Route 3: Fuel-Efficient Path")
    # print("4. Route 4: Weighted Path")
    
    # Prompt user to choose a route
    # while True:
    #     try:
    #         choice = int(input("Select a route to travel (1-4): "))
    #         if choice not in [1, 2, 3, 4]:
    #             print("Invalid choice. Please select a number between 1 and 4.")
    #             continue
    #         break
    #     except ValueError:
    #         print("Invalid input. Please enter a number between 1 and 4.")
    
    # Assign selected path based on user choice
    # if choice == 1:
    #     selected_path = path_shortest
    #     route_name = "Route 1: Shortest Path"
    # elif choice == 2:
    #     selected_path = path_safest
    #     route_name = "Route 2: Safest Path"
    # elif choice == 3:
    #     selected_path = path_fuel
    #     route_name = "Route 3: Fuel-Efficient Path"
    # elif choice == 4:
    #     selected_path = path_weighted
    #     route_name = "Route 4: Weighted Path"
    
    # if not selected_path:
    #     print(f"{route_name} could not be found.")
    #     return
    
    # Simulate travel for 3 hours
    # new_position = simulate_travel(
    #     path=selected_path,
    #     wind_speed_map=wind_speed_map,
    #     wind_angle_map_deg=wind_angle_map_deg,
    #     wave_height_map=wave_height_map,
    #     usurf_map=usurf_map,
    #     vsurf_map=vsurf_map,
    #     pirate_risk_map=pirate_risk_map,
    #     lat_min=lat_min,
    #     lon_min=lon_min,
    #     lat_res=lat_res,
    #     lon_res=lon_res,
    #     grid_size=grid_size,
    #     ship_params=ship_params,
    #     travel_time=3  # hours
    # )
    
    # Save the new position to a CSV
    # save_path_as_latlon_csv([new_position], lat_min, lon_min, lat_res, lon_res, grid_size, 'new_position.csv')
    
    # print(f"\nAfter traveling for 3 hours along {route_name}, the new position is:")
    # print(f"Latitude: {new_position[0]:.4f}, Longitude: {new_position[1]:.4f}")
    
    # # Plot the new position on the map
    # plot_paths(binary_map, path_shortest, path_safest, path_fuel, path_weighted, lat_min, lon_min, lat_res, lon_res, grid_size, new_position=new_position)
    
    total_path_points=len(path_safest)
    print("TOTAL_path ",total_path_points)
    # Output results for all paths
    if path_shortest:
        total_time, total_fuel, total_risk = calculate_path_metrics(
            path_shortest,
            wind_speed_map,
            wind_angle_map_deg,
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
        # print("----- Route 1: Shortest Path -----")
        # print(f"Total travel time : {total_time:.2f} hours")
        # print(f"Total cumulative risk: {total_risk:.2f}")
        # print(f"Total fuel consumption: {total_fuel:.2f} gallons")

        total_time_shortest1 = total_time
        total_risk_shortest1 = total_risk
        total_fuel_shortest1 = total_fuel

    
    if path_safest:
        print("\n----- Route 2: Safest Path -----")
        total_time, total_fuel, total_risk = calculate_path_metrics(
            path_safest,
            wind_speed_map,
            wind_angle_map_deg,
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
        total_time_safest1 = total_time
        total_risk_safest1 = total_risk
        total_fuel_safest1 = total_fuel
    
    if path_fuel:
        print("\n----- Route 3: Fuel-Efficient Path -----")
        total_time, total_fuel, total_risk = calculate_path_metrics(
            path_fuel,
            wind_speed_map,
            wind_angle_map_deg,
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
        total_time_fuel1 = total_time
        total_risk_fuel1 = total_risk
        total_fuel_fuel1 = total_fuel
    
    # if path_weighted:
    #     print("\n----- Route 4: Weighted Path -----")
    #     total_time, total_fuel, total_risk = calculate_path_metrics(
    #         path_weighted,
    #         wind_speed_map,
    #         wind_angle_map_deg,
    #         wave_height_map,
    #         usurf_map,
    #         vsurf_map,
    #         pirate_risk_map,
    #         lat_min,
    #         lon_min,
    #         lat_res,
    #         lon_res,
    #         grid_size,
    #         ship_params
    #     )
    #     print(f"Total travel time : {total_time:.2f} hours")
    #     print(f"Total cumulative risk: {total_risk:.2f}")
    #     print(f"Total fuel consumption: {total_fuel:.2f} gallons")
    
    # if not path_shortest and not path_safest and not path_fuel and not path_weighted:
    #     print("No path could be found.")

    results = [[total_time_safest1, total_risk_safest1, total_fuel_safest1], [total_time_shortest1, total_risk_shortest1, total_fuel_shortest1], [total_time_fuel1, total_risk_fuel1, total_fuel_fuel1]]



    return results

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
    return 0.2 * h  # Example: linear relation with wave height

def calculate_added_resistance_wind(F, Cp, Af):
    """
    Calculate the added resistance due to wind (R_aa) based on ISO 15016:2015(E).
    """
    # ISO 15016:2015(E) formula implementation
    # R_aa = 0.5 * rho * Cp * Af * F^2
    return 0.5 * rho * Cp * Af * F**2

# ---------------------- Execute Main Function ---------------------- #

if __name__ == "__main__":
    main()