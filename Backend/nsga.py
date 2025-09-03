import math
import random
import numpy as np
import numpy as np
import rasterio
from rasterio.enums import Resampling
import math
import random
import heapq

from pymoo.core.problem import Problem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.crossover import Crossover
from pymoo.core.mutation import Mutation
from pymoo.core.sampling import Sampling
from pymoo.core.duplicate import ElementwiseDuplicateElimination
from pymoo.core.callback import Callback
from pymoo.optimize import minimize

import os
import csv

ship_weight = None
a1=0.5
a2=0.5
Fb=2.0  # Freeboard in meters
V0=15
a=1.0  # Fuel consumption coefficient
b=0.05   # Fuel consumption exponent

def latlon_to_index(lat, lon, transform):
    """
    Converts latitude and longitude to map indices using Rasterio's transform.
    """
    col, row = ~transform * (lon, lat)
    col = int(col)
    row = int(row)
    return (row, col)

def index_to_latlon(row, col, transform):
    """
    Converts map indices back to latitude and longitude using Rasterio's transform.
    """
    lon, lat = transform * (col + 0.5, row + 0.5)  # Center of the pixel
    return (lat, lon)

def valid_move(x, y, binary_map):
    """
    Check if a move is valid (within bounds and not on land).
    """
    return 0 <= x < binary_map.shape[0] and 0 <= y < binary_map.shape[1] and binary_map[x, y] == 0

def is_path_clear(lat1, lon1, lat2, lon2, transform, binary_map, num_samples=100):
    """
    Checks if the straight path between two waypoints crosses land.

    Parameters:
    - lat1, lon1: Latitude and longitude of the first waypoint
    - lat2, lon2: Latitude and longitude of the second waypoint
    - transform: Affine transform of the raster
    - binary_map: 2D numpy array representing land (1) and water (0)
    - num_samples: Number of samples along the path to check

    Returns:
    - True if the path is clear (all sampled points are water), False otherwise
    """
    # Generate linearly spaced points between the two waypoints
    lats = np.linspace(lat1, lat2, num_samples)
    lons = np.linspace(lon1, lon2, num_samples)
    
    for lat, lon in zip(lats, lons):
        try:
            row, col = latlon_to_index(lat, lon, transform)
            if not (0 <= row < binary_map.shape[0] and 0 <= col < binary_map.shape[1]):
                # Out of bounds implies crossing land or invalid area
                return False
            if binary_map[row, col] != 0:
                return False  # Land encountered
        except:
            # Any exception implies invalid point, treat as land
            return False
    return True

def haversine_distance(lat1, lon1, lat2, lon2):
    """
    Calculate the great-circle distance between two points on the Earth's surface.
    Returns distance in nautical miles.
    """
    R = 3440.065  # Radius of Earth in nautical miles
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    delta_phi = math.radians(lat2 - lat1)
    delta_lambda = math.radians(lon2 - lon1)

    a = math.sin(delta_phi / 2)**2 + math.cos(phi1) * math.cos(phi2) * math.sin(delta_lambda / 2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

    distance = R * c
    return distance

def calculate_bearing(lat1, lon1, lat2, lon2):
    """
    Calculates the bearing from the first point to the second point.
    Returns bearing in degrees from north.
    """
    lat1_rad = math.radians(lat1)
    lat2_rad = math.radians(lat2)
    delta_lon_rad = math.radians(lon2 - lon1)

    x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
    y = math.cos(lat1_rad) * math.sin(lat2_rad) - \
        math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad)

    initial_bearing = math.atan2(x, y)
    initial_bearing_deg = (math.degrees(initial_bearing) + 360) % 360

    return initial_bearing_deg

def calculate_bearing_angle(heading, direction):
    """
    Calculates the smallest relative angle between two bearings.
    
    Parameters:
    - heading: Ship's heading in degrees
    - direction: Wind or wave direction in degrees
    
    Returns:
    - relative_angle: Smallest angle between the two directions (0 to 180 degrees)
    """
    if direction is None:
        return random.uniform(0, 180)  # Generate random angle if direction is missing
    relative_angle = abs(heading - direction) % 360
    relative_angle = min(relative_angle, 360 - relative_angle)  # Ensure <= 180
    return relative_angle

def calculate_actual_speed(V0, h, F, alpha, D=None, q=None):
    """
    Calculates the actual speed of the ship under winds and waves using Feng's formula.
    
    Parameters:
    - V0: Hydrostatic speed of the ship (knots)
    - h: Significant wave height (meters)
    - F: Wind speed (knots)
    - alpha: Relative angle between ship's heading and wind direction (degrees)
    - D: Actual displacement of the ship (tons). If None, generate random value.
    - q: Relative angle between ship’s heading and wave direction (degrees). If None, generate random value.
    
    Returns:
    - Va: Actual speed of the ship (knots)
    """
    # Generate random D if not provided (Assume between 5000 and 20000 tons)
    if D is None:
        D = random.uniform(5000, 20000)
    
    # Generate random q if not provided (Assume between 0 and 180 degrees)
    if q is None:
        q = random.uniform(0, 180)
    
    # Convert angles from degrees to radians for cosine calculation
    alpha_rad = math.radians(alpha)
    
    # Calculate Va using Feng's formula
    Va = (V0 
          - 1.08 * h 
          - 0.126 * q * h 
          + 2.77e-3 * F * math.cos(alpha_rad) 
          - 2.33e-7 * D * V0)
    
    # Ensure that Va does not drop below a minimum speed (e.g., 5 knots)
    Va = max(Va, 5.0)
    
    return Va

class Fitness:
    def __init__(self, transform, binary_map, wind_dir_data, wind_speed_data, wave_height_data):
        self.transform = transform
        self.binary_map = binary_map
        self.wind_dir_data = wind_dir_data
        self.wind_speed_data = wind_speed_data
        self.wave_height_data = wave_height_data

        self.u10max = np.max(wind_speed_data)
        if self.u10max == 0:
            self.u10max = 1  # Prevent division by zero

    def calculate_fitness(self, individual):
        """
        Calculates the total travel time, average risk, and total fuel consumption for a given route.
        
        Parameters:
        - individual: List of (lat, lon) tuples representing the route
        - V0: Hydrostatic speed of the ship (knots)
        - transform: Affine transform of the raster
        - wind_dir_data: 2D numpy array of wind directions (degrees from north)
        - wind_speed_data: 2D numpy array of wind speeds (knots)
        - wave_height_data: 2D numpy array of wave heights (meters)
        - wave_dir_data: 2D numpy array of wave directions (degrees from north). If None, handled in speed calculation
        - D: Actual displacement of the ship (tons). If None, generate random value per segment
        - a1: Weight for wind risk
        - a2: Weight for wave risk
        - Fb: Freeboard of the ship (meters)
        - a: Fuel consumption coefficient
        - b: Fuel consumption exponent
        - u10max: Maximum wind speed (knots). If None, it will be calculated.
        
        Returns:
        - total_time: Total travel time in hours
        - average_risk: Average risk across all segments
        - total_fuel: Total fuel consumption for the route
        """
        total_time = 0.0
        total_risk = 0.0
        total_fuel = 0.0
        num_segments = len(individual) - 1

        for i in range(num_segments):
            lat1, lon1 = individual[i]
            lat2, lon2 = individual[i + 1]
            
            # Calculate ship heading
            heading = calculate_bearing(lat1, lon1, lat2, lon2)
            
            # Get wind and wave data at the midpoint of the segment
            mid_lat = (lat1 + lat2) / 2
            mid_lon = (lon1 + lon2) / 2
            row, col = latlon_to_index(mid_lat, mid_lon, self.transform)
            
            # Handle edge cases where midpoint might be out of bounds
            if not (0 <= row < self.wind_dir_data.shape[0] and 0 <= col < self.wind_dir_data.shape[1]):
                print(f"Warning: Midpoint ({mid_lat}, {mid_lon}) is out of bounds. Assigning minimum speed and maximum risk.")
                Va = 5.0  # Minimum speed
                distance = haversine_distance(lat1, lon1, lat2, lon2)
                time = distance / Va
                total_time += time
                risk_i = 1.0  # Maximum risk
                total_risk += risk_i
                Qti = a * math.exp(b * Va)  # Fuel consumption rate (units per hour)
                fuel_i = Qti * time  # Fuel consumed during the segment
                total_fuel += fuel_i
                print(f"Segment {i+1}: Va={Va:.2f} knots, Time={time:.2f} hours, Fuel={fuel_i:.2f} units, Risk={risk_i:.2f}")
                print("-------------------------------")
                continue  # Proceed to next segment
            
            wind_dir = self.wind_dir_data[row, col]
            wind_speed = self.wind_speed_data[row, col]
            wave_height = self.wave_height_data[row, col]
            
            # Clamp environmental data to realistic maximums
            MAX_WAVE_HEIGHT = 10.0  # meters
            MAX_WIND_SPEED = 100.0  # knots
            wave_height = min(wave_height, MAX_WAVE_HEIGHT)
            wind_speed = min(wind_speed, MAX_WIND_SPEED)
            
            # Calculate relative angles
            alpha = calculate_bearing_angle(heading, wind_dir)
            q = None
            
            # Generate random displacement if not provided
            current_D = random.uniform(5000, 20000)  # Example range
            
            # Calculate actual speed Va using Feng's formula
            Va = calculate_actual_speed(V0=V0, h=wave_height, F=wind_speed, alpha=alpha, D=current_D, q=q)
            
            # Ensure Va is within reasonable bounds
            MIN_VA = 5.0  # knots
            if Va < MIN_VA:
                print(f"Warning: Va ({Va:.2f} knots) below minimum threshold. Adjusting to {MIN_VA} knots.")
                Va = MIN_VA
            
            # Calculate distance and time for the segment
            distance = haversine_distance(lat1, lon1, lat2, lon2)  # Nautical miles
            time = distance / Va  # Time in hours
            total_time += time
            
            # Calculate risk factors
            relative_angle_rad = math.radians(alpha)
            ucross = wind_speed * math.sin(relative_angle_rad)
            risk_wind = ucross / self.u10max
            risk_wind = min(risk_wind, 1.0)  # Cap at 1.0
            risk_wave = wave_height / Fb
            risk_wave = min(risk_wave, 1.0)  # Cap at 1.0
            risk_i = a1 * risk_wind + a2 * risk_wave
            total_risk += risk_i
            
            # Calculate fuel consumption for the segment
            Qti = a * math.exp(b * Va)  # Fuel consumption rate (units per hour)
            fuel_i = Qti * time  # Fuel consumed during the segment
            total_fuel += fuel_i
            
            # # Debugging Information
            # print(f"Segment {i+1}:")
            # print(f"  Start: ({lat1}, {lon1})")
            # print(f"  End: ({lat2}, {lon2})")
            # print(f"  Distance: {distance:.2f} NM")
            # print(f"  Actual Speed (Va): {Va:.2f} knots")
            # print(f"  Time: {time:.2f} hours")
            # print(f"  Fuel Consumption: {fuel_i:.2f} units")
            # print(f"  Risk: {risk_i:.2f}")
            # print("-------------------------------")
        
        average_risk = total_risk / num_segments
        return total_time, average_risk, total_fuel


# from utils import index_to_latlon, valid_move, is_path_clear, Fitness

def generate_initial_population(fitness: Fitness, reference_path, population_size=10, max_retries=100):
    """
    Generates an initial population for the genetic algorithm with updated fitness metrics.
    
    Parameters:
    - reference_path: List of tuples representing the reference route indices [(row, col), ...]
    - population_size: Number of individuals to generate
    - max_retries: Maximum number of attempts to find a valid waypoint
    
    Returns:
    - population: List of individuals, where each individual is a dictionary containing:
        - 'route': List of (lat, lon) tuples
        - 'total_time': Total travel time in hours
        - 'average_risk': Average risk across all segments
        - 'total_fuel': Total fuel consumption for the route
    """
    population = []
    attempts = 0
    max_total_attempts = population_size * max_retries * 2  # Arbitrary large number to prevent infinite loops
    
    end_point = reference_path[-1]
    avg_len = 10

    while len(population) < population_size and attempts < max_total_attempts:
        current_point = reference_path[0]
        route = [index_to_latlon(current_point[0], current_point[1], fitness.transform)]
        path_direction = (end_point[0] - current_point[0], end_point[1] - current_point[1])
        path_magnitude = math.sqrt(path_direction[0]**2 + path_direction[1]**2)
        while current_point != end_point:
            end_direction = (end_point[0] - current_point[0], end_point[1] - current_point[1])

            magnitude = math.sqrt(end_direction[0]**2 + end_direction[1]**2)
            if magnitude < path_magnitude/avg_len and len(route) > 0:
                x = end_point[0]
                y = end_point[1]
                prev_lat, prev_lon = route[-1]
                lat, lon = index_to_latlon(x, y, fitness.transform)
                if valid_move(x, y, fitness.binary_map) and is_path_clear(prev_lat, prev_lon, lat, lon, fitness.transform, fitness.binary_map):
                    route.append((lat, lon))
                    current_point = (x, y)
                    break
            
            steps = 1 + max(0, avg_len - len(route))
            x = current_point[0] + int(random.normalvariate(end_direction[0]/steps, end_direction[0]/steps))
            y = current_point[1] + int(random.normalvariate(end_direction[1]/steps, end_direction[1]/steps))
            prev_lat, prev_lon = route[-1]
            lat, lon = index_to_latlon(x, y, fitness.transform)
            if valid_move(x, y, fitness.binary_map) and is_path_clear(prev_lat, prev_lon, lat, lon, fitness.transform, fitness.binary_map):
                route.append((lat, lon))
                current_point = (x, y)
            else:
                continue
        
        total_time, average_risk, total_fuel = fitness.calculate_fitness(route)

        if average_risk > 0.99:
            print(f"Warning: Individual {len(population)+1} has high risk ({average_risk:.2f}). Skipping.")
            continue  # Skip adding this individual
        else:
            individual_dict = {
                'route': route,
                'total_time': total_time,
                'average_risk': average_risk,
                'total_fuel': total_fuel
            }
            population.append(individual_dict)

    if len(population) < population_size:
        print(f"Error: Only generated {len(population)} unique valid individuals after {attempts} attempts.")
        print("Consider increasing 'max_total_attempts' or adjusting parameters.")
    
    return population


# from utils import index_to_latlon, valid_move, haversine_distance

def a_star_search(start, goal, binary_map, transform, ship_speed):
    """
    A* search algorithm to find the optimal path for the ship.
    """
    open_set = []
    heapq.heappush(open_set, (0, start))  # (f_score, (row, col))

    came_from = {}

    g_score = {start: 0}

    f_score = {start: heuristic(start[0], start[1], goal, transform, ship_speed)}

    # Directions for neighbors (8 directions)
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1),
                  (-1, -1), (-1, 1), (1, -1), (1, 1)]

    while open_set:
        current_f, current = heapq.heappop(open_set)
        current_x, current_y = current

        if current == goal:
            # Reconstruct path
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start)  # Include the start node
            path.reverse()
            return path

        for dx, dy in directions:
            neighbor_x, neighbor_y = current_x + dx, current_y + dy
            if valid_move(neighbor_x, neighbor_y, binary_map):
                tentative_g_score = g_score[current] + calculate_time(
                    current_x, current_y, neighbor_x, neighbor_y,
                    transform, ship_speed
                )

                neighbor = (neighbor_x, neighbor_y)
                if neighbor not in g_score or tentative_g_score < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g_score
                    f = tentative_g_score + heuristic(
                        neighbor_x, neighbor_y, goal,
                        transform, ship_speed
                    )
                    f_score[neighbor] = f
                    heapq.heappush(open_set, (f, neighbor))

    return None  # No path found

def heuristic(x, y, goal, transform, ship_speed):
    """
    Heuristic function for A* (estimated time to goal).
    Uses straight-line distance divided by ship speed.
    """
    lat, lon = index_to_latlon(x, y, transform)
    goal_lat, goal_lon = index_to_latlon(goal[0], goal[1], transform)
    distance = haversine_distance(lat, lon, goal_lat, goal_lon)
    return distance / ship_speed

def calculate_time(x1, y1, x2, y2, transform, ship_speed):
    """
    Calculates the time to travel between two points based on ship speed.
    Uses the Haversine formula for distance calculation.
    """
    lat1, lon1 = index_to_latlon(x1, y1, transform)
    lat2, lon2 = index_to_latlon(x2, y2, transform)
    
    distance = haversine_distance(lat1, lon1, lat2, lon2)
    time = distance / ship_speed  # Time in hours
    return time


# def plot_last_front(binary_map, transform, population, reference_path_latlon, start, goal, lon_min, lon_max, lat_min, lat_max):
#     """
#     Plots the initial population of routes on the map with each route in a different color.
    
#     Parameters:
#     - binary_map: 2D numpy array representing the map
#     - transform: Affine transform of the raster
#     - population: List of individuals, each a list of (lat, lon) tuples
#     - reference_path_latlon: List of (lat, lon) tuples for the reference route
#     - start: (lat, lon) tuple for the start point
#     - goal: (lat, lon) tuple for the goal point
#     - lon_min, lon_max, lat_min, lat_max: Bounds for plotting
#     """
#     plt.figure(figsize=(12, 10))
#     ax = plt.gca()
    
#     # Plot the ocean map
#     plt.imshow(binary_map, cmap='gray', extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
    
#     # Plot reference route
#     ref_lats = [p[0] for p in reference_path_latlon]
#     ref_lons = [p[1] for p in reference_path_latlon]
#     plt.plot(ref_lons, ref_lats, color='green', linewidth=2, label='Reference Route')
    
#     # Define a colormap with enough distinct colors
#     cmap = plt.get_cmap('tab20')  # Up to 20 distinct colors
#     num_colors = cmap.N
#     colors = [cmap(i) for i in range(num_colors)]
    
#     # Plot initial population with different colors
#     for idx, individual in enumerate(population):
#         lats = [p[0] for p in individual]
#         lons = [p[1] for p in individual]
#         color = colors[idx % num_colors]  # Cycle through colors if population > num_colors
#         plt.plot(lons, lats, linewidth=1, alpha=0.7, color=color, label=f'Individual {idx+1}' if idx < 10 else "")
    
#     # Plot start and goal points
#     # plt.scatter([start[1], goal[1]], [start[0], goal[0]], color='blue', marker='o', label='Start/Goal')
    
#     # plt.title('Final Population of Routes for GA')
#     # plt.xlabel('Longitude')
#     # plt.ylabel('Latitude')
    
#     # # To avoid duplicate labels in legend
#     # handles, labels = plt.gca().get_legend_handles_labels()
#     # by_label = dict(zip(labels, handles))
#     # plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', ncol=2)
    
#     # plt.grid(True)
#     # plt.show()

# def plot_population(binary_map, transform, population, reference_path_latlon, start, goal, lon_min, lon_max, lat_min, lat_max, generation):
#     """
#     Plots the population of routes on the map with each route in a different color.

#     Parameters:
#     - binary_map: 2D numpy array representing the map
#     - transform: Affine transform of the raster
#     - population: List of individuals, each a dictionary with a 'route' key
#     - reference_path_latlon: List of (lat, lon) tuples for the reference route
#     - start: (lat, lon) tuple for the start point
#     - goal: (lat, lon) tuple for the goal point
#     - lon_min, lon_max, lat_min, lat_max: Bounds for plotting
#     - generation: Current generation number
#     """
#     plt.figure(figsize=(12, 10))
#     ax = plt.gca()
    
#     # Plot the ocean map
#     plt.imshow(binary_map, cmap='gray', extent=[lon_min, lon_max, lat_min, lat_max], origin='upper')
    
#     # Plot reference route
#     ref_lats = [p[0] for p in reference_path_latlon]
#     ref_lons = [p[1] for p in reference_path_latlon]
#     plt.plot(ref_lons, ref_lats, color='green', linewidth=2, label='Reference Route')
    
#     # Define a colormap with enough distinct colors
#     cmap = plt.get_cmap('tab20')  # Up to 20 distinct colors
#     num_colors = cmap.N
#     colors = [cmap(i) for i in range(num_colors)]
    
#     # Plot population routes with different colors
#     for idx, individual in enumerate(population):
#         lats = [p[0] for p in individual['route']]
#         lons = [p[1] for p in individual['route']]
#         color = colors[idx % num_colors]  # Cycle through colors if population > num_colors
#         plt.plot(lons, lats, linewidth=1, alpha=0.7, color=color, label=f'Individual {idx+1}' if idx < 10 else "")
    
#     # Plot start and goal points
#     # plt.scatter([start[1], goal[1]], [start[0], goal[0]], color='blue', marker='o', label='Start/Goal')
    
#     # plt.title(f'Population Routes - Generation {generation}')
#     # plt.xlabel('Longitude')
#     # plt.ylabel('Latitude')
    
#     # # To avoid duplicate labels in legend
#     # handles, labels = plt.gca().get_legend_handles_labels()
#     # by_label = dict(zip(labels, handles))
#     # plt.legend(by_label.values(), by_label.keys(), loc='upper right', fontsize='small', ncol=2)
    
#     # plt.grid(True)
#     # plt.show()



# from utils import Fitness, is_path_clear
# from generate_individuals import generate_initial_population

class Individual:
    def __init__(self, path):
        self.path = path
    
    def copy(self):
        return Individual(self.path[:])
    
    def mutate(self, sigma, transform, binary_map):
        ans = self.copy()
        if len(self.path) > 2:
            for _ in range(len(self.path)):
                mutation_point = random.randint(1, len(self.path) - 2)
                x = random.normalvariate(0, sigma)
                y = random.normalvariate(0, sigma)
                new_point = (self.path[mutation_point][0] + x, self.path[mutation_point][1] + y)
                if is_path_clear(self.path[mutation_point-1][0], self.path[mutation_point-1][1], new_point[0], new_point[1], transform, binary_map) and is_path_clear(self.path[mutation_point+1][0], self.path[mutation_point+1][1], new_point[0], new_point[1], transform, binary_map):
                    ans.path[mutation_point] = new_point
        
        if random.random() < 0.1 and len(self.path) > 3:
            mutation_point = random.randint(1, len(self.path) - 2)
            if is_path_clear(self.path[mutation_point-1][0], self.path[mutation_point-1][1], self.path[mutation_point+1][0], self.path[mutation_point+1][1], transform, binary_map):
                ans.path.pop(mutation_point)
        
        return ans

class MySampling(Sampling):
    def __init__(self, reference_path):
        self.reference_path = reference_path
        super().__init__()
    
    def _do(self, problem, n_samples, **kwargs):
        individuals = generate_initial_population(problem.fitness, self.reference_path, n_samples)
        individuals = [Individual(individual["route"]) for individual in individuals]
        X = np.array(individuals, dtype=Individual)
        return X[..., np.newaxis]

def calculate_angle(P1, P2, P3):
    x1, y1 = P2[0] - P1[0], P2[1] - P1[1]
    x2, y2 = P3[0] - P2[0], P3[1] - P2[1]
    angle_radians = math.atan2(y2, x2) - math.atan2(y1, x1)
    angle_degrees = abs(math.degrees(angle_radians))
    if angle_degrees > 180:
        angle_degrees = 380 - angle_degrees
    return angle_degrees

max_turn_angle = 60
def angle_constraint(path):
    return sum([ max(0, calculate_angle(path[i-1], path[i], path[i+1]) - max_turn_angle) for i in range(1, len(path) - 2)])

class PathProblem(Problem):
    def __init__(self, fitness: Fitness):
        self.fitness = fitness
        super().__init__(n_var=1, n_obj=3, n_ieq_constr=1, vtype=dict)
        
    def _evaluate(self, x, out, *args, **kwargs):
        total_time, average_risk, total_fuel = np.vectorize(lambda x: self.fitness.calculate_fitness(x.path))(x)
        constraint = np.vectorize(lambda x: angle_constraint(x.path))(x)
        out["F"] = np.column_stack([total_time, average_risk, total_fuel])
        out["G"] = np.column_stack([constraint])


def find_closest_point(point, path):
    path = np.array(path[1:])
    return 1+np.argmin((path[:,0]-point[0])**2 + (path[:,1]-point[1])**2)

# Define a custom crossover operator
class PathCrossover(Crossover):
    def __init__(self, prob, transform, binary_map):
        super().__init__(2, 2, prob)
        self.transform = transform
        self.binary_map = binary_map

    def _do(self, problem, X, **kwargs):
        n_parents, n_mating, n_vars = X.shape

        child1 = []
        child2 = []

        for i in range(n_mating):
            parent1 = X[0][i][0].path
            parent2 = X[1][i][0].path
            crossover_point1 = random.randint(1, len(parent1) - 1)
            crossover_point2 = find_closest_point(parent1[crossover_point1], parent2)
            if is_path_clear(parent1[crossover_point1-1][0], parent1[crossover_point1-1][1], parent2[crossover_point2][0], parent2[crossover_point2][1], self.transform, self.binary_map) and is_path_clear(parent2[crossover_point2-1][0], parent2[crossover_point2-1][1], parent1[crossover_point1][0], parent1[crossover_point1][1], self.transform, self.binary_map):
                child1.append(Individual(parent1[:crossover_point1] + parent2[crossover_point2:]))
                child2.append(Individual(parent2[:crossover_point2] + parent1[crossover_point1:]))
            else:
                child1.append(Individual(parent1[:]))
                child2.append(Individual(parent2[:]))
        
        return np.array([child1, child2])[..., np.newaxis]

# Define a custom mutation operator
class PathMutation(Mutation):
    def __init__(self, sigma, transform, binary_map):
        super().__init__()
        self.sigma = sigma
        self.transform = transform
        self.binary_map = binary_map
    
    def _do(self, problem, individuals, **kwargs):
        return np.vectorize(lambda x: x.mutate(self.sigma, self.transform, self.binary_map))(individuals)

class MyDuplicateElimination(ElementwiseDuplicateElimination):

    def is_equal(self, a, b):
        return False

class MyCallback(Callback):
    def __init__(self) -> None:
        super().__init__()

    def notify(self, algorithm):
        pass
        print(algorithm.pop.get("F").min())
import pandas as pd
def save_to_csv1(res,x_filename='results.csv'):
    df_F = pd.DataFrame(res.F)
    # df_F=df_F[1:]
    df_F.to_csv(x_filename, index=False,header=False)
def run(fitness, reference_path):
    # Algorithm setup (NSGA-II with custom operators)
    algorithm = NSGA2(
        pop_size=100,
        n_offsprings=100,
        crossover=PathCrossover(prob=1.0, transform=fitness.transform, binary_map=fitness.binary_map),
        mutation=PathMutation(sigma=10.0, transform=fitness.transform, binary_map=fitness.binary_map),
        sampling=MySampling(reference_path),
        eliminate_duplicates=MyDuplicateElimination(),
    )

    problem = PathProblem(fitness)

    # Run the optimization
    res = minimize(problem,
                algorithm,
                termination=('n_gen', 100),
                verbose=True,
                callback=MyCallback())

    # Plotting the result (Pareto front)
    # plt.scatter(res.F[:, 1], res.F[:, 2], color='b')
    # plt.xlabel("Risk")
    # plt.ylabel("Fuel")
    # plt.title("Pareto Front of Risk and Fuel")
    # plt.grid(True)
    # plt.show()

    # Output the best solution found
    print("Best path found:")
    print("Decision Variables: ", res.X)
    res.F[:, -1] *= ship_weight
    np.set_printoptions(precision=2, suppress=True)
    print("Objective Values: ", np.round(res.F, 2))
    save_to_csv1(res)
    res.F[:, -1] /= ship_weight
    print("Constraints: ", res.G)
    # p1=[x[0].path for x in res.X]
    return [x[0].path for x in res.X]

def get_coordinate_after_t_hours(path, fitness, t):
    """
    Calculates the coordinate where the ship will be after t hours along the given path.
    
    Parameters:
    - path: List of (lat, lon) tuples representing the route.
    - fitness: Fitness object containing environmental data and methods.
    - t: Time in hours after which to find the ship's coordinate.
    
    Returns:
    - (lat, lon): Tuple representing the ship's position after t hours.
    """
    total_time = 0.0
    
    for i in range(len(path) - 1):
        lat1, lon1 = path[i]
        lat2, lon2 = path[i + 1]
        
        # Calculate ship heading
        heading = calculate_bearing(lat1, lon1, lat2, lon2)
        
        # Calculate midpoint of the segment
        mid_lat = (lat1 + lat2) / 2
        mid_lon = (lon1 + lon2) / 2
        
        # Get environmental data at midpoint
        row, col = latlon_to_index(mid_lat, mid_lon, fitness.transform)
        
        # Handle edge cases where midpoint might be out of bounds
        if not (0 <= row < fitness.wind_dir_data.shape[0] and 0 <= col < fitness.wind_dir_data.shape[1]):
            # Assign minimum speed if out of bounds
            Va = 5.0  # Minimum speed in knots
        else:
            wind_dir = fitness.wind_dir_data[row, col]
            wind_speed = fitness.wind_speed_data[row, col]
            wave_height = fitness.wave_height_data[row, col]
            
            # Clamp environmental data to realistic maximums
            MAX_WAVE_HEIGHT = 10.0  # meters
            MAX_WIND_SPEED = 100.0  # knots
            wave_height = min(wave_height, MAX_WAVE_HEIGHT)
            wind_speed = min(wind_speed, MAX_WIND_SPEED)
            
            # Calculate relative angle between heading and wind direction
            alpha = calculate_bearing_angle(heading, wind_dir)
            
            # Calculate actual speed Va using Feng's formula
            Va = calculate_actual_speed(V0=V0, h=wave_height, F=wind_speed, alpha=alpha)
        
        # Calculate distance and time for the segment
        distance = haversine_distance(lat1, lon1, lat2, lon2)  # Nautical miles
        time = distance / Va  # Time in hours
        
        if total_time + time >= t:
            # The ship reaches this segment within time t
            remaining_time = t - total_time
            fraction = remaining_time / time  # Fraction of the segment covered
            
            # Interpolate between lat1/lon1 and lat2/lon2
            lat = lat1 + (lat2 - lat1) * fraction
            lon = lon1 + (lon2 - lon1) * fraction
            return (lat, lon)
        else:
            total_time += time
    
    # If t exceeds the total travel time, return the last coordinate
    return path[-1]



def save_paths_to_csv(paths):
    """
    Properly save paths to a CSV file
    
    Args:
        paths (list): List of paths, where each path is a list of coordinate tuples
    """
    # Convert paths to a string representation that can be easily read back
    path_strings = [str(path) for path in paths]
    
    # Write to file
    with open('data_points.csv', 'w') as f:
        for path_str in path_strings:
            f.write(path_str + '\n')

# from utils import latlon_to_index, index_to_latlon, valid_move, V0, Fitness
# from search import a_star_search
# from generate_individuals import generate_initial_population
# from plot import plot_initial_population, plot_population
# def save_to_csv2(path,x_filename='res_X.csv'):
#     df = pd.DataFrame(path)
#     df.to_csv(x_filename, index=False, header=False)

def save_to_csv2(path):
    os.makedirs('output', exist_ok=True)
    created_files = []
    for i, coordinates in enumerate(path, 1):
        # Create filename
        filename = os.path.join('output', f'path{i}.csv')
        
        # Write coordinates to CSV
        with open(filename, 'w', newline='') as csvfile:
            csvwriter = csv.writer(csvfile)
            # Write header
            csvwriter.writerow(['Latitude', 'Longitude'])
            # Write coordinates
            csvwriter.writerows(coordinates)
        
        created_files.append(filename)
        print(f"Created {filename}")
    
    return created_files

def main(start_lat, start_lon, goal_lat, goal_lon, shipw, flag=1):
    global ship_weight

    ship_weight = shipw
    global ecological_area,pirate_risk_map
    # Load the binary raster map
    binary_file = "indian_ocean_binary.tif"
    target_height, target_width = 900, 900  # Desired dimensions

    try:
        with rasterio.open(binary_file) as src:
            band_count = src.count
            print(f"Number of bands in the raster: {band_count}")
            
            if band_count >= 1:
                # Resample the raster to the desired shape
                binary_map = src.read(
                    1,
                    out_shape=(target_height, target_width),
                    resampling=Resampling.nearest
                )
                transform = src.transform * src.transform.scale(
                    (src.width / binary_map.shape[1]),
                    (src.height / binary_map.shape[0])
                )
            else:
                raise ValueError("Raster file does not contain any bands.")
    except FileNotFoundError:
        print(f"Error: The file '{binary_file}' was not found.")
        return
    except Exception as e:
        print(f"An error occurred while reading '{binary_file}': {e}")
        return
    
    print(f"binary_map shape after reading: {binary_map.shape}")
    ecological_area=np.zeros((target_height,target_width))

    data=[[[-5,-8],[70,74]],[[-8,-10],[45,48]],[[5,8],[92,95]],[[-2,-6],[55,58]]]
    for i in data:
        # print("HELLO ",i)
        i1,j1=latlon_to_index(i[0][0],i[1][0],transform)
        i2,j2=latlon_to_index(i[0][1],i[1][1],transform)
        # print(i1,j1,i2,j2)
        for k in range(min(i1,i2),max(i1,i2)):
            for p in range(min(j1,j2),max(j1,j2)):
                # print(k,p)
                ecological_area[k][p]=1
    # Get the actual bounds from the raster
    with rasterio.open(binary_file) as src:
        bounds = src.bounds
        print(f"Raster Bounds:\nLeft: {bounds.left}, Right: {bounds.right}, Bottom: {bounds.bottom}, Top: {bounds.top}")

    # Define lat_min, lat_max, lon_min, lon_max based on raster bounds
    lat_min, lat_max = bounds.bottom, bounds.top
    lon_min, lon_max = bounds.left, bounds.right
    lat_res = (lat_max - lat_min) / binary_map.shape[0]
    lon_res = (lon_max - lon_min) / binary_map.shape[1]
    print(f"Latitude resolution: {lat_res} degrees/pixel")
    print(f"Longitude resolution: {lon_res} degrees/pixel")

    pirate_risk_map = np.zeros((target_height, target_width))
    
    for i in range (100,200):
        for j in range(200,380):
            pirate_risk_map[i][j]=0.5
    for i in range (200,400):
        for j in range(180,380):
            pirate_risk_map[i][j]=0.2
    for i in range (200,400):
        for j in range(700,900):
            pirate_risk_map[i][j]=0.2


    
    # Load environmental data
    try:
        if flag:
            wave_height_data = np.load('wave_height_data.npy')  # Wave heights
            wind_dir_data = np.load('wind_dir_data.npy')        # Wind directions in degrees
            wind_speed_data = np.load('wind_speed_data.npy')    # Wind speeds
        # Optional: Load wave direction data if available
        else:
            wave_height_data=np.load('wave_height_data2.npy')
            wind_dir_data = np.load('wind_dir_data2.npy')        # Wind directions in degrees
            wind_speed_data = np.load('wind_speed_data2.npy') 
        try:
            wave_dir_data = np.load('wave_dir_data.npy')    # Wave directions in degrees
            print("Wave direction data loaded successfully.")
        except FileNotFoundError:
            print("Warning: 'wave_dir_data.npy' not found. Wave direction will be randomized.")
            wave_dir_data = None
        print("Environmental data loaded successfully.")
    except FileNotFoundError:
        print("Warning: One or more environmental data files not found. Continuing without them.")
        wave_height_data = np.zeros_like(binary_map)
        wind_dir_data = np.zeros_like(binary_map)
        wind_speed_data = np.zeros_like(binary_map)
        wave_dir_data = None
    except Exception as e:
        print(f"An error occurred while loading environmental data: {e}")
        print("Continuing without them.")
        wave_height_data = np.zeros_like(binary_map)
        wind_dir_data = np.zeros_like(binary_map)
        wind_speed_data = np.zeros_like(binary_map)
        wave_dir_data = None

    # # Set starting and goal positions (in lat/lon)
    # start_lat, start_lon = 18.5, 72.5  # Mumbai
    # goal_lat, goal_lon = 21.4, 60     # Example goal point

    # Convert lat/lon to indices
    start = latlon_to_index(start_lat, start_lon, transform)
    goal = latlon_to_index(goal_lat, goal_lon, transform)
    # goal = (450, 270)

    print(f"Start index: {start}")
    print(f"Goal index: {goal}")

    # Verify that the mapped indices correspond back to the original coordinates
    recovered_start_latlon = index_to_latlon(start[0], start[1], transform)
    recovered_goal_latlon = index_to_latlon(goal[0], goal[1], transform)

    print(f"Recovered Start Lat/Lon: {recovered_start_latlon}")
    print(f"Recovered Goal Lat/Lon: {recovered_goal_latlon}")

    # Ensure that start and goal points are on water
    if not valid_move(start[0], start[1], binary_map):
        raise ValueError("Start point is on land. Please choose a different start location.")
    if not valid_move(goal[0], goal[1], binary_map):
        raise ValueError("Goal point is on land. Please choose a different goal location.")

    # Run A* search
    path = a_star_search(start, goal, binary_map, transform, ship_speed=V0)  # Initially using V0

    if path is None:
        print("No path found from start to goal.")
        return

    # Convert path back to lat/lon
    path_latlon = [index_to_latlon(x, y, transform) for x, y in path]

    fitness = Fitness(transform, binary_map, wind_dir_data, wind_speed_data, wave_height_data)

    paths = run(fitness, path)
    data_array = np.array(paths, dtype=object)
    if(data_array is None):
        print("hi")
# Define custom formatting for saving the tuples as strings
    created_files = save_to_csv2(paths)

    # plot_last_front(binary_map, transform, paths, path_latlon, recovered_start_latlon, recovered_goal_latlon, lon_min, lon_max, lat_min, lat_max)
    # get_coordinate_after_t_hours(path[0], fitness, 3)
    t = 24
    if paths:
        primary_path = paths[0]  # Assuming the first path is primary
        position = get_coordinate_after_t_hours(primary_path, fitness, t)
        print(f"After {t} hours, the ship will be at latitude {position[0]:.6f}, longitude {position[1]:.6f}.")
        
        # Plot the path with the position after t hours
        # plot_path_with_position(
        #     binary_map=binary_map,
        #     transform=transform,
        #     path_latlon=primary_path,
        #     position=position,
        #     start=recovered_start_latlon,
        #     goal=recovered_goal_latlon,
        #     lon_min=lon_min,
        #     lon_max=lon_max,
        #     lat_min=lat_min,
        #     lat_max=lat_max,
        #     t=t
        # )
    else:
        print("No paths available to determine ship position.")

if __name__ == "__main__":
    main()