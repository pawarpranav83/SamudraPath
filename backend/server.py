import os
import subprocess
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import filter_csv
import shutil

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})  # Allow all origins

@app.route('/calculate_route', methods=['POST'])
def calculate_route():
    try:
        data = request.json
        required_params = [
            "start_lat", "start_lon", "goal_lat", "goal_lon", 
            "ship_speed", "ship_height", "ship_dis", "area_front", "ship_reso", 
            "hull_eff", "prop_eff", "engine_eff", "c_sfoc"
        ]

        # Validate required parameters
        missing_params = [param for param in required_params if data.get(param) is None]
        if missing_params:
            return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400

        # Define static filenames
        original_output_files = [
            'path_fuel.csv', 'path_safe.csv', 'path_short.csv', 'path_weighted.csv',
            'wind_speed_map.svg', 'wave_height_map.svg', 'usurf_map.svg', 'vsurf_map.svg'
        ]
        smoothed_output_files = [
            'path_fuel_smoothed.csv', 'path_safe_smoothed.csv', 'path_short_smoothed.csv', 'path_weighted_smoothed.csv'
        ]
        svg_files = [
            'wind_speed_map.svg', 'wave_height_map.svg', 'usurf_map.svg', 'vsurf_map.svg'
        ]
        save_folder = "C:/Users/Admin/Desktop/SamudraPath/SamudraPath/public"
        
        # Specific files to remove
        files_to_remove = [
            'path_fuel.csv', 'path_safe.csv', 'path_short.csv', 'path_weighted.csv',
            'path_fuel_smoothed.csv', 'path_safe_smoothed.csv', 'path_short_smoothed.csv', 'path_weighted_smoothed.csv',
            'wind_speed_map.svg', 'wave_height_map.svg', 'usurf_map.svg', 'vsurf_map.svg'
        ]

        # Selectively remove only route-related files
        for filename in files_to_remove:
            file_path = os.path.join(save_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Determine which script to run based on user weights
        python_exec = os.path.abspath('.venv/Scripts/python')
        
        algorithm_script = os.path.abspath('algorithm.py')
            
        command = [
            python_exec, algorithm_script,
            str(data['start_lat']), str(data['start_lon']),
            str(data['goal_lat']), str(data['goal_lon']),
            str(data['ship_speed']), str(data['ship_dis']), str(data['ship_height']),
            str(data['area_front']), str(data['ship_reso']),
            str(data['hull_eff']), str(data['prop_eff']),
            str(data['engine_eff']), str(data['c_sfoc'])
        ]

        try:
            # Run the subprocess and capture output
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or str(e)
            return jsonify({"error": f"Algorithm execution failed: {error_msg}"}), 500

        # Verify output files
        if not all(os.path.exists(file) for file in original_output_files):
            return jsonify({"error": "Route calculation failed. One or more output files not found."}), 500

        # Smooth routes using filter_csv module
        try:
            smoothed_route_fuel = filter_csv.process_route("path_fuel.csv", epsilon=0.001, window_size=3)
            filter_csv.save_to_csv(smoothed_route_fuel, "path_fuel_smoothed.csv")

                # Process and save route for safe
            smoothed_route_safe = filter_csv.process_route("path_safe.csv", epsilon=0.001, window_size=3)
            filter_csv.save_to_csv(smoothed_route_safe, "path_safe_smoothed.csv")

                # Process and save route for short
            smoothed_route_short = filter_csv.process_route("path_short.csv", epsilon=0.001, window_size=3)
            filter_csv.save_to_csv(smoothed_route_short, "path_short_smoothed.csv")

            smoothed_route_weighted = filter_csv.process_route("path_weighted.csv", epsilon=0.001, window_size=3)
            filter_csv.save_to_csv(smoothed_route_weighted, "path_weighted_smoothed.csv")

        except Exception as e:
            return jsonify({"error": f"Route smoothing failed: {str(e)}"}), 500

        # Verify smoothed output files
        if not all(os.path.exists(file) for file in smoothed_output_files):
            return jsonify({"error": "Route smoothing failed. Smoothed files not created."}), 500

        # Copy all output files to the specified folder
        for file in svg_files + smoothed_output_files:
            if os.path.exists(file):
                dest_path = os.path.join(save_folder, os.path.basename(file))
                shutil.copy2(file, dest_path)

        # Return a success response
        return jsonify({
            "message": "Route calculation completed successfully. Files saved to specified folder.",
            "files": [os.path.basename(file) for file in svg_files + smoothed_output_files]
        }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    
@app.route('/calculate_new_position', methods=['POST'])
def calulate_weighted_route():
    try:
        data = request.json
        required_params = [
            "start_lat", "start_lon", "goal_lat", "goal_lon", 
            "ship_speed", "ship_height", "ship_dis", "area_front", "ship_reso", 
            "hull_eff", "prop_eff", "engine_eff", "c_sfoc","weight_shortest", "weight_safest", "weight_fuel"
        ]

        # Validate required parameters
        missing_params = [param for param in required_params if data.get(param) is None]
        if missing_params:
            return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400

        # Define static filenames
        output_files = [
            'path_weighted.csv'
        ]
        smoothed_output_files = [
            'path_weighted_smoothed.csv'
        ]
        save_folder = "C:/Users/Admin/Desktop/SamudraPath/SamudraPath/public"
        
        # Specific files to remove
        files_to_remove = [
            'path_weighted.csv','path_weighted_smoothed.csv'
        ]

        # Selectively remove only route-related files
        for filename in files_to_remove:
            file_path = os.path.join(save_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Determine which script to run based on user weights
        python_exec = os.path.abspath('.venv/Scripts/python')
        
        algorithm_script = os.path.abspath('weighted_algorithm.py')
            
        command = [
            python_exec, algorithm_script,
            str(data['start_lat']), str(data['start_lon']),
            str(data['goal_lat']), str(data['goal_lon']),
            str(data['ship_speed']), str(data['ship_dis']), str(data['ship_height']),
            str(data['area_front']), str(data['ship_reso']),
            str(data['hull_eff']), str(data['prop_eff']),
            str(data['engine_eff']), str(data['c_sfoc']),                
            str(data['weight_shortest']),
            str(data['weight_safest']),
            str(data['weight_fuel'])
        ]

        try:
            # Run the subprocess and capture output
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or str(e)
            return jsonify({"error": f"Algorithm execution failed: {error_msg}"}), 500

        # Verify output files
        if not all(os.path.exists(file) for file in output_files):
            return jsonify({"error": "Route calculation failed. One or more output files not found."}), 500

        # Smooth routes using filter_csv module
        try:
            smoothed_route_weighted = filter_csv.process_route("path_weighted.csv", epsilon=0.001, window_size=3)
            filter_csv.save_to_csv(smoothed_route_weighted, "path_weighted_smoothed.csv")

        except Exception as e:
            return jsonify({"error": f"Route smoothing failed: {str(e)}"}), 500

        # Verify smoothed output files
        if not all(os.path.exists(file) for file in smoothed_output_files):
            return jsonify({"error": "Route smoothing failed. Smoothed files not created."}), 500

        # Copy all output files to the specified folder
        for file in smoothed_output_files:
            if os.path.exists(file):
                dest_path = os.path.join(save_folder, os.path.basename(file))
                shutil.copy2(file, dest_path)

        # Return a success response
        return jsonify({
            "message": "Route calculation completed successfully. Files saved to specified folder.",
            "files": [os.path.basename(file) for file in smoothed_output_files]
        }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500

    

@app.route('/calculate_new_position', methods=['POST'])
def calculate_new_position():
    try:
        data = request.json
        required_params = [
            "start_lat", "start_lon", "goal_lat", "goal_lon", 
            "ship_speed", "ship_height", "ship_dis", "area_front", "ship_reso", 
            "hull_eff", "prop_eff", "engine_eff", "c_sfoc", "path"
        ]

        # Validate required parameters
        missing_params = [param for param in required_params if data.get(param) is None]
        if missing_params:
            return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400

        # Assign default weights if not provided
        weight_shortest = data.get("weight_shortest", 0.25)
        weight_safest = data.get("weight_safest", 0.375)
        weight_fuel = data.get("weight_fuel", 0.375)

        # Define static filenames
        output_files = ['new_position.csv']
        save_folder = "C:/Users/Admin/Desktop/SamudraPath/SamudraPath/public"
        
        # Specific files to remove
        files_to_remove = [
            'new_position.csv'
        ]

        # Selectively remove only route-related files
        for filename in files_to_remove:
            file_path = os.path.join(save_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        # Determine which script to run based on user weights
        python_exec = os.path.abspath('.venv/Scripts/python')
        algorithm_script = os.path.abspath('calculate_position.py')
            
        command = [
                python_exec, algorithm_script,
                str(data['start_lat']), str(data['start_lon']),
                str(data['goal_lat']), str(data['goal_lon']),
                str(data['ship_speed']), str(data['ship_dis']), str(data['ship_height']),
                str(data['area_front']), str(data['ship_reso']),
                str(data['hull_eff']), str(data['prop_eff']),
                str(data['engine_eff']), str(data['c_sfoc']),  
                str(data['path']),              
                str(weight_shortest),
                str(weight_safest),
                str(weight_fuel),
        ]
        try:
            # Run the subprocess and capture output
            result = subprocess.run(command, check=True)
        except subprocess.CalledProcessError as e:
            error_msg = e.stderr or e.stdout or str(e)
            return jsonify({"error": f"Algorithm execution failed: {error_msg}"}), 500

        # Verify output files
        if not all(os.path.exists(file) for file in output_files):
            return jsonify({"error": "Route calculation failed. One or more output files not found."}), 500

        # Copy all output files to the specified folder
        for file in output_files:
            if os.path.exists(file):
                dest_path = os.path.join(save_folder, os.path.basename(file))
                shutil.copy2(file, dest_path)

        # Return a success response
        return jsonify({
            "message": "New position calculation completed successfully",
            "files": [os.path.basename(file) for file in output_files]
        }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    


if __name__ == '__main__':
    app.run(debug=True)