import os
import subprocess
import json
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import filter_csv
import filter_csv_nsga
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
            "hull_eff", "prop_eff", "engine_eff", "c_sfoc", "shipw"
        ]

        # Validate required parameters
        missing_params = [param for param in required_params if data.get(param) is None]
        if missing_params:
            return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400

        # Define static filenames
        original_output_files = [
            'path_fuel.csv', 'path_safe.csv', 'path_short.csv','data_points.csv', 'results.csv'
        ]
        smoothed_output_files = [
            'path_fuel_smoothed.csv', 'path_safe_smoothed.csv', 'path_short_smoothed.csv', 'results.csv'
        ]
        save_folder = "C:/Users/Admin/Desktop/SamudraPath/SamudraPath/public"
        
        # Specific files to remove
        files_to_remove = [
            'path_fuel.csv', 'path_safe.csv', 'path_short.csv',
            'path_fuel_smoothed.csv', 'path_safe_smoothed.csv', 'path_short_smoothed.csv'
            'data_points.csv', 'results.csv'
        ]

        # Selectively remove only route-related files
        for filename in files_to_remove:
            file_path = os.path.join(save_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        
        from new_algorithm import main
            
        try:
            results = main(
                start_lat=data['start_lat'],
                start_lon=data['start_lon'],
                goal_lat=data['goal_lat'],
                goal_lon=data['goal_lon'],
                ship_speed=data['ship_speed'],
                ship_dis=data['ship_dis'],
                area_front=data['area_front'],
                ship_height=data['ship_height'],
                ship_reso=data['ship_reso'],
                hull_eff=data['hull_eff'],
                prop_eff=data['prop_eff'],
                engine_eff=data['engine_eff'],
                c_sfoc=data['c_sfoc']
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        

        from nsga import main
        try:
            main(
                start_lat=data["start_lat"],
                start_lon=data["start_lon"],
                goal_lat=data["goal_lat"],
                goal_lon=data["goal_lon"],
                shipw=data["shipw"]
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400


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

            dynamic_files = [file for file in os.listdir() if file.startswith('path_') and not file.endswith('_smoothed.csv')]
            for i, file in enumerate(dynamic_files+1):
                smoothed_route = filter_csv.process_route(file, epsilon=0.001, window_size=3)
                smoothed_file_name = f"path_{i}_smoothed.csv"
                filter_csv.save_to_csv(smoothed_route, smoothed_file_name)

                # Copy smoothed dynamic file to the save folder
                dest_path = os.path.join(save_folder, smoothed_file_name)
                shutil.copy2(smoothed_file_name, dest_path)

        except Exception as e:
            return jsonify({"error": f"Route smoothing failed: {str(e)}"}), 500

        # Verify smoothed output files
        for file in smoothed_output_files:
            if os.path.exists(file):
                dest_path = os.path.join(save_folder, os.path.basename(file))
                shutil.copy2(file, dest_path)
                os.remove(file)  # Remove the file from the output directory

# Also, remove the dynamic smoothed files
        dynamic_smoothed_files = [file for file in os.listdir() if file.startswith('path_') and file.endswith('_smoothed.csv')]
        for file in dynamic_smoothed_files:
            os.remove(file)

        # Return a success response
        return jsonify(results), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500
    

    

@app.route('/new_positon', methods=['POST'])
def new_position():
    try:
        data = request.json

        required_params = [
            "start_lat", "start_lon", "goal_lat", "goal_lon", 
            "shipw", "flag"
        ]

        # Validate required parameters
        missing_params = [param for param in required_params if data.get(param) is None]
        if missing_params:
            return jsonify({"error": f"Missing parameters: {', '.join(missing_params)}"}), 400


        # Call the main calculation function and get the new position
         # Ensure the main function is in calculate_position.py

        original_output_files = [
            
        ]
        save_folder = "C:/Users/Admin/Desktop/SamudraPath/SamudraPath/public"
        
        # Specific files to remove
        files_to_remove = [
            
        ]

        # Selectively remove only route-related files
        for filename in files_to_remove:
            file_path = os.path.join(save_folder, filename)
            if os.path.exists(file_path):
                os.remove(file_path)

        # Ensure the save folder exists
        os.makedirs(save_folder, exist_ok=True)

        from nsga import main
        try:
            main(
                start_lat=data["start_lat"],
                start_lon=data["start_lon"],
                goal_lat=data["goal_lat"],
                goal_lon=data["goal_lon"],
                shipw=data["shipw"],
                flag=data['flag']
            )
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        

        if not all(os.path.exists(file) for file in original_output_files):
            return jsonify({"error": "Route smoothing failed. Smoothed files not created."}), 500

        # Copy all output files to the specified folder
        for file in original_output_files:
            if os.path.exists(file):
                dest_path = os.path.join(save_folder, os.path.basename(file))
                shutil.copy2(file, dest_path)

        # Return the new position to the frontend
        return jsonify({
            "message": "Route position calculation completed successfully"
        }), 200

    except Exception as e:
        return jsonify({"error": f"An unexpected error occurred: {str(e)}"}), 500


if __name__ == '__main__':
    app.run(debug=True)