import React from "react";
import Navbar from "../components/Homepage/Navbar"; // Assuming Navbar is in the same directory
import { CircularProgressbarWithChildren, buildStyles } from 'react-circular-progressbar'; // Import gauge meter
import 'react-circular-progressbar/dist/styles.css';

const VoyageDetails = () => {
  // Sample data for the table (this can be dynamic or fetched from an API)
  const tableData = [
    {
      parameter: "Safety Index",
      safePathValue: 85,
      fuelEfficientPathValue: 70,
      shortestPathValue: 60,
    },
    {
      parameter: "Fuel Consumption",
      safePathValue: 55,
      fuelEfficientPathValue: 80,
      shortestPathValue: 65,
    },
    {
      parameter: "Distance Covered",
      safePathValue: 75,
      fuelEfficientPathValue: 70,
      shortestPathValue: 90,
    },
  ];

  // Function to determine color based on values
  const getColor = (value) => {
    if (value > 80) return 'green';
    if (value > 50) return 'orange';
    return 'red';
  };

  // Sample probable location after 6 hours (could be dynamic based on ship's route and speed)
  const probableLocation = {
    latitude: 19.0760, // Example coordinates (Mumbai)
    longitude: 72.8777,
    city: "Mumbai, India",
  };

  return (
    <div className="h-screen bg-gray-100">
      <Navbar /> {/* Reusing the Navbar component */}

      <div className="container mx-auto p-6">
        {/* Ship Details Section */}
        <div className="bg-white p-4 rounded-md shadow-md mb-6">
          <h2 className="text-2xl font-bold text-teal-600">Ship Details</h2>
          <div className="mt-4">
            <p><strong>Ship Name:</strong> Samudra 01</p>
            <p><strong>Ship Type:</strong> Cargo Ship</p>
            <p><strong>Current Location:</strong> Mumbai Port</p>
            {/* Add more details as required */}
          </div>
        </div>

        {/* Table Section with Health Barometers (Gauge Meters) */}
        <div className="bg-white p-4 rounded-md shadow-md mb-6">
          <table className="min-w-full table-auto border-collapse">
            <thead>
              <tr className="bg-teal-600 text-white">
                <th className="px-6 py-3 text-left">Parameter</th>
                <th className="px-6 py-3 text-left">Safe Path</th>
                <th className="px-6 py-3 text-left">Fuel Efficient Path</th>
                <th className="px-6 py-3 text-left">Shortest Path</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((row, index) => (
                <tr key={index}>
                  <td className="px-6 py-3 text-left">{row.parameter}</td>
                  <td className="px-6 py-3 text-center">
                    <div className="relative w-24 h-24">
                      <CircularProgressbarWithChildren
                        value={row.safePathValue}
                        maxValue={100}
                        styles={buildStyles({
                          pathColor: getColor(row.safePathValue),
                          trailColor: '#ddd',
                          strokeLinecap: 'round',
                          rotation: 0.75,
                          strokeWidth: 10,
                        })}
                      >
                        <div className="text-center">
                          <strong className="text-xl font-semibold">{row.safePathValue}%</strong>
                        </div>
                      </CircularProgressbarWithChildren>
                    </div>
                  </td>
                  <td className="px-6 py-3 text-center">
                    <div className="relative w-24 h-24">
                      <CircularProgressbarWithChildren
                        value={row.fuelEfficientPathValue}
                        maxValue={100}
                        styles={buildStyles({
                          pathColor: getColor(row.fuelEfficientPathValue),
                          trailColor: '#ddd',
                          strokeLinecap: 'round',
                          rotation: 0.75,
                          strokeWidth: 10,
                        })}
                      >
                        <div className="text-center">
                          <strong className="text-xl font-semibold">{row.fuelEfficientPathValue}%</strong>
                        </div>
                      </CircularProgressbarWithChildren>
                    </div>
                  </td>
                  <td className="px-6 py-3 text-center">
                    <div className="relative w-24 h-24">
                      <CircularProgressbarWithChildren
                        value={row.shortestPathValue}
                        maxValue={100}
                        styles={buildStyles({
                          pathColor: getColor(row.shortestPathValue),
                          trailColor: '#ddd',
                          strokeLinecap: 'round',
                          rotation: 0.75,
                          strokeWidth: 10,
                        })}
                      >
                        <div className="text-center">
                          <strong className="text-xl font-semibold">{row.shortestPathValue}%</strong>
                        </div>
                      </CircularProgressbarWithChildren>
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>

        {/* Probable Location After 6 Hours */}
        <div className="bg-white p-4 rounded-md shadow-md mt-6">
          <h2 className="text-2xl font-bold text-teal-600">Probable Location After 6 Hours</h2>
          <div className="mt-4">
            <p><strong>Location:</strong> {probableLocation.city}</p>
            <p><strong>Latitude:</strong> {probableLocation.latitude}</p>
            <p><strong>Longitude:</strong> {probableLocation.longitude}</p>
            {/* In future, you can integrate a map to show the location visually */}
          </div>
        </div>
      </div>
    </div>
  );
};

export default VoyageDetails;
