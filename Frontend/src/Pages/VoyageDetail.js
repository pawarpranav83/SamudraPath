import React, { useContext } from "react";
import Navbar from "../components/Homepage/Navbar";
import 'react-circular-progressbar/dist/styles.css';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";
import { ShipContext } from "../ShipContext";

const VoyageDetails = () => {
  const { routeData } = useContext(ShipContext)



  const tableData = [
    {
      parameter: "Safety Index",
      safePathValue: routeData?.safest_path?.risk || 0,
      fuelEfficientPathValue: routeData?.fuel_efficient_path?.risk || 0,
      shortestPathValue: routeData?.shortest_path?.risk || 0,
    },
  ];

  const barGraphData = [
    { path: "Safe Path", fuel: routeData?.safest_path?.fuel || 0, time: routeData?.safest_path?.time || 0, risk: routeData?.safest_path?.risk || 0 },
    { path: "Fuel Efficient Path", fuel: routeData?.fuel_efficient_path?.fuel || 0, time: routeData?.fuel_efficient_path?.time || 0, risk: routeData?.fuel_efficient_path?.risk || 0 },
    { path: "Shortest Path", fuel: routeData?.shortest_path?.fuel || 0, time: routeData?.shortest_path?.time || 0, risk: routeData?.shortest_path?.risk || 0 },
  ];

  const getColor = (value) => {
    if (value > 80) return "green";
    if (value > 50) return "orange";
    return "red";
  };

  return (
    <div className="min-h-screen bg-gray-100">
      <Navbar />

      <div className="container mx-auto p-6">
        {/* Ship Details Section */}
        <div className="bg-white p-4 rounded-md shadow-md mb-6">
          <h2 className="text-2xl font-bold text-teal-600">Ship Details</h2>
          <div className="mt-4">
            <p><strong>Ship Name:</strong> Samudra 01</p>
            <p><strong>Ship Type:</strong> Cargo Ship</p>
            <p><strong>Current Location:</strong> Mumbai Port</p>
          </div>
        </div>        

        {/* Path Performance Metrics Section */}
        <div className="bg-white p-4 rounded-md shadow-md mb-6 ml-4">
          <h2 className="text-2xl font-bold text-teal-600 mb-4">
            Path Performance Metrics
          </h2>
          <div className="flex flex-col space-y-6">
            <div>
              <h3 className="text-lg font-semibold text-gray-700 mb-2 text-center">
                Risk Comparison
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={barGraphData}
                  layout="vertical"
                  margin={{ top: 10, right: 20, bottom: 10, left: 40 }}
                >
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="path" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="risk" fill="#c1121f" name="Risk" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Fuel Graph */}
            <div>
              <h3 className="text-lg font-semibold text-gray-700 mb-2 text-center">
                Fuel Comparison
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={barGraphData}
                  layout="vertical"
                  margin={{ top: 10, right: 20, bottom: 10, left: 40 }}
                >
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="path" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="fuel" fill="#4caf50" name="Fuel (Gal)" />
                </BarChart>
              </ResponsiveContainer>
            </div>

            {/* Time Graph */}
            <div>
              <h3 className="text-lg font-semibold text-gray-700 mb-2 text-center">
                Time Comparison
              </h3>
              <ResponsiveContainer width="100%" height={300}>
                <BarChart
                  data={barGraphData}
                  layout="vertical"
                  margin={{ top: 10, right: 20, bottom: 10, left: 40 }}
                >
                  <XAxis type="number" />
                  <YAxis type="category" dataKey="path" />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="time" fill="#2196f3" name="Time (Hrs)" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        </div>

        {/* Probable Location After 24 Hours Section */}
        <div className="bg-white p-4 rounded-md shadow-md">
          <h2 className="text-2xl font-bold text-teal-600">Probable Location After 24 Hours</h2>
          <div className="mt-4">
            <p><strong>Expected Location:</strong> Kochi Port</p>
            <p><strong>Latitude:</strong> 48</p>
            <p><strong>Longitude:</strong> 14</p>
          </div>
        </div>
      </div>
    </div>
  );
};

export default VoyageDetails;