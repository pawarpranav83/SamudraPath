import React from "react";
import Navbar from "../components/Homepage/Navbar";
import { CircularProgressbarWithChildren, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';
import { BarChart, Bar, XAxis, YAxis, Tooltip, Legend, ResponsiveContainer } from "recharts";

const VoyageDetails = () => {
  const tableData = [
    {
      parameter: "Safety Index",
      safePathValue: 85,
      fuelEfficientPathValue: 70,
      shortestPathValue: 60,
    },
  ];

  const barGraphData = [
    { path: "Safe Path", fuel: 40, time: 30 },
    { path: "Fuel Efficient Path", fuel: 25, time: 35 },
    { path: "Shortest Path", fuel: 80, time: 20 },
  ];

  const getColor = (value) => {
    if (value > 80) return "green";
    if (value > 50) return "orange";
    return "red";
  };

  return (
    <div className="h-screen bg-gray-100">
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

        {/* Safety Metrics Section */}
        <div className="bg-white p-4 rounded-md shadow-md mb-6">
          <h2 className="text-2xl font-bold text-teal-600 mb-4">Safety Metrics</h2>
          <table className="min-w-full table-auto border-collapse">
            <thead>
              <tr className="bg-teal-600 text-white">
                <th className="px-6 py-3 text-left">Parameter</th>
                <th className="px-6 py-3 text-left">Risk</th>
                <th className="px-6 py-3 text-left">Safety Index</th>
              </tr>
            </thead>
            <tbody>
              {tableData.map((data, index) => (
                [
                  {
                    path: "Safe Path",
                    value: data.safePathValue,
                    color: getColor(data.safePathValue),
                  },
                  {
                    path: "Fuel Efficient Path",
                    value: data.fuelEfficientPathValue,
                    color: getColor(data.fuelEfficientPathValue),
                  },
                  {
                    path: "Shortest Path",
                    value: data.shortestPathValue,
                    color: getColor(data.shortestPathValue),
                  },
                ].map((item, idx) => (
                  <tr key={`${index}-${idx}`} className="odd:bg-gray-50 even:bg-gray-100">
                    <td className="px-6 py-4 text-left font-medium">{item.path}</td>
                    <td className="px-6 py-4 text-left font-medium">{item.value}%</td>
                    <td className="px-6 py-4 text-left">
                      <div className="w-16 h-16">
                        <CircularProgressbarWithChildren
                          value={item.value}
                          maxValue={100}
                          styles={buildStyles({
                            pathColor: item.color,
                            trailColor: "#ddd",
                            strokeLinecap: "round",
                            strokeWidth: 10,
                          })}
                        >
                          <div className="text-center">
                            <strong className="text-sm font-semibold">
                              {item.value}%
                            </strong>
                          </div>
                        </CircularProgressbarWithChildren>
                      </div>
                    </td>
                  </tr>
                ))
              ))}
            </tbody>
          </table>
        </div>

        {/* Path Performance Metrics Section */}
        <div className="bg-white p-4 rounded-md shadow-md mb-6 ml-4">
          <h2 className="text-2xl font-bold text-teal-600 mb-4">
            Path Performance Metrics
          </h2>
          <div className="flex flex-col space-y-6">
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
