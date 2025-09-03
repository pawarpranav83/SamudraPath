import React from 'react';
import { CircularProgressbarWithChildren, buildStyles } from 'react-circular-progressbar';
import 'react-circular-progressbar/dist/styles.css';

const HealthBarometer = ({ riskLevel, fuelLevel }) => {
  const color = riskLevel > 8 ? 'red' : riskLevel > 5 ? 'orange' : 'green';
  const fuelColor = fuelLevel > 50 ? 'green' : fuelLevel > 20 ? 'orange' : 'red';

  return (
    <div className="bg-white shadow-lg rounded-xl p-8 mb-8 border border-gray-300 w-full">
      <h3 className="text-3xl font-semibold text-indigo-600 mb-6 border-b-2 border-indigo-500 pb-2">Health Meter</h3>

      {/* Container for the side-by-side layout */}
      <div className="flex justify-between space-x-10">
        {/* Health Barometer Section */}
        <div className="flex flex-col items-center w-1/4">
          {/* Circular Progress Bar for Risk Level */}
          <div className="relative w-64 h-64 mb-4">
            <CircularProgressbarWithChildren
              value={riskLevel}
              maxValue={10}
              styles={buildStyles({
                pathColor: color,
                trailColor: '#ddd',
                strokeLinecap: 'round',
                rotation: 0.75,
                strokeWidth: 30,
              })}
            >
              <div className="text-center">
                <strong className="text-xl font-semibold">Risk Level: {riskLevel.toFixed(2)}</strong>
                <p className="text-lg font-bold mt-2">{riskLevel < 4 ? 'LOW' : riskLevel < 7 ? 'MODERATE' : 'HIGH'}</p>
              </div>
            </CircularProgressbarWithChildren>
          </div>
        </div>

        {/* Maintenance & Location Details */}
        <div className="flex flex-col items-center w-1/4">
          <h4 className="text-xl font-semibold text-indigo-600 mb-4">Maintenance & Location Details</h4>
          <ul className="space-y-2 text-lg text-gray-800">
            <li><strong className="font-medium text-indigo-600">Maintenance Type:</strong> Routine</li>
            <li><strong className="font-medium text-indigo-600">Parts Replaced:</strong> None</li>
            <li><strong className="font-medium text-indigo-600">Next Scheduled Maintenance:</strong> 3 days to go</li>
          </ul>
        </div>

        {/* Fuel Level Section */}
        <div className="flex flex-col items-center w-1/4">
          <div className="relative w-64 h-64 mb-4">
            <CircularProgressbarWithChildren
              value={fuelLevel}
              maxValue={100}
              styles={buildStyles({
                pathColor: fuelColor,
                trailColor: '#ddd',
                strokeLinecap: 'round',
                rotation: 0.75,
                strokeWidth: 30,
              })}
            >
              <div className="text-center">
                <strong className="text-xl font-semibold">Fuel Level: {fuelLevel}%</strong>
              </div>
            </CircularProgressbarWithChildren>
          </div>
        </div>

        {/* Nearby Assistance Details */}
        <div className="flex flex-col items-center w-1/4">
          <h4 className="text-xl font-semibold text-indigo-600 mb-4">Nearby Assistance</h4>
          <ul className="space-y-2 text-lg text-gray-800">
            <li><strong className="font-medium text-indigo-600">Current Position:</strong> 28.0330° N, 125.5654° E</li>
            <li><strong className="font-medium text-indigo-600">Latitude:</strong> 28.0330° N</li>
            <li><strong className="font-medium text-indigo-600">Longitude:</strong> 125.5654° E</li>
            <li><strong className="font-medium text-indigo-600">Nearby City:</strong> Taiwan</li>
            <li><strong className="font-medium text-indigo-600">Nearby Port:</strong> Taiwan International Port</li>
            <li><strong className="font-medium text-indigo-600">Nearby Ship:</strong> Evergreen fleet 2</li>
            <li><strong className="font-medium text-indigo-600">Military Ship:</strong> INS Vikramaditya, IN</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default HealthBarometer;
