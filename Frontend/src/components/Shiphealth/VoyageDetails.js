import React from 'react';

const VoyageDetails = () => {
  return (
    <div className="relative bg-white shadow-lg rounded-xl p-6 mb-8 border border-gray-300 max-w-xs w-full mx-auto">
      <h3 className="text-3xl font-semibold text-indigo-600 mb-4 border-b-2 border-indigo-500 pb-2">
        Voyage Details
      </h3>

      <div className="space-y-4">
        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Current Position:</strong>
          <p className="ml-4">28.0330° N, 125.5654° E</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Latitude:</strong>
          <p className="ml-4">28.0330° N</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Longitude:</strong>
          <p className="ml-4">125.5654° E</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Last Updated:</strong>
          <p className="ml-4">15 min Ago</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Navigation:</strong>
          <p className="ml-4">Active</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Captain Name:</strong>
          <p className="ml-4">Bob</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Crew Size:</strong>
          <p className="ml-4">25</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Average Speed:</strong>
          <p className="ml-4">21 Knots</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Hours in Water:</strong>
          <p className="ml-4">175</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Remaining Journey:</strong>
          <p className="ml-4">0.53</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Estimated Remaining Time:</strong>
          <p className="ml-4">185 Hrs</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Wave Height:</strong>
          <p className="ml-4">2 m</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Wind Speed:</strong>
          <p className="ml-4">2 km/hr</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">Fuel Details:</strong>
          <p className="ml-4">62.5%</p>
        </div>

        <div className="text-lg text-gray-800">
          <strong className="font-medium text-indigo-600">CO₂ Emission:</strong>
          <p className="ml-4">3,852 tons of CO₂</p>
        </div>
      </div>
    </div>
  );
};

export default VoyageDetails;
