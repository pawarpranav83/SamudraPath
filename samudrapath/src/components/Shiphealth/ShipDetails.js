import React from "react";

const ShipDetails = ({ details, riskLevel }) => {
  return (
    <div className="bg-white shadow-lg rounded-xl p-4 m-8 mt-4 border border-gray-300">
      <div className="flex space-x-8">
        {/* First Section */}
        <div className="flex-1 bg-white shadow-md rounded-lg border border-gray-200">
          {/* Heading */}
          <div className="bg-teal-600 text-white font-semibold text-xl rounded-t-lg p-4">
            Ship Details
          </div>
          {/* Content */}
          <div className="p-6 space-y-4">
            <p className="text-lg text-gray-800">
              <strong className="font-medium text-teal-600">Name:</strong> {details.name}
            </p>
            <p className="text-lg text-gray-800">
              <strong className="font-medium text-teal-600">IMO:</strong> {details.imo}
            </p>
            <p className="text-lg text-gray-800">
              <strong className="font-medium text-teal-600">MMSI:</strong> {details.mmsi}
            </p>
            <p className="text-lg text-gray-800">
              <strong className="font-medium text-teal-600">Speed (knots):</strong> {details.speed}
            </p>
            <p className="text-lg text-gray-800">
              <strong className="font-medium text-teal-600">Heading (degrees):</strong> {details.heading}
            </p>
            <p className="text-lg text-gray-800">
              <strong className="font-medium text-teal-600">Position Timestamp:</strong>{" "}
              {new Date(details.positionTime).toLocaleString()}
            </p>

            {/* Display risk level if available */}
            {riskLevel && (
              <p className="text-lg text-gray-800">
                <strong className="font-medium text-teal-600">Risk Level:</strong> {riskLevel}
              </p>
            )}
          </div>
        </div>

        {/* Second Section */}
        <div className="flex-1 bg-white shadow-md rounded-lg border border-gray-200">
          {/* Heading */}
          <div className="bg-teal-600 text-white font-semibold text-xl rounded-t-lg p-4">
            Location Details
          </div>
          {/* Content */}
          <div className="p-6 space-y-4 flex flex-col items-center justify-center">
  <p className="text-lg text-gray-800">
    <strong className="font-medium text-teal-600">Latitude:</strong> {details.latitude}
  </p>
  <p className="text-lg text-gray-800">
      <strong className="font-medium text-teal-600">Longitude:</strong> {details.longitude}
  </p>
  <button
    className="bg-teal-600 text-white font-semibold py-2 px-4 rounded hover:bg-teal-500"
    onClick={() => alert("Calculating optimal route...")}
  >
    Calculate Optimal Route
  </button>
</div>

        </div>
      </div>
    </div>
  );
};

export default ShipDetails;
