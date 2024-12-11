// GaugeMeter.jsx
import React from "react";

const GaugeMeter = ({ value, label }) => {
  return (
    <div className="flex flex-col items-center">
      <div
        className="w-16 h-16 rounded-full border-4 border-teal-500 flex items-center justify-center"
        style={{
          background: `conic-gradient(#4caf50 ${value}%, #f44336 ${value}% 100%)`,
        }}
      >
        <span className="text-white">{value}%</span>
      </div>
      <span className="text-teal-500 mt-2 text-sm">{label}</span>
    </div>
  );
};

export default GaugeMeter;
