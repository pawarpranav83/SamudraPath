import React from 'react';

const LastMaintenanceHistory = () => {
  // Example data, you can replace these with actual data dynamically
  const lastMaintenance = {
    date: '2024-11-20',
    type: 'Routine',
    partsReplaced: ['Engine Filters', 'Cooling System'],
    technician: {
      name: 'John Doe',
      company: 'MarineTech Maintenance',
      contact: 'john.doe@marine-tech.com'
    }, 
    notes: 'Routine maintenance completed on schedule. All systems are operational.',
    duration: '4 hours',
  };

  return (
    <div className="bg-white shadow-lg rounded-xl p-8 mb-8 border border-gray-300">
      <h3 className="text-3xl font-semibold text-indigo-600 mb-6 border-b-2 border-indigo-500 pb-2">Last Maintenance History</h3>

      {/* Maintenance Details */}
      <div className="space-y-4 text-lg text-gray-800">
        <div>
          <strong className="font-medium text-indigo-600">Last Maintenance Date:</strong>
          <p>{lastMaintenance.date}</p>
        </div>

        <div>
          <strong className="font-medium text-indigo-600">Maintenance Type:</strong>
          <p>{lastMaintenance.type}</p>
        </div>

        <div>
          <strong className="font-medium text-indigo-600">Parts Replaced:</strong>
          <ul className="list-disc pl-5">
            {lastMaintenance.partsReplaced.map((part, index) => (
              <li key={index}>{part}</li>
            ))}
          </ul>
        </div>

        <div>
          <strong className="font-medium text-indigo-600">Technician:</strong>
          <p>{lastMaintenance.technician.name}</p>
          <p><strong>Company:</strong> {lastMaintenance.technician.company}</p>
          <p><strong>Contact:</strong> {lastMaintenance.technician.contact}</p>
        </div>

        <div>
          <strong className="font-medium text-indigo-600">Maintenance Notes:</strong>
          <p>{lastMaintenance.notes}</p>
        </div>

        <div>
          <strong className="font-medium text-indigo-600">Duration:</strong>
          <p>{lastMaintenance.duration}</p>
        </div>
      </div>
    </div>
  );
};

export default LastMaintenanceHistory;
