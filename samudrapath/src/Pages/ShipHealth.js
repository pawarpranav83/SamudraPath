import React, { useState } from "react";
import Header from "../components/Shiphealth/Header";
import ShipDetails from "../components/Shiphealth/ShipDetails";
import Navbar from "../components/Homepage/Navbar";

const ShipHealth = () => {
  const [shipData, setShipData] = useState(null);
  const [error, setError] = useState(null);
  const [loading, setLoading] = useState(false);

  const fetchShipDetails = async (imo) => {
    const options = {
      method: 'GET',
      headers: {
        Authorization: '88f1VhNUindHAbwmYSJZgmCp', // Your API key here
      },
    };
  
    setLoading(true);
    setError(null);
    setShipData(null);
  
    try {
      const response = await fetch(`https://cors-anywhere.herokuapp.com/https://api.terminal49.com/v2/vessels/${imo}`, options);
  
      if (!response || !response.ok) {
        // If response is undefined or status is not ok, throw error
        throw new Error(`Error: ${response ? response.statusText : 'No response from API'}`);
      }
  
      const data = await response.json();
      console.log('Ship Data:', data);
  
      if (!data || !data.data || !data.data.attributes) {
        throw new Error('No valid ship data returned');
      }
  
      const details = {
        name: data.data.attributes.name,
        imo: data.data.attributes.imo,
        mmsi: data.data.attributes.mmsi,
        latitude: data.data.attributes.latitude,
        longitude: data.data.attributes.longitude,
        speed: data.data.attributes.nautical_speed_knots,
        heading: data.data.attributes.navigational_heading_degrees,
        positionTime: data.data.attributes.position_timestamp,
      };
  
      // Risk level logic
      const riskLevel = details.speed > 15 ? 'High' : 'Low';
  
      setShipData({ details, riskLevel });
    } catch (err) {
      console.error('Error:', err);
      setError(err.message || 'An unexpected error occurred');
    } finally {
      setLoading(false);
    }
  };
  

  return (
    <div className="">
      <Navbar />
      <Header onSearch={fetchShipDetails} />

      {loading && <p className="text-indigo-600 mt-4">Loading ship details...</p>}

      {error && <p className="text-red-500 mt-4">Error: {error}</p>}

      {shipData && (
        <>
          <ShipDetails details={shipData.details} />
          {/* Optionally render additional components like a health bar */}
        </>
      )}

      {!loading && !shipData && !error && (
        <p className="text-gray-500 mt-4">Enter an IMO number to view ship details.</p>
      )}
    </div>
  );
};

export default ShipHealth;
