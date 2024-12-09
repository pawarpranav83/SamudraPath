import React, { useState, useEffect, useContext } from "react";
import Navbar from "../components/Homepage/Navbar";
import Sidebar from "../components/Homepage/Sidebar";
import MapView from "../components/Homepage/MapView";
import Papa from "papaparse";
import { ShipContext } from "../ShipContext";
const shipCategories = {
  "Cargo Ships": ["General Cargo Ship", "Refrigerated Cargo Ship", "Heavy Lift Cargo Ship"],
  Tankers: ["Crude Oil Tanker", "Product Tanker", "Chemical Tanker"],
  "Container Ships": ["Feeder Ship", "Panamax Ship", "Ultra Large Container Ship (ULCS)"],
  "Passenger Ships": ["Cruise Ship", "Ferries", "Yachts"],
  "Fishing Vessels": ["Trawler", "Longliner", "Seiner"],
  "Naval Ships": ["Aircraft Carrier", "Destroyer", "Frigate"],
  "Bulk Carriers": ["Handysize Bulk Carrier", "Panamax Bulk Carrier", "Capesize Bulk Carrier"],
  "Research Vessels": ["Oceanographic Research Vessel", "Marine Research Vessel"],
  "Offshore Vessels": [
    "Platform Supply Vessel (PSV)",
    "Anchor Handling Tug Supply Vessel (AHTS)",
    "Offshore Support Vessel (OSV)",
  ],
  Tugboats: ["Harbor Tug", "Ocean-going Tug"],
};



const HomePage = () => {
  const {
    source,
    setSource,
    destination,
    setDestination
  } = useContext(ShipContext)

  const [departureDate, setDepartureDate] = useState("");
  const [departureTime, setDepartureTime] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedSubtype, setSelectedSubtype] = useState("");
  const [sourceCoordinates, setSourceCoordinates] = useState(null);
  const [destinationCoordinates, setDestinationCoordinates] = useState(null);
  const [carriageWeight, setCarriageWeight] = useState("");
  
  const [routes, setRoutes] = useState([
    { id: 1, coordinates: [], color: "#00ff00", visible: true, name: "Safest Path", description: "Safest Path" },
    { id: 2, coordinates: [], color: "#0000FF", visible: true, name: "Fuel Efficient Path", description: "Fuel Efficient Path" },
    { id: 3, coordinates: [], color: "#FFA500", visible: true, name: "Shortest Path", description: "Shortest Path" },
    { id: 4, coordinates: [], color: "#00FFFF", visible: true, name: "Optimal Route", description: "Equal Weight Optimal Route " },
  ]);
  const [pirateCoordinates, setPirateCoordinates] = useState([]);
  
  const handleCategoryChange = (event) => {
    setSelectedCategory(event.target.value);
    setSelectedSubtype("");
  };

  const handleSubtypeChange = (event) => {
    setSelectedSubtype(event.target.value);
  };

  const handleMapClick = (event) => {
    const { lng, lat } = event.lngLat;

    if (!sourceCoordinates && !source) {
      setSourceCoordinates({ lat, lng });
      setSource(`${lat}, ${lng}`);
    } else if (!destinationCoordinates && !destination) {
      setDestinationCoordinates({ lat, lng });
      setDestination(`${lat}, ${lng}`);
    }
  };

    // Function to update coordinates of a route by id
  const updateCoordinates = (id, newCoordinates) => {
    setRoutes((prevRoutes) =>
      prevRoutes.map((route) =>
        route.id === id ? { ...route, coordinates: newCoordinates } : route
      )
    );
  };

  // Function to update visibility of a route by id
  const updateVisibility = (id, visibility) => {
    setRoutes((prevRoutes) =>
      prevRoutes.map((route) =>
        route.id === id ? { ...route, visible: visibility } : route
      )
    );
  };


  useEffect(() => {
    fetch("/filtered_coordinates.csv") // Adjust path as needed
      .then((response) => response.text())
      .then((data) => {
        Papa.parse(data, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const coordinates = results.data.map((row) => [
              parseFloat(row.longitude),
              parseFloat(row.latitude),
            ]);

            setPirateCoordinates(coordinates);
          },
        });
      });

    // Fetch and parse CSV file
    fetch("/path_safe_smoothed.csv") // Adjust path as needed
      .then((response) => response.text())
      .then((data) => {
        Papa.parse(data, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const coordinates = results.data.map((row) => [
              parseFloat(row.Longitude),
              parseFloat(row.Latitude),
            ]);
            updateCoordinates(1, coordinates)
          },
        });
      });
    fetch("/path_fuel_smoothed.csv") // Adjust path as needed
      .then((response) => response.text())
      .then((data) => {
        Papa.parse(data, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const coordinates = results.data.map((row) => [
              parseFloat(row.Longitude),
              parseFloat(row.Latitude),
            ]);
            updateCoordinates(2, coordinates)
          },
        });
      });
    fetch("/path_short_smoothed.csv") // Adjust path as needed
      .then((response) => response.text())
      .then((data) => {
        Papa.parse(data, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const coordinates = results.data.map((row) => [
              parseFloat(row.Longitude),
              parseFloat(row.Latitude),
            ]);
            updateCoordinates(3, coordinates)
          },
        });
      });
    fetch("/path_weighted_smoothed.csv") // Adjust path as needed
      .then((response) => response.text())
      .then((data) => {
        Papa.parse(data, {
          header: true,
          skipEmptyLines: true,
          complete: (results) => {
            const coordinates = results.data.map((row) => [
              parseFloat(row.Longitude),
              parseFloat(row.Latitude),
            ]);
            updateCoordinates(4, coordinates)
          },
        });
      });
  }, []);

  return (
    <div className="flex flex-col h-screen">
      <Navbar />
      <div className="flex flex-row flex-grow overflow-hidden">
        <Sidebar          
          selectedCategory={selectedCategory}
          setSelectedCategory={setSelectedCategory}
          selectedSubtype={selectedSubtype}
          setSelectedSubtype={setSelectedSubtype}
          departureDate={departureDate}
          setDepartureDate={setDepartureDate}
          departureTime={departureTime}
          setDepartureTime={setDepartureTime}
          shipCategories={shipCategories}
          carriageWeight={carriageWeight}
          setCarriageWeight={setCarriageWeight}
          handleCategoryChange={handleCategoryChange}
          handleSubtypeChange={handleSubtypeChange}
          setSourceCoordinates={setSourceCoordinates}
          setDestinationCoordinates={setDestinationCoordinates} 
          routes={routes}
          updateVisibility={updateVisibility}  
        />
        <MapView
          handleMapClick={handleMapClick}
          routes={routes}
          pirateCoordinates={pirateCoordinates}
        />
        </div>
      </div>
  );
};

export default HomePage;
