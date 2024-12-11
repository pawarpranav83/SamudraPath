import React, { useState, useEffect, useContext } from "react";
import Navbar from "../components/Homepage/Navbar";
import Sidebar from "../components/Homepage/Sidebar";
import MapView from "../components/Homepage/MapView";
import Papa from "papaparse";
import { ShipContext } from "../ShipContext";

const shipCategories = {
  "Cargo Ships": [
    "General Cargo Ship",
    "Refrigerated Cargo Ship",
    "Heavy Lift Cargo Ship",
  ],
  Tankers: ["Crude Oil Tanker", "Product Tanker", "Chemical Tanker"],
  "Container Ships": [
    "Feeder Ship",
    "Panamax Ship",
    "Ultra Large Container Ship (ULCS)",
  ],
  "Passenger Ships": ["Cruise Ship", "Ferries", "Yachts"],
  "Fishing Vessels": ["Trawler", "Longliner", "Seiner"],
  "Naval Ships": ["Aircraft Carrier", "Destroyer", "Frigate"],
  "Bulk Carriers": [
    "Handysize Bulk Carrier",
    "Panamax Bulk Carrier",
    "Capesize Bulk Carrier",
  ],
  "Research Vessels": [
    "Oceanographic Research Vessel",
    "Marine Research Vessel",
  ],
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
    setDestination,
  } = useContext(ShipContext);

  const [departureDate, setDepartureDate] = useState("");
  const [departureTime, setDepartureTime] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedSubtype, setSelectedSubtype] = useState("");
  const [sourceCoordinates, setSourceCoordinates] = useState(null);
  const [destinationCoordinates, setDestinationCoordinates] = useState(null);
  const [carriageWeight, setCarriageWeight] = useState("");

  const [routes, setRoutes] = useState([
    {
      id: 1,
      coordinates: [],
      color: "#00ff00",
      visible: true,
      name: "Safest Path",
      description: "Safest Path",
    },
    {
      id: 2,
      coordinates: [],
      color: "#0000FF",
      visible: true,
      name: "Fuel Efficient Path",
      description: "Fuel Efficient Path",
    },
    {
      id: 3,
      coordinates: [],
      color: "#FFA500",
      visible: true,
      name: "Shortest Path",
      description: "Shortest Path",
    },
    // {
    //   id: 4,
    //   coordinates: [],
    //   color: "#00FFFF",
    //   visible: true,
    //   name: "Optimal Route",
    //   description: "Equal Weight Optimal Route ",
    // },
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
    // Helper function to fetch and parse CSV files
    const fetchCSV = (url, parseFunction, isPirate = false) => {
      fetch(`${url}?t=${Date.now()}`) // Cache busting with timestamp
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Network response was not ok for ${url}`);
          }
          return response.text();
        })
        .then((data) => {
          Papa.parse(data, {
            header: true,
            skipEmptyLines: true,
            complete: (results) => {
              const coordinates = results.data.map((row) => [
                parseFloat(row.Longitude || row.longitude),
                parseFloat(row.Latitude || row.latitude),
              ]);

              if (isPirate) {
                parseFunction(coordinates);
              } else {
                // For routes, parseFunction expects (id, coordinates)
                const routeId = parseInt(url.match(/path_(\w+)_smoothed\.csv$/)[1]);
                // Extracting the number based on the filename
                // Assuming filenames are like 'path_safe_smoothed.csv', etc.
                // You might need to adjust this if the naming convention changes
                const idMap = {
                  safe: 1,
                  fuel: 2,
                  short: 3,
                  weighted: 4,
                };
                const key = url.match(/path_(\w+)_smoothed\.csv$/)[1];
                const routeIdFinal = idMap[key] || 1; // Default to 1 if not found
                parseFunction(routeIdFinal, coordinates);
              }
            },
          });
        })
        .catch((error) => {
          console.error(`Error fetching ${url}:`, error);
        });
    };

    // Function to fetch all CSV data
    const fetchAllData = () => {
      // Fetch pirate coordinates
      fetchCSV("/filtered_coordinates.csv", setPirateCoordinates, true);

      // Fetch routes
      fetchCSV("/path_safe_smoothed.csv", updateCoordinates);
      fetchCSV("/path_fuel_smoothed.csv", updateCoordinates);
      fetchCSV("/path_short_smoothed.csv", updateCoordinates);
    };

    // Initial fetch
    fetchAllData();

    // Set up polling interval (e.g., every 30 seconds)
    const intervalId = setInterval(fetchAllData, 3000); 

    // Cleanup interval on component unmount
    return () => clearInterval(intervalId);
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
