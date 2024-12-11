import React, { useState, useEffect, useContext, useRef } from "react";
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
    // Dynamic routes will be added with id >=4
  ]);

  const dynamicRouteIdRef = useRef(4); // Starting ID for dynamic routes
  const fetchedDynamicRoutesRef = useRef(new Set()); // To track fetched dynamic routes

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

  // Function to update coordinates of a predefined route by id
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

  const nsga_paths_length = 3; // Adjust based on the number of dynamic paths you have

  useEffect(() => {
  // Define a color palette for dynamic routes
  const dynamicColors = [
    "#FF0000", // Red
    "#00FFFF", // Cyan
    "#FF00FF", // Magenta
    "#800000", // Maroon
    "#808000", // Olive
    "#008080", // Teal
    "#800080", // Purple
    "#008000", // Green
    "#000080", // Navy
    "#FFA500", // Orange
    // Add more colors if needed
  ];

  // Helper function to get color for dynamic routes
  const getDynamicColor = (index) => {
    return dynamicColors[index % dynamicColors.length];
  };

  // Helper function to fetch and parse CSV files
  const fetchCSV = (url, parseFunction, isPirate = false, dynamicIndex = null) => {
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
              // Determine if the path is predefined or dynamic
              const predefinedRouteKeys = {
                safe: 1,
                fuel: 2,
                short: 3,
              };

              const match = url.match(/path_(\w+)_smoothed\.csv$/);
              if (match) {
                const key = match[1];
                const routeId = predefinedRouteKeys[key];

                if (routeId) {
                  // Predefined route
                  parseFunction(routeId, coordinates);
                } else if (dynamicIndex !== null) {
                  // Dynamic route
                  // Check if this dynamic route has already been fetched
                  if (!fetchedDynamicRoutesRef.current.has(url)) {
                    let newRoute = {}
                    if (dynamicIndex == 1) {
                      newRoute = {
                        id: dynamicRouteIdRef.current,
                        coordinates: coordinates,
                        color: getDynamicColor(dynamicRouteIdRef.current - 4),
                        visible: true, // Set to true or false based on your preference
                        name: `Optimal Path ${dynamicRouteIdRef.current - 3}`,
                        description: `Optimal Path ${dynamicRouteIdRef.current - 3}`,
                      };
                    }
                    else {
                      newRoute = {
                        id: dynamicRouteIdRef.current,
                        coordinates: coordinates,
                        color: getDynamicColor(dynamicRouteIdRef.current - 4),
                        visible: false, // Set to true or false based on your preference
                        name: `Optimal Path ${dynamicRouteIdRef.current - 3}`,
                        description: `Optimal Path ${dynamicRouteIdRef.current - 3}`,
                      };
                    }
                    setRoutes((prevRoutes) => [...prevRoutes, newRoute]);
                    fetchedDynamicRoutesRef.current.add(url);
                    console.log(`Added new dynamic route: ${newRoute.name}`);
                    dynamicRouteIdRef.current += 1;
                  } else {
                    // Dynamic route already fetched
                    console.log(`Dynamic route ${url} already fetched.`);
                  }
                }
              }
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

    // Fetch predefined routes
    fetchCSV("/path_safe_smoothed.csv", updateCoordinates);
    fetchCSV("/path_fuel_smoothed.csv", updateCoordinates);
    fetchCSV("/path_short_smoothed.csv", updateCoordinates);

    // Fetch additional dynamic routes using a for loop
    for (let i = 1; i <= nsga_paths_length; i++) {
      const routePath = `/path_${i}_smoothed.csv`;
      fetchCSV(routePath, updateCoordinates, false, i);
    }
  };

  // Initial fetch
  fetchAllData();

  // Set up polling interval (e.g., every 3 seconds)
  const intervalId = setInterval(fetchAllData, 3000); 

  // Cleanup interval on component unmount
  return () => clearInterval(intervalId);
}, [nsga_paths_length]);

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
