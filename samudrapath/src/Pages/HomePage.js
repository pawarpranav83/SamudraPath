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
  const { source, setSource, destination, setDestination } = useContext(ShipContext);

  const [departureDate, setDepartureDate] = useState("");
  const [departureTime, setDepartureTime] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("");
  const [selectedSubtype, setSelectedSubtype] = useState("");
  const [sourceCoordinates, setSourceCoordinates] = useState(null);
  const [destinationCoordinates, setDestinationCoordinates] = useState(null);

  const [routes, setRoutes] = useState([
    {
      id: 1,
      coordinates: [],
      color: "#00ff00",
      visible: false,
      name: "Safest Path",
      description: "Safest Path",
    },
    {
      id: 2,
      coordinates: [],
      color: "#0000FF",
      visible: false,
      name: "Fuel Efficient Path",
      description: "Fuel Efficient Path",
    },
    {
      id: 3,
      coordinates: [],
      color: "#FFA500",
      visible: false,
      name: "Shortest Path",
      description: "Shortest Path",
    },
    // Dynamic routes will start from id >= 4
  ]);

  const dynamicRouteIdRef = useRef(4); // Starting ID for dynamic routes
  const fetchedDynamicRoutesRef = useRef(new Set()); // To track fetched dynamic routes

  const [pirateCoordinates, setPirateCoordinates] = useState([]);
  const [routeData, setRouteData] = useState({}); 
  
  // New state to hold the number of paths from results.csv
  const nsgaPathsLength = 3;

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

  // Update predefined or dynamic route coordinates
  const updateCoordinates = (id, newCoordinates) => {
    setRoutes((prevRoutes) =>
      prevRoutes.map((route) =>
        route.id === id ? { ...route, coordinates: newCoordinates } : route
      )
    );
  };

  const updateVisibility = (id, visibility) => {
    setRoutes((prevRoutes) =>
      prevRoutes.map((route) =>
        route.id === id ? { ...route, visible: visibility } : route
      )
    );
  };

  


  useEffect(() => {
    // Only proceed if nsgaPathsLength is known
    if (nsgaPathsLength === null) return;

    // Define dynamicColors and fetchCSV inside this effect so they can use nsgaPathsLength
    const dynamicColors = [
      "#FF0000", "#00FFFF", "#FF00FF", "#800000",
      "#808000", "#008080", "#800080", "#008000",
      "#000080", "#FFA500",
    ];

    const getDynamicColor = (index) => {
      return dynamicColors[index % dynamicColors.length];
    };

    const fetchCSV = (url, parseFunction, isPirate = false, dynamicIndex = null) => {
      fetch(`${url}?t=${Date.now()}`)
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
                setPirateCoordinates(coordinates);
              } else {
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
                    parseFunction(routeId, coordinates);
                  }
                } else if (dynamicIndex !== null) {
                  // Dynamic route
                  if (!fetchedDynamicRoutesRef.current.has(url)) {
                    const newRoute = {
                      id: dynamicRouteIdRef.current,
                      coordinates: coordinates,
                      color: getDynamicColor(dynamicRouteIdRef.current - 4),
                      visible: true,
                      name: `Optimal Path ${dynamicRouteIdRef.current - 3}`,
                      description: `Optimal Path ${dynamicRouteIdRef.current - 3}`,
                    };
                    setRoutes((prevRoutes) => [...prevRoutes, newRoute]);
                    fetchedDynamicRoutesRef.current.add(url);
                    console.log(`Added new dynamic route: ${newRoute.name}`);
                    dynamicRouteIdRef.current += 1;
                  } else {
                    console.log(`Dynamic route ${url} already fetched.`);
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

    // Once we know nsgaPathsLength, fetch all data
    const fetchAllData = () => {
      // Fetch pirate coordinates
      fetchCSV("/filtered_coordinates.csv", null, true);

      // Fetch predefined routes
      fetchCSV("/path_safe_smoothed.csv", updateCoordinates);
      fetchCSV("/path_fuel_smoothed.csv", updateCoordinates);
      fetchCSV("/path_short_smoothed.csv", updateCoordinates);

      // Fetch dynamic routes based on nsgaPathsLength
      for (let i = 1; i <= nsgaPathsLength; i++) {
        const routePath = `/path${i}.csv`;
        fetchCSV(routePath, updateCoordinates, false, i);
      }
    };

    fetchAllData();
  }, [nsgaPathsLength]);

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
