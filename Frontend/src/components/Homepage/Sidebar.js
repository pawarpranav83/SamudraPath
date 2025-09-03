import React, { useState, useContext } from "react";
import Papa from 'papaparse';
import {
  FaMapMarkerAlt,
  FaMapPin,
  FaCalendarAlt,
  FaClock,
  FaRoute,
  FaShip,
  FaWeightHanging,
  FaList,
  FaWeight,
  FaChartArea,
  FaEye,
  FaEyeSlash
} from "react-icons/fa";
import { ShipContext } from "../../ShipContext";
import Modal from "./RouteDetails";
import axios from "axios";

const Sidebar = ({
  selectedCategory,
  setSelectedCategory,
  selectedSubtype,
  setSelectedSubtype,
  departureDate,
  setDepartureDate,
  departureTime,
  setDepartureTime,
  shipCategories,
  handleCategoryChange,
  handleSubtypeChange,
  setSourceCoordinates,
  setDestinationCoordinates,
  routes,
  updateVisibility
}) => {

  const [isLoading, setIsLoading] = useState(false);
  const [activeTab, setActiveTab] = useState("details");
  const [showCustomizedRoute, setShowCustomizedRoute] = useState(false);
  const [errorMessage, setErrorMessage] = useState("");
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [showRecalculateButton, setShowRecalculateButton] = useState(false);
  const [routeType, setRouteType] = useState("");

  const {
    source,
    setSource,
    destination,
    setDestination,
    shipDisplacement,
    setShipDisplacement,
    frontalArea,
    setFrontalArea,
    hullEfficiency,
    setHullEfficiency,
    propellerEfficiency,
    setPropellerEfficiency,
    engineShaftEfficiency,
    setEngineShaftEfficiency,
    heightAboveSea,
    setHeightAboveSea,
    resonantPeriod,
    setResonantPeriod,
    csfoc,
    setCsfoc,
    safetyWeight,
    setSafetyWeight,
    distanceWeight,
    setDistanceWeight,
    fuelWeight,
    setFuelWeight,
    setRouteData,
    carriageWeight,
    setCarriageWeight
  } = useContext(ShipContext);

  const { routeData } = useContext(ShipContext);

  const mapImages = [
    "https://via.placeholder.com/150?text=Map1",
    "https://via.placeholder.com/150?text=Map2",
    "https://via.placeholder.com/150?text=Map3",
    "https://via.placeholder.com/150?text=Map4",
  ];

  let routeDetails = {};
  if (!routes || routes.length === 0) {
    routeDetails = {
      fuelCost: "$500",
      duration: "10 hours",
      safetyIndex: "8.5",
    };
  } else {
    routeDetails = {
      fuelCost: "$500",
      duration: "10 hours",
      safetyIndex: "8.5",
      coordinates: routes[0].coordinates,
    };
  }

  const handleWeightSubmit = () => {
    const totalWeight =
      parseFloat(safetyWeight) +
      parseFloat(fuelWeight) +
      parseFloat(distanceWeight);
    if (totalWeight === 1) {
      setShowCustomizedRoute(true);
      setErrorMessage(""); // Clear any previous error messages
    } else {
      setErrorMessage("The weights must add up to 1. Please adjust them.");
      setShowCustomizedRoute(false);
    }
  };

  const handleFindRoutes = () => {
    setIsLoading(true); // Show loader
    setTimeout(() => {
      setIsLoading(false); // Hide loader after 5 seconds
      setActiveTab("routes"); // Navigate to routes tab
    }, 5000);
  };

  const handleClick = (routeId) => {
    setShowRecalculateButton(true);
    // Logic to show the position on map goes here
  };

  const handleRecalculateRoute = (routeId) => {
    alert(`Recalculating route from position after 6 hours for Route ID: ${routeId}`);
    // Logic to recalculate the route goes here
  };

  const handleSubmit = async (e) => {
    e.preventDefault();

    // Validate source and destination
    if (!source.includes(",")) {
      console.error("Invalid source format. It should be in 'lat, lon' format.");
      return;
    }

    if (!destination.includes(",")) {
      console.error("Invalid destination format. It should be in 'lat, lon' format.");
      return;
    }

    const [start_lat, start_lon] = source.split(",").map((val) => val.trim());
    const [goal_lat, goal_lon] = destination.split(",").map((val) => val.trim());

    const payload = {
      start_lat: parseFloat(start_lat),
      start_lon: parseFloat(start_lon),
      goal_lat: parseFloat(goal_lat),
      goal_lon: parseFloat(goal_lon),
      ship_speed: 40,
      ship_dis: parseFloat(shipDisplacement),
      ship_height: parseFloat(heightAboveSea),
      area_front: parseFloat(frontalArea),
      ship_reso: parseFloat(resonantPeriod),
      hull_eff: parseFloat(hullEfficiency),
      prop_eff: parseFloat(propellerEfficiency),
      engine_eff: parseFloat(engineShaftEfficiency),
      c_sfoc: parseFloat(csfoc),
      shipw: parseFloat(carriageWeight)
    };

    try {
      // Set loading state
      setIsLoading(true);

      console.log(payload)

      const response = await axios.post("http://localhost:5000/calculate_route", payload, {
        headers: {
          "Content-Type": "application/json",
        },
      });

      setRouteData(response.data)
    } catch (error) {
      // Handle any errors
      console.error("Error calculating route:", error.response ? error.response.data : error.message);

      // Optional: Show error message to user
      if (error.response) {
        alert(`Error: ${error.response.data.error || 'Failed to calculate route'}`);
      } else if (error.request) {
        alert('No response received from server. Please check your connection.');
      } else {
        alert('Error in request setup');
      }
    } finally {
      // Reset loading state
      setIsLoading(false);
    }
  };

  return (
    <div
      className="flex h-screen"
      style={{ minWidth: "450px", maxWidth: "450px" }}
    >
      {/* Navigation */}
      <nav className="w-1/6 bg-gray-600 text-white flex flex-col">
        <button
          className={`p-4 text-left hover:bg-teal-600 ${activeTab === "details" ? "bg-teal-700 drop-shadow-md" : ""}`}
          onClick={() => setActiveTab("details")}
        >
          <FaList /> Ship Details
        </button>
        <button
          className={`p-4 text-left hover:bg-teal-600 ${activeTab === "routes" ? "bg-teal-700" : ""}`}
          onClick={() => setActiveTab("routes")}
        >
          <FaRoute /> Ship Routes
        </button>
      </nav>

      {/* Sidebar Content */}
      <aside
        className="w-5/6 bg-gradient-to-br from-gray-100 to-gray-200 p-5 shadow-lg flex flex-col gap-4 transition-all overflow-y-auto"
        style={{
          maxHeight: "100vh", // To prevent it from growing beyond the screen height
          scrollbarWidth: "none", // For Firefox
        }}
      >
        <style>
          {`
            /* Hide scrollbar for Webkit browsers */
            ::-webkit-scrollbar {
              display: none;
            }
          `}
        </style>

        {/* Loading State */}
        {isLoading && (
          <div className="flex flex-col items-center justify-center h-full">
            <div className="loader mb-4 border-t-4 border-teal-500 rounded-full w-12 h-12 animate-spin"></div>
            <p className="text-gray-700 font-medium">Calculating optimized routes...</p>
          </div>
        )}

        {/* Details Tab */}
        {!isLoading && activeTab === "details" && (
          <div
            className="space-y-4"
            style={{
              maxHeight: "calc(100vh - 100px)", // Adjusts to viewport height
              overflowY: "auto", // Ensures vertical scroll
            }}
          >
            {/* Source and Destination Form */}
            <form onSubmit={handleSubmit} className="space-y-4">
              {/* Source Input */}
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                  <FaMapMarkerAlt /> Source
                </label>
                <input
                  type="text"
                  placeholder="Enter source or click on map"
                  value={source}
                  onChange={(e) => {
                    setSource(e.target.value);
                    setSourceCoordinates(null);
                  }}
                  className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                />
              </div>

              {/* Destination Input */}
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                  <FaMapPin /> Destination
                </label>
                <input
                  type="text"
                  placeholder="Enter destination or click on map"
                  value={destination}
                  onChange={(e) => {
                    setDestination(e.target.value);
                    setDestinationCoordinates(null);
                  }}
                  className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                />
              </div>

              {/* Ship Category Dropdown */}
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                  <FaShip />
                  Select Ship Category
                </label>
                <select
                  className="w-full bg-white p-2 rounded-md shadow-md focus:outline-none h-9"
                  value={selectedCategory}
                  onChange={handleCategoryChange}
                >
                  <option value="">-- Select Category --</option>
                  {Object.keys(shipCategories).map((category) => (
                    <option key={category} value={category}>
                      {category}
                    </option>
                  ))}
                </select>
              </div>

              {/* Ship Subcategory Dropdown */}
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                  Select Ship Subcategory
                </label>
                <select
                  className="w-full bg-white p-2 rounded-md shadow-md focus:outline-none h-9"
                  value={selectedSubtype}
                  onChange={handleSubtypeChange}
                  disabled={!selectedCategory}
                >
                  <option value="">-- Select Subcategory --</option>
                  {selectedCategory &&
                    shipCategories[selectedCategory].map((subtype) => (
                      <option key={subtype} value={subtype}>
                        {subtype}
                      </option>
                    ))}
                </select>
              </div>

              {/* Carriage Weight */}
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                  <FaWeightHanging /> Carriage Weight (tonnes)
                </label>
                <input
                  type="number"
                  placeholder="Enter Carriage Weight in tonnes"
                  value={carriageWeight}
                  onChange={(e) => setCarriageWeight(e.target.value)}
                  className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                />
              </div>

              {/* Ship Displacement */}
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                  <FaWeight /> Ship Displacement (tonnes)
                </label>
                <input
                  type="number"
                  placeholder="Enter ship displacement in tonnes"
                  value={shipDisplacement}
                  onChange={(e) => setShipDisplacement(e.target.value)}
                  className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                />
              </div>

              {/* Frontal Area */}
              <div className="space-y-1">
                <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                  <FaChartArea /> Frontal Area (m²)
                </label>
                <input
                  type="number"
                  placeholder="Enter Frontal Area in m²"
                  value={frontalArea}
                  onChange={(e) => setFrontalArea(e.target.value)}
                  className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                />
              </div>

              {/* Fuel Consumption */}
              <div className="flex-1">
                <label className="text-sm font-semibold text-gray-700">
                  Specific Fuel Oil Consumption (g/kWh)
                </label>
                <input
                  type="number"
                  value={csfoc}
                  placeholder="Enter fuel consumption per hour"
                  onChange={(e) => setCsfoc(e.target.value)}
                  className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                />
              </div>

              {/* Resonant Period and Height Above Sea */}
              <div className="flex space-x-4">
                {/* Resonant Period */}
                <div className="flex-1 space-y-1">
                  <label className="text-sm font-semibold text-gray-700">
                    Resonant Period (sec)
                  </label>
                  <input
                    type="number"
                    value={resonantPeriod}
                    placeholder="Resonant period in seconds"
                    onChange={(e) => setResonantPeriod(e.target.value)}
                    className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                    min="0"
                  />
                </div>

                {/* Height Above Sea */}
                <div className="flex-1 space-y-1">
                  <label className="text-sm font-semibold text-gray-700">
                    Height above sea (m)
                  </label>
                  <input
                    type="number"
                    value={heightAboveSea}
                    placeholder="Height of ship above sea"
                    onChange={(e) => setHeightAboveSea(e.target.value)}
                    className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                    min="0"
                  />
                </div>
              </div>

              {/* Efficiency Inputs */}
              <p className="text-sm font-semibold text-gray-700 flex items-center gap-2 mb-0">
                <FaChartArea /> Enter Efficiencies for each
              </p>

              <div className="flex space-x-3 mt-0">
                {/* Hull Efficiency */}
                <div className="flex-1 space-y-1">
                  <label className="text-sm font-semibold text-gray-700">
                    Hull (ηₕ)
                  </label>
                  <input
                    type="number"
                    value={hullEfficiency}
                    placeholder="Efficiency"
                    onChange={(e) => setHullEfficiency(e.target.value)}
                    className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                    step="0.01"
                    min="0"
                  />
                </div>

                {/* Propeller Efficiency */}
                <div className="flex-1 space-y-1">
                  <label className="text-sm font-semibold text-gray-700">
                    Propeller (ηₚ)
                  </label>
                  <input
                    type="number"
                    value={propellerEfficiency}
                    placeholder="Efficiency"
                    onChange={(e) => setPropellerEfficiency(e.target.value)}
                    className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                    step="0.01"
                    min="0"
                  />
                </div>

                {/* Engine Shaft Efficiency */}
                <div className="flex-1 space-y-1">
                  <label className="text-sm font-semibold text-gray-700">
                    Engine Shaft (ηₑ)
                  </label>
                  <input
                    type="number"
                    value={engineShaftEfficiency}
                    placeholder="Efficiency"
                    onChange={(e) => setEngineShaftEfficiency(e.target.value)}
                    className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                    step="0.01"
                    min="0"
                  />
                </div>
              </div>

              {/* Departure Date and Time */}
              <div className="flex space-x-4">
                {/* Departure Date */}
                <div className="flex-1 space-y-1">
                  <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                    <FaCalendarAlt /> Departure Date
                  </label>
                  <input
                    type="date"
                    value={departureDate}
                    onChange={(e) => setDepartureDate(e.target.value)}
                    className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                  />
                </div>

                {/* Departure Time */}
                <div className="flex-1 space-y-1">
                  <label className="flex items-center gap-2 text-sm font-semibold text-gray-700">
                    <FaClock /> Departure Time
                  </label>
                  <input
                    type="time"
                    value={departureTime}
                    onChange={(e) => setDepartureTime(e.target.value)}
                    className="w-full p-3 border rounded-md focus:outline-none focus:ring-2 focus:ring-teal-500 h-8"
                  />
                </div>
              </div>

              {/* Find Routes Button */}
              <button
                type="submit"
                className="w-full flex items-center justify-center gap-2 p-3 bg-teal-600 text-white rounded-md hover:bg-teal-700 transform transition duration-300 h-10"
              >
                <FaRoute /> Find Optimized Routes
              </button>
            </form>
          </div>
        )}

        {/* Routes Tab */}
        {!isLoading && activeTab === "routes" && (
          <div
            className="flex flex-col gap-4"
            style={{
              maxHeight: "calc(100vh - 100px)", // Adjusts to viewport height
              overflowY: "auto", // Ensures vertical scroll
            }}
          >
            <h2 className="text-xl font-semibold">
              Respective Optimized Routes
            </h2>

            {/* Hardcoded Route 1 */}
            <div className="p-3 bg-white rounded-md shadow-md hover:shadow-lg transition mb-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold">Optimal Route 1</h3>
                  {/* <p className="text-gray-700">{route.description}</p> */}
                </div>

                <button
                  className="mr-4 focus:outline-none"
                  onClick={() => {
                    const route = routes.find(route => route.id === 1);
                    updateVisibility(1, !route.visible);
                  }}
                >
                  {routes.find(route => route.id === 1).visible ? (
                    <FaEye className="text-2xl" style={{ color: "#FF0000" }} />
                  ) : (
                    <FaEyeSlash className="text-2xl text-gray-400" />
                  )}
                </button>
              </div>

              <div className="flex items-start gap-4 mt-3">
                {/* Route Data Section */}
                <div
                  className="flex flex-col p-4 bg-white text-sm text-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 ease-in-out flex-grow"
                  style={{
                    borderLeft: `4px solid #FF0000`, // Modern border styling
                  }}
                >
                  <div className="mb-2">
                    <span className="font-semibold">Fuel:</span> {112978.787 || 'N/A'}
                  </div>
                  <div className="mb-2">
                    <span className="font-semibold">Duration(hrs):</span> {62.035 || 'N/A'}
                  </div>
                  <div>
                    <span className="font-semibold">Safety Index:</span> {0.576 || 'N/A'}
                  </div>
                </div>

                {/* Position Button */}
                <button
                  className="w-32 p-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 flex items-center justify-center transition-colors duration-300 ease-in-out shadow-md hover:shadow-lg"
                  style={{
                    borderWidth: "2px",
                    borderColor: "#FF0000", // Dynamically set the border color
                  }}
                  onClick={() => handleClick(1)} // Ensure route.id is passed
                >
                  Position After 24 Hours
                </button>
              </div>

              {/* Conditionally render the "Recalculate Route" button */}
              {showRecalculateButton && (
                <button
                  className="w-full flex items-center justify-center gap-2 p-3 bg-red-600 text-white rounded-md hover:bg-red-700 transform transition duration-300 h-10 mt-2"
                  onClick={() => handleRecalculateRoute(1)}
                >
                  Recalculate Route
                </button>
              )}
            </div>

            {/* Hardcoded Route 2 */}
            <div className="p-3 bg-white rounded-md shadow-md hover:shadow-lg transition mb-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold">Optimal Route 2</h3>
                  {/* <p className="text-gray-700">{route.description}</p> */}
                </div>

                <button
                  className="mr-4 focus:outline-none"
                  onClick={() => {
                    const route = routes.find(route => route.id === 2);
                    updateVisibility(2, !route.visible);
                  }}
                >
                  {routes.find(route => route.id === 2).visible ? (
                    <FaEye className="text-2xl" style={{ color: "#0000FF" }} />
                  ) : (
                    <FaEyeSlash className="text-2xl text-gray-400" />
                  )}
                </button>
              </div>

              <div className="flex items-start gap-4 mt-3">
                {/* Route Data Section */}
                <div
                  className="flex flex-col p-4 bg-white text-sm text-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 ease-in-out flex-grow"
                  style={{
                    borderLeft: `4px solid #FF00FF`, // Modern border styling
                  }}
                >
                  <div className="mb-2">
                    <span className="font-semibold">Fuel:</span> {132364 || 'N/A'}
                  </div>
                  <div className="mb-2">
                    <span className="font-semibold">Duration(hrs):</span> {65.432 || 'N/A'}
                  </div>
                  <div>
                    <span className="font-semibold">Safety Index:</span> {0.225|| 'N/A'}
                  </div>
                </div>

                {/* Position Button */}
                <button
                  className="w-32 p-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 flex items-center justify-center transition-colors duration-300 ease-in-out shadow-md hover:shadow-lg"
                  style={{
                    borderWidth: "2px",
                    borderColor: "#FF00FF", // Dynamically set the border color
                  }}
                  onClick={() => handleClick(2)} // Ensure route.id is passed
                >
                  Position After 24 Hours
                </button>
              </div>

              {/* Conditionally render the "Recalculate Route" button */}
              {showRecalculateButton && (
                <button
                  className="w-full flex items-center justify-center gap-2 p-3 bg-red-600 text-white rounded-md hover:bg-red-700 transform transition duration-300 h-10 mt-2"
                  onClick={() => handleRecalculateRoute(2)}
                >
                  Recalculate Route
                </button>
              )}
            </div>

            {/* Hardcoded Route 3 */}
            <div className="p-3 bg-white rounded-md shadow-md hover:shadow-lg transition mb-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="text-xl font-semibold">Optimal Route 3</h3>
                  {/* <p className="text-gray-700">{route.description}</p> */}
                </div>

                <button
                  className="mr-4 focus:outline-none"
                  onClick={() => {
                    const route = routes.find(route => route.id === 3);
                    updateVisibility(3, !route.visible);
                  }}
                >
                  {routes.find(route => route.id === 3).visible ? (
                    <FaEye className="text-2xl" style={{ color: "#FFA500" }} />
                  ) : (
                    <FaEyeSlash className="text-2xl text-gray-400" />
                  )}
                </button>
              </div>

              <div className="flex items-start gap-4 mt-3">
                {/* Route Data Section */}
                <div
                  className="flex flex-col p-4 bg-white text-sm text-gray-800 rounded-lg shadow-md hover:shadow-lg transition-shadow duration-300 ease-in-out flex-grow"
                  style={{
                    borderLeft: `4px solid #00FFFF`, // Modern border styling
                  }}
                >
                  <div className="mb-2">
                    <span className="font-semibold">Fuel:</span> {120810.09 || 'N/A'}
                  </div>
                  <div className="mb-2">
                    <span className="font-semibold">Duration(hrs):</span> {60.73 || 'N/A'}
                  </div>
                  <div>
                    <span className="font-semibold">Safety Index:</span> {0.26 || 'N/A'}
                  </div>
                </div>

                {/* Position Button */}
                <button
                  className="w-32 p-3 bg-teal-600 text-white rounded-lg hover:bg-teal-700 flex items-center justify-center transition-colors duration-300 ease-in-out shadow-md hover:shadow-lg"
                  style={{
                    borderWidth: "2px",
                    borderColor: "#00FFFF", // Dynamically set the border color
                  }}
                  onClick={() => handleClick(3)} // Ensure route.id is passed
                >
                  Position After 24 Hours
                </button>
              </div>

              {/* Conditionally render the "Recalculate Route" button */}
              {showRecalculateButton && (
                <button
                  className="w-full flex items-center justify-center gap-2 p-3 bg-red-600 text-white rounded-md hover:bg-red-700 transform transition duration-300 h-10 mt-2"
                  onClick={() => handleRecalculateRoute(3)}
                >
                  Recalculate Route
                </button>
              )}
            </div>

              
            

          </div>
        )}
      </aside>
    </div>
  );
};

export default Sidebar;
