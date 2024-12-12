import React from "react";
import { FaAnchor, FaRoute, FaShip, FaCompass } from "react-icons/fa";
import { Link, useLocation } from "react-router-dom";

const Navbar = () => {
  const location = useLocation();

  // Function to determine if a link should be active
  const isActive = (path) => location.pathname === path;

  return (
    <nav className="flex h-16 bg-teal-600 text-white shadow-md">
      {/* Left Section: Logo */}
      <div className="flex items-center justify-left pl-8 w-1/2">
        <Link to="/">
          <h1 className="text-2xl font-bold flex items-center gap-2">
            <FaAnchor /> SamudraPath
          </h1>
        </Link>
      </div>

      {/* Right Section: Navigation Links */}
      <div className="flex items-center justify-around w-1/2">
        {/* Optimal Route Button */}
        <Link
          to="/"
          className={`flex-1 text-center py-5 hover:bg-teal-500 transition ${
            isActive("/") ? "bg-gray-600" : ""
          }`}
        >
          <FaRoute className="inline mr-2" /> Optimal Route
        </Link>

        {/* Voyage Details Button */}
        <Link
          to="/VoyageDetail"
          className={`flex-1 text-center py-5 hover:bg-teal-500 transition ${
            isActive("/VoyageDetail") ? "bg-gray-600" : ""
          }`}
        >
          <FaCompass className="inline mr-2" /> Voyage Details
        </Link>

        {/* Health Checkup Button */}
        <Link
          to="/shiphealth"
          className={`flex-1 text-center py-5 hover:bg-teal-500 transition ${
            isActive("/shiphealth") ? "bg-gray-600" : ""
          }`}
        >
          <FaShip className="inline mr-2" /> Track Vessel
        </Link>
      </div>
    </nav>
  );
};

export default Navbar;
