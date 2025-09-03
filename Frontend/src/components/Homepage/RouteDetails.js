import React, { useState, useContext } from "react";
import { Transition } from "@headlessui/react"; // Optional for animation
import "tailwindcss/tailwind.css"; // Make sure tailwind is installed
import { ShipContext } from "../../ShipContext";

const Modal = ({ routeType, isOpen, closeModal, mapImages, routeDetails }) => {
  const tempImg = [
    "/wind_speed_map.svg",
    "/wave_height_map.svg",
    "/usurf_map.svg",
    "/vsurf_map.svg",
  ];
  const [currentImageIndex, setCurrentImageIndex] = useState(0);

  const { routeData } = useContext(ShipContext);

  console.log(routeData)

  const goToPreviousImage = () => {
    setCurrentImageIndex((prevIndex) =>
      prevIndex === 0 ? mapImages.length - 1 : prevIndex - 1
    );
  };

  const goToNextImage = () => {
    setCurrentImageIndex((prevIndex) =>
      prevIndex === mapImages.length - 1 ? 0 : prevIndex + 1
    );
  };

  return (
    <Transition show={isOpen}>
      <div className="fixed inset-0 z-50 bg-gray-300 bg-opacity-30 flex items-center justify-center backdrop-blur-sm">
        <div className="relative bg-white bg-opacity-20 backdrop-blur-lg p-8 rounded-xl shadow-2xl w-full h-full md:h-auto md:max-w-7xl">
          <div className="absolute top-0 right-0 pt-4 pr-4">
            <button
              className="text-4xl font-bold text-white hover:text-gray-300 transition"
              onClick={closeModal}
            >
              &times;
            </button>
          </div>

          <div className="flex flex-col md:flex-row gap-6">
            {/* Left side: Map images slider */}
            <div className="relative w-full md:w-5/6">
              <div className="relative w-full h-full overflow-hidden">
                {/* Image Transition using Tailwind CSS classes */}
                <div className="flex transition-transform duration-500">
                  {/* Slide in current image */}
                  <img
                    key={currentImageIndex} // Add key to ensure proper re-render
                    src={tempImg[currentImageIndex]} // Display current image
                    alt={`Map ${currentImageIndex + 1}`}
                    className="w-full h-full object-cover rounded-lg shadow-md"
                  />
                </div>
              </div>
              <button
                onClick={goToPreviousImage}
                className="absolute left-2 top-1/2 transform -translate-y-1/2 bg-gray-900 text-white rounded-full p-2 shadow-lg"
              >
                &lt;
              </button>
              <button
                onClick={goToNextImage}
                className="absolute right-2 top-1/2 transform -translate-y-1/2 bg-gray-900 text-white rounded-full p-2 shadow-lg"
              >
                &gt;
              </button>
            </div>

            {/* Right side: Route details */}
            <div className="w-full md:w-1/2">
              <h2 className="text-3xl font-semibold text-gray-900">Route Details</h2>
              <div className="space-y-4 mt-4">
                <div className="flex justify-between">
                  <p className="text-gray-700">Fuel Used(gallons):</p>
                  {/* Conditionally display based on routeType */}
                  <p className="font-semibold text-gray-900">
                    {routeType === 1 ? routeData.safest_path.fuel : ''}
                    {routeType === 2 ? routeData.fuel_efficient_path.fuel : ''}
                    {routeType === 3 ? routeData.shortest_path.fuel : ''}
                  </p>
                </div>

                <div className="flex justify-between">
                  <p className="text-gray-700">Duration of Route(hrs):</p>
                  {/* Conditionally display based on routeType */}
                  <p className="font-semibold text-gray-900">
                    {routeType === 1 ? routeData.safest_path.time : ''}
                    {routeType === 2 ? routeData.fuel_efficient_path.time : ''}
                    {routeType === 3 ? routeData.shortest_path.time : ''}
                  </p>
                </div>

                <div className="flex justify-between">
                  <p className="text-gray-700">Risk:</p>
                  {/* Conditionally display based on routeType */}
                  <p className="font-semibold text-gray-900">
                    {routeType === 1 ? routeData.safest_path.risk : ''}
                    {routeType === 2 ? routeData.fuel_efficient_path.risk : ''}
                    {routeType === 3 ? routeData.shortest_path.risk : ''}
                  </p>
                </div>
              </div>

              {/* Scrollable coordinates */}
              <div className="mt-6">
                <p className="font-semibold text-gray-900 mb-2">Coordinates:</p>
                <div className="overflow-y-auto max-h-72 bg-white bg-opacity-40 p-4 rounded-lg shadow-lg">
                  {/* Mapping the coordinates array to display each value on a new line */}
                  {routeDetails.coordinates.map((coord, index) => (
                    <p key={index} className="text-sm text-gray-700 whitespace-pre-wrap">
                      {coord}
                    </p>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  );
};

export default Modal;
