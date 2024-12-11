import React, { useState } from "react";
import Map, { Source, Layer, NavigationControl } from "react-map-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const MapView = ({ handleMapClick, routes, pirateCoordinates }) => {
  const [isMapLoaded, setIsMapLoaded] = useState(false);

  const handleMapLoad = () => {
    setIsMapLoaded(true);
  };

  return (
    <Map
      initialViewState={{
        longitude: 74.5,
        latitude: 10.5,
        zoom: 2.5,
      }}
      style={{ width: "100%", height: "100%" }}
      // mapStyle="mapbox://styles/jinx83/cm4daovqb01km01si5zzlhdc4" // Default map
      //  mapStyle="mapbox://styles/mapbox/streets-v11"   // 
      mapStyle="mapbox://styles/jinx83/cm4jtqz1x006201sib07gdzhn"   
      mapboxAccessToken={process.env.REACT_APP_MAPBOX_TOKEN}
      onClick={handleMapClick}
      onLoad={handleMapLoad} // Triggered when the map and style are fully loaded
    >
      {isMapLoaded && (
        <>
          <Source
            id="pirates"
            type="geojson"
            data={{
              type: "FeatureCollection",
              features: pirateCoordinates.map((coord) => ({
                type: "Feature",
                geometry: {
                  type: "Point",
                  coordinates: coord,
                },
              })),
            }}
          >
            
          </Source>
          {routes.map(
            (route, index) =>
              route.visible && (
                <Source
                  key={index}
                  id={`route-${index}`}
                  type="geojson"
                  data={{
                    type: "Feature",
                    geometry: {
                      type: "LineString",
                      coordinates: route.coordinates, // Use the route coordinates here
                    },
                  }}
                >
                  <Layer
                    id={`route-layer-${index}`}
                    type="line"
                    paint={{
                      "line-color": route.color || "#FF5733", // You can pass different colors for each route
                      "line-width": 4,
                    }}
                  />
                </Source>
              )
          )}
        </>
      )}
      <NavigationControl />
    </Map>
  );
};

export default MapView;
