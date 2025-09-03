import React, { useState } from "react";
import Map, { Source, Layer, Marker, NavigationControl, Popup } from "react-map-gl";
import "mapbox-gl/dist/mapbox-gl.css";

const MapView = ({ handleMapClick, routes = [], pirateCoordinates = [] }) => {
  const [isMapLoaded, setIsMapLoaded] = useState(false);
  const [hoverInfo, setHoverInfo] = useState(null);

  const handleMapLoad = () => {
    setIsMapLoaded(true);
  };

  const handlePointHover = (event) => {
    const feature = event.features && event.features[0];
    if (feature) {
      const { coordinates } = feature.geometry;
      setHoverInfo({
        longitude: coordinates[0],
        latitude: coordinates[1],
        coordinates: coordinates,
      });
    } else {
      setHoverInfo(null);
    }
  };

  return (
    <Map
      initialViewState={{
        longitude: 74.5,
        latitude: 10.5,
        zoom: 2.5,
      }}
      style={{ width: "100%", height: "100%" }}
      mapStyle="mapbox://styles/jinx83/cm4jtqz1x006201sib07gdzhn"
      mapboxAccessToken={process.env.REACT_APP_MAPBOX_TOKEN}
      onClick={handleMapClick}
      onLoad={handleMapLoad}
      interactiveLayerIds={routes.map((_, index) => `route-points-layer-${index}`)}
      onMouseMove={handlePointHover}
    >
      {isMapLoaded && (
        <>
          {/* Render Routes and Highlight Points */}
          {Array.isArray(routes) &&
            routes.map((route, index) => {
              if (!route.visible || !Array.isArray(route.coordinates) || route.coordinates.length === 0) {
                return null;
              }

              const firstPoint = route.coordinates[0];
              const lastPoint = route.coordinates[route.coordinates.length - 1];

              return (
                <React.Fragment key={index}>
                  <Source
                    id={`route-${index}`}
                    type="geojson"
                    data={{
                      type: "Feature",
                      geometry: {
                        type: "LineString",
                        coordinates: route.coordinates,
                      },
                    }}
                  >
                    <Layer
                      id={`route-layer-${index}`}
                      type="line"
                      paint={{
                        "line-color": route.color || "#FFFF44",
                        "line-width": 4,
                      }}
                    />
                  </Source>

                  <Source
                    id={`route-points-${index}`}
                    type="geojson"
                    data={{
                      type: "FeatureCollection",
                      features: route.coordinates.map((coord) => ({
                        type: "Feature",
                        geometry: {
                          type: "Point",
                          coordinates: coord,
                        },
                      })),
                    }}
                  >
                    <Layer
                      id={`route-points-layer-${index}`}
                      type="circle"
                      paint={{
                        "circle-color":   "#fff",
                        // "circle-color": route.color || "#000000",
                        "circle-radius": 2,
                      }}
                    />
                  </Source>

                  <Marker longitude={firstPoint[0]} latitude={firstPoint[1]} anchor="bottom">
                    <img
                      src="./starting.png"
                      alt="Start Point"
                      style={{ width: "32px", height: "32px" }}
                    />
                  </Marker>
                  <Marker longitude={lastPoint[0]} latitude={lastPoint[1]} anchor="bottom">
                    <img
                      src="./ending.png"
                      alt="End Point"
                      style={{ width: "32px", height: "32px" }}
                    />
                  </Marker>
                </React.Fragment>
              );
            })}
        </>
      )}

      {/* Display Popup on Hover */}
      {hoverInfo && (
        <Popup
          longitude={hoverInfo.longitude}
          latitude={hoverInfo.latitude}
          closeButton={false}
          closeOnClick={false}
          anchor="top"
        >
          <div>Coordinates: {hoverInfo.coordinates.join(", ")}</div>
        </Popup>
      )}

      <NavigationControl />
    </Map>
  );
};

export default MapView;
