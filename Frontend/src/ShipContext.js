import React, { createContext, useState } from "react";

export const ShipContext = createContext();

export const ShipProvider = ({ children }) => {
  const [source, setSource] = useState("");
  const [destination, setDestination] = useState("");
  const [shipDisplacement, setShipDisplacement] = useState("");
  const [frontalArea, setFrontalArea] = useState("");
  const [hullEfficiency, setHullEfficiency] = useState("");
  const [propellerEfficiency, setPropellerEfficiency] = useState("");
  const [engineShaftEfficiency, setEngineShaftEfficiency] = useState("");
  const [heightAboveSea, setHeightAboveSea] = useState("");
  const [resonantPeriod, setResonantPeriod] = useState("");
  const [csfoc, setCsfoc] = useState("");
  const [safetyWeight, setSafetyWeight] = useState("");
  const [distanceWeight, setDistanceWeight] = useState("");
  const [fuelWeight, setFuelWeight] = useState("");
  const [carriageWeight, setCarriageWeight] = useState("");

  const [routeData, setRouteData] = useState([]);

  return (
    <ShipContext.Provider
      value={{
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
        routeData,
        setRouteData,
        carriageWeight,
        setCarriageWeight
      }}
    >
      {children}
    </ShipContext.Provider>
  );
};
