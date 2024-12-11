import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./Pages/HomePage";
import ShipHealth from "./Pages/ShipHealth"
import { ShipProvider } from "./ShipContext";
import VoyageDetail from "./Pages/VoyageDetail";

const App = () => {
  return (
    <div className="h-screen">
      {/* Wrap the context provider around the Router */}
      <ShipProvider>
        <Router>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/shiphealth" element={<ShipHealth />} />
            <Route path="/voyagedetail" element={< VoyageDetail/>} />
          </Routes>
        </Router>
      </ShipProvider>
    </div>
  );
};

export default App;
