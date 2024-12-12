import React, { useState } from 'react';

const Header = ({ onSearch }) => {
  const [imo, setImo] = useState("");

  const handleSubmit = (e) => {
    e.preventDefault();
    onSearch(imo);
  };

  return (
    <div className="pb-0 pt-10  pl-96 pr-96" >

    <header className="bg-teal-600 p-6 mb-6 rounded-lg shadow-md flex flex-col justify-between items-center gap-4 ">
      <h1 className="text-white text-3xl font-bold">Vessel Details and Location</h1>
      <form onSubmit={handleSubmit} className="flex flex-col items-center gap-4 ">
        <input
          type="text"
          value={imo}
          onChange={(e) => setImo(e.target.value)}
          placeholder="Enter IMO number"
          className="p-2 rounded border border-gray-300 focus:outline-none focus:ring-2 focus:ring-indigo-500"
        />
        <button
          
          type="submit"
          className="bg-white text-black p-2 rounded hover:bg-teal-400 transition duration-200 "
          >
          Search
        </button>
      </form>
    </header>
          </div>
  );
};

export default Header;
