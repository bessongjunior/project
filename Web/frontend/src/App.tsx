import React, { useState, useEffect } from 'react';
import { RouteTask, MapPoint } from './types';
import MapSimulation from './components/MapSimulation';

const App: React.FC = () => {
  const [tasks, setTasks] = useState<RouteTask[]>([]);
  const [optimizedRoute, setOptimizedRoute] = useState<MapPoint[]>([]);

  const handleOptimization = async () => {
    // This will trigger the FastAPI optimize endpoint
    console.log("Re-calculating GCN route...");
  };

  return (
    <div className="flex h-screen bg-gray-900 text-white font-sans">
      {/* Sidebar for Task Management */}
      <div className="w-80 bg-gray-800 border-r border-gray-700 p-4">
        <h1 className="text-xl font-bold mb-6 text-blue-400">SmartPick Dashboard</h1>
        <button 
          onClick={handleOptimization}
          className="w-full bg-blue-600 hover:bg-blue-500 py-2 rounded font-medium transition"
        >
          Optimize Current Sequence
        </button>
        <div className="mt-8 space-y-4">
          <p className="text-sm text-gray-400">Next Pickups (AOI Clusters)</p>
          {/* Mapping tasks here */}
        </div>
      </div>

      {/* Main Map Viewport */}
      <div className="flex-1 relative">
        <MapSimulation 
          points={optimizedRoute} 
          center={[29.563, 106.551]} // Default to Chongqing
        />
      </div>
    </div>
  );
};

export default App;
