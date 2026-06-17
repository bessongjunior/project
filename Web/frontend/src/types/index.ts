export interface MapPoint {
  lat: number;
  lng: number;
  id: string;
  name?: string;
}

export interface RouteTask {
  id: string;
  type: 'pickup' | 'delivery';
  aoi_id: string;
  promised_time: number;
  status: 'pending' | 'en_route' | 'completed';
}

export interface OptimizationResponse {
  optimized_sequence: string[];
  estimated_eta: number;
  road_geometry: number[][]; // [lat, lng] array for OSRM path
}
