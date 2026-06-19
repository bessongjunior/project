# -*- coding: utf-8 -*-
"""
Map Data Processor (v2.1.0, Feb 2026)
This module automates the extraction of real-world road network graphs from OpenStreetMap (OSM) for specific delivery regions.
It bridges the gap between raw coordinates and actual street topology for advanced GCN-based route optimization.
"""

import os
import osmnx as ox
import pandas as pd
import numpy as np
import pickle
from tqdm import tqdm
from research_impl.pre_processing.utils import dir_check, ws

# Configuration for OSMnx 2.1.0 (Feb 2026 Release)
# Ensuring high-fidelity topology and speed metadata for logistics modeling
ox.settings.use_cache = True
ox.settings.log_console = False

class CityMapExtractor:
    def __init__(self, output_dir=os.path.join(ws, 'research_impl/processed')):
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def get_road_network(self, city_name, network_type='drive'):
        """
        Fetches the road network for a specified city and saves it as a high-performance graph object.
        :param city_name: String (e.g., 'Chongqing, China' or 'Shanghai, China')
        :param network_type: 'drive', 'bike', or 'walk' depending on delivery mode
        """
        print(f"[*] Extracting road network for: {city_name}...")
        try:
            # Download the graph from OSM using the Feb 2026 v2.1.0 API
            # This captures all drivable edges with geometry and metadata (maxspeed, lanes, etc.)
            G = ox.graph_from_place(city_name, network_type=network_type, simplify=True)
            
            # Impute missing speeds and calculate travel times based on Feb 2026 road standards
            G = ox.add_edge_speeds(G)
            G = ox.add_edge_travel_times(G)
            
            # Save the graph as a pickle for fast GCN loading
            safe_name = city_name.split(',')[0].strip().lower().replace(" ", "_")
            file_path = os.path.join(self.output_dir, f"{safe_name}_graph.pkl")
            
            with open(file_path, 'wb') as f:
                pickle.dump(G, f)
            
            print(f"[+] Successfully saved {city_name} graph to {file_path}")
            return G
        except Exception as e:
            print(f"[!] Error processing {city_name}: {str(e)}")
            return None

    def map_points_to_nodes(self, G, df, lat_col='lat', lon_col='lng'):
        """
        Matches delivery coordinates to the nearest physical road network nodes.
        :param G: NetworkX multidigraph from OSM
        :param df: DataFrame containing delivery locations
        :return: Array of node IDs matching the physical street network
        """
        print("[*] Snapping delivery points to nearest road nodes...")
        # v2.1.0 optimized nearest_nodes for batch processing
        nodes = ox.nearest_nodes(G, X=df[lon_col].values, Y=df[lat_col].values)
        return nodes

def process_all_cities(cities):
    """
    Main entry point to process all target cities in the project scope.
    :param cities: List of city strings
    """
    extractor = CityMapExtractor()
    for city in cities:
        G = extractor.get_road_network(city)
        
        # Example: If raw data exists for this city, we could snap it here
        # For now, we focus on establishing the physical infrastructure (the maps)
        if G is not None:
            print(f"[#] {city} network: {len(G.nodes)} nodes, {len(G.edges)} segments.")

if __name__ == "__main__":
    # Define target cities for the research project (Minimum 5 cities)
    target_cities = [
        "Chongqing, China",
        "Shanghai, China",
        "Jilin, China",
        "Yantai, China",
        "Hangzhou, China"
    ]
    process_all_cities(target_cities)
