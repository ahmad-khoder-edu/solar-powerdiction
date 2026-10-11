"""
Data Fetching Utilities for Solar Power Prediction Project
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple, Optional
import time
import os
from pathlib import Path
import json

from pvgis_client import PVGISClient, parse_pv_output_to_dataframe


class SolarDataFetcher:
    """
    Utility class for fetching and managing solar data from PVGIS API.
    """
    
    def __init__(self, data_dir: str = "data", cache_enabled: bool = True):
        """
        Initialize data fetcher.
        
        Args:
            data_dir: Directory to store cached data
            cache_enabled: Whether to use cached data
        """
        self.client = PVGISClient()
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.cache_enabled = cache_enabled
        
    def _get_cache_path(self, endpoint: str, params: Dict) -> Path:
        """Generate cache file path based on endpoint and parameters."""
        param_str = "_".join([f"{k}_{v}" for k, v in sorted(params.items())])
        filename = f"{endpoint}_{param_str}.json"
        return self.data_dir / filename
    
    def _load_from_cache(self, cache_path: Path) -> Optional[Dict]:
        """Load data from cache if available."""
        if not self.cache_enabled or not cache_path.exists():
            return None
        
        try:
            with open(cache_path, 'r') as f:
                return json.load(f)
        except Exception:
            return None
    
    def _save_to_cache(self, cache_path: Path, data: Dict):
        """Save data to cache."""
        if not self.cache_enabled:
            return
        
        try:
            with open(cache_path, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass
    
    def fetch_location_data(self, lat: float, lon: float, 
                          use_cache: bool = True) -> Dict:
        """
        Fetch comprehensive solar data for a location.
        
        Args:
            lat: Latitude
            lon: Longitude
            use_cache: Whether to use cached data
            
        Returns:
            Dictionary containing all fetched data
        """
        location_data = {
            'coordinates': {'lat': lat, 'lon': lon},
            'pv_calculation': None,
            'hourly_radiation': None,
            'daily_radiation': None,
            'monthly_radiation': None,
            'tmy_data': None
        }
        
        # PV Calculation
        try:
            params = {'lat': lat, 'lon': lon, 'peakpower': 1.0}
            cache_path = self._get_cache_path('PVcalc', params)
            
            if use_cache:
                cached = self._load_from_cache(cache_path)
                if cached:
                    location_data['pv_calculation'] = cached
                else:
                    location_data['pv_calculation'] = self.client.get_pv_calculation(lat, lon)
                    self._save_to_cache(cache_path, location_data['pv_calculation'])
            else:
                location_data['pv_calculation'] = self.client.get_pv_calculation(lat, lon)
                
        except Exception as e:
            print(f"Error fetching PV calculation: {e}")
        
        # Hourly Radiation
        try:
            params = {'lat': lat, 'lon': lon}
            cache_path = self._get_cache_path('seriescalc', params)
            
            if use_cache:
                cached = self._load_from_cache(cache_path)
                if cached:
                    location_data['hourly_radiation'] = cached
                else:
                    location_data['hourly_radiation'] = self.client.get_hourly_radiation(lat, lon)
                    self._save_to_cache(cache_path, location_data['hourly_radiation'])
            else:
                location_data['hourly_radiation'] = self.client.get_hourly_radiation(lat, lon)
                
        except Exception as e:
            print(f"Error fetching hourly radiation: {e}")
        
        # Daily Radiation
        try:
            params = {'lat': lat, 'lon': lon}
            cache_path = self._get_cache_path('DRcalc', params)
            
            if use_cache:
                cached = self._load_from_cache(cache_path)
                if cached:
                    location_data['daily_radiation'] = cached
                else:
                    location_data['daily_radiation'] = self.client.get_daily_radiation(lat, lon)
                    self._save_to_cache(cache_path, location_data['daily_radiation'])
            else:
                location_data['daily_radiation'] = self.client.get_daily_radiation(lat, lon)
                
        except Exception as e:
            print(f"Error fetching daily radiation: {e}")
        
        # Monthly Radiation
        try:
            params = {'lat': lat, 'lon': lon}
            cache_path = self._get_cache_path('MRcalc', params)
            
            if use_cache:
                cached = self._load_from_cache(cache_path)
                if cached:
                    location_data['monthly_radiation'] = cached
                else:
                    location_data['monthly_radiation'] = self.client.get_monthly_radiation(lat, lon)
                    self._save_to_cache(cache_path, location_data['monthly_radiation'])
            else:
                location_data['monthly_radiation'] = self.client.get_monthly_radiation(lat, lon)
                
        except Exception as e:
            print(f"Error fetching monthly radiation: {e}")
        
        return location_data
    
    def fetch_multiple_locations(self, locations: List[Tuple[float, float]], 
                               delay: float = 1.0) -> List[Dict]:
        """
        Fetch data for multiple locations with rate limiting.
        
        Args:
            locations: List of (lat, lon) tuples
            delay: Delay between requests in seconds
            
        Returns:
            List of location data dictionaries
        """
        results = []
        
        for i, (lat, lon) in enumerate(locations):
            print(f"Fetching data for location {i+1}/{len(locations)}: ({lat}, {lon})")
            
            try:
                location_data = self.fetch_location_data(lat, lon)
                results.append(location_data)
                
                # Rate limiting
                if i < len(locations) - 1:
                    time.sleep(delay)
                    
            except Exception as e:
                print(f"Error fetching data for ({lat}, {lon}): {e}")
                continue
        
        return results
    
    def export_to_csv(self, location_data: Dict, output_dir: str = "data"):
        """
        Export location data to CSV files.
        
        Args:
            location_data: Dictionary containing location data
            output_dir: Output directory for CSV files
        """
        output_path = Path(output_dir)
        output_path.mkdir(exist_ok=True)
        
        lat = location_data['coordinates']['lat']
        lon = location_data['coordinates']['lon']
        prefix = f"location_{lat}_{lon}"
        
        # Export hourly data
        if location_data['hourly_radiation']:
            try:
                df = parse_pv_output_to_dataframe(location_data['hourly_radiation'])
                df.to_csv(output_path / f"{prefix}_hourly.csv")
                print(f"Exported hourly data to {prefix}_hourly.csv")
            except Exception as e:
                print(f"Error exporting hourly data: {e}")
        
        # Export daily data
        if location_data['daily_radiation']:
            try:
                df = parse_pv_output_to_dataframe(location_data['daily_radiation'])
                df.to_csv(output_path / f"{prefix}_daily.csv")
                print(f"Exported daily data to {prefix}_daily.csv")
            except Exception as e:
                print(f"Error exporting daily data: {e}")
        
        # Export monthly data
        if location_data['monthly_radiation']:
            try:
                df = parse_pv_output_to_dataframe(location_data['monthly_radiation'])
                df.to_csv(output_path / f"{prefix}_monthly.csv")
                print(f"Exported monthly data to {prefix}_monthly.csv")
            except Exception as e:
                print(f"Error exporting monthly data: {e}")
    
    def get_location_summary(self, location_data: Dict) -> Dict:
        """
        Generate summary statistics for a location.
        
        Args:
            location_data: Dictionary containing location data
            
        Returns:
            Summary statistics
        """
        summary = {
            'coordinates': location_data['coordinates']
        }
        
        # PV calculation summary
        if location_data['pv_calculation']:
            pv_data = location_data['pv_calculation']
            if 'outputs' in pv_data and 'totals' in pv_data['outputs']:
                totals = pv_data['outputs']['totals']['fixed']
                summary['annual_energy'] = totals.get('E_y', 0)
                summary['specific_production'] = totals.get('E_y_per_kWp', 0)
        
        # Hourly radiation summary
        if location_data['hourly_radiation']:
            try:
                df = parse_pv_output_to_dataframe(location_data['hourly_radiation'])
                if 'G(i)' in df.columns:  # Global irradiance
                    summary['avg_hourly_irradiance'] = df['G(i)'].mean()
                    summary['max_hourly_irradiance'] = df['G(i)'].max()
                    summary['total_hours_data'] = len(df)
            except Exception:
                pass
        
        return summary


# Example usage
if __name__ == "__main__":
    fetcher = SolarDataFetcher()
    
    # Example: Fetch data for a single location
    location_data = fetcher.fetch_location_data(lat=45.0, lon=8.0)
    
    # Print summary
    summary = fetcher.get_location_summary(location_data)
    print("Location Summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")
    
    # Export to CSV
    fetcher.export_to_csv(location_data)
