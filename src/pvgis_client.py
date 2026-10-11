"""
PVGIS API Client for Solar Power Prediction Project
"""

import requests
import pandas as pd
import json
from typing import Dict, Optional, Union
from urllib.parse import urljoin


class PVGISClient:
    """
    Client for interacting with the PVGIS API to fetch solar radiation and PV system data.
    
    Documentation: https://joint-research-centre.ec.europa.eu/photovoltaic-geographical-information-system-pvgis/getting-started-pvgis/api-non-interactive-service_en
    """
    
    def __init__(self, base_url: str = "https://re.jrc.ec.europa.eu/api/v5_3/"):
        """
        Initialize PVGIS client.
        
        Args:
            base_url: Base URL for PVGIS API
        """
        self.base_url = base_url
        self.session = requests.Session()
        
    def _make_request(self, endpoint: str, params: Dict, outputformat: str = "json") -> Union[Dict, str]:
        """
        Make API request to PVGIS.
        
        Args:
            endpoint: API endpoint (e.g., 'PVcalc', 'seriescalc')
            params: Request parameters
            outputformat: Output format ('json', 'csv', 'basic')
            
        Returns:
            API response as dict or string
        """
        params['outputformat'] = outputformat
        
        url = urljoin(self.base_url, endpoint)
        
        try:
            response = self.session.get(url, params=params)
            response.raise_for_status()
            
            if outputformat == "json":
                return response.json()
            else:
                return response.text
                
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {e}")
    
    def get_pv_calculation(self, lat: float, lon: float, peakpower: float = 1.0, 
                          loss: float = 14, angle: Optional[float] = None,
                          aspect: float = 0, **kwargs) -> Dict:
        """
        Calculate PV system energy production.
        
        Args:
            lat: Latitude
            lon: Longitude
            peakpower: Peak power of PV system in kW
            loss: System losses in percentage
            angle: Tilt angle of panels (degrees)
            aspect: Azimuth angle (0=South, 90=West, -90=East)
            **kwargs: Additional parameters
            
        Returns:
            PV calculation results
        """
        params = {
            'lat': lat,
            'lon': lon,
            'peakpower': peakpower,
            'loss': loss,
            'aspect': aspect
        }
        
        if angle is not None:
            params['angle'] = angle
            
        params.update(kwargs)
        
        return self._make_request('PVcalc', params)
    
    def get_hourly_radiation(self, lat: float, lon: float, 
                           startyear: Optional[int] = None,
                           endyear: Optional[int] = None, **kwargs) -> Dict:
        """
        Get hourly solar radiation data.
        
        Args:
            lat: Latitude
            lon: Longitude
            startyear: Start year for data
            endyear: End year for data
            **kwargs: Additional parameters
            
        Returns:
            Hourly radiation data
        """
        params = {
            'lat': lat,
            'lon': lon
        }
        
        if startyear:
            params['startyear'] = startyear
        if endyear:
            params['endyear'] = endyear
            
        params.update(kwargs)
        
        return self._make_request('seriescalc', params)
    
    def get_daily_radiation(self, lat: float, lon: float, month: Optional[int] = None,
                          global_radiation: bool = True, **kwargs) -> Dict:
        """
        Get daily solar radiation data.
        
        Args:
            lat: Latitude
            lon: Longitude
            month: Month (1-12), if None returns all months
            global_radiation: Include global radiation
            **kwargs: Additional parameters
            
        Returns:
            Daily radiation data
        """
        params = {
            'lat': lat,
            'lon': lon
        }
        
        if month:
            params['month'] = month
        if global_radiation:
            params['global'] = 1
            
        params.update(kwargs)
        
        return self._make_request('DRcalc', params)
    
    def get_monthly_radiation(self, lat: float, lon: float, 
                            horizontal_radiation: bool = True, **kwargs) -> Dict:
        """
        Get monthly solar radiation data.
        
        Args:
            lat: Latitude
            lon: Longitude
            horizontal_radiation: Include horizontal radiation
            **kwargs: Additional parameters
            
        Returns:
            Monthly radiation data
        """
        params = {
            'lat': lat,
            'lon': lon
        }
        
        if horizontal_radiation:
            params['horirrad'] = 1
            
        params.update(kwargs)
        
        return self._make_request('MRcalc', params)
    
    def get_tmy_data(self, lat: float, lon: float, **kwargs) -> Dict:
        """
        Get Typical Meteorological Year (TMY) data.
        
        Args:
            lat: Latitude
            lon: Longitude
            **kwargs: Additional parameters
            
        Returns:
            TMY data
        """
        params = {
            'lat': lat,
            'lon': lon
        }
        
        params.update(kwargs)
        
        return self._make_request('tmy', params)
    
    def get_offgrid_pv_calculation(self, lat: float, lon: float, peakpower: float,
                                 batterysize: float, consumptionday: float,
                                 cutoff: float, loss: float = 14, **kwargs) -> Dict:
        """
        Calculate off-grid PV system performance.
        
        Args:
            lat: Latitude
            lon: Longitude
            peakpower: Peak power in kW
            batterysize: Battery size in kWh
            consumptionday: Daily consumption in kWh
            cutoff: Battery cutoff level in percentage
            loss: System losses in percentage
            **kwargs: Additional parameters
            
        Returns:
            Off-grid PV calculation results
        """
        params = {
            'lat': lat,
            'lon': lon,
            'peakpower': peakpower,
            'batterysize': batterysize,
            'consumptionday': consumptionday,
            'cutoff': cutoff,
            'loss': loss
        }
        
        params.update(kwargs)
        
        return self._make_request('SHScalc', params)


def parse_pv_output_to_dataframe(data: Dict) -> pd.DataFrame:
    """
    Parse PVGIS output to pandas DataFrame.
    
    Args:
        data: PVGIS API response
        
    Returns:
        Formatted DataFrame
    """
    if 'outputs' not in data:
        raise ValueError("Invalid PVGIS response format")
    
    outputs = data['outputs']
    
    # Handle different output formats
    if 'hourly' in outputs:
        df = pd.DataFrame(outputs['hourly'])
        df['time'] = pd.to_datetime(df['time'])
        df.set_index('time', inplace=True)
        
    elif 'daily' in outputs:
        df = pd.DataFrame(outputs['daily'])
        df['date'] = pd.to_datetime(df['date'])
        df.set_index('date', inplace=True)
        
    elif 'monthly' in outputs:
        df = pd.DataFrame(outputs['monthly'])
        
    else:
        raise ValueError("No recognizable data format in outputs")
    
    return df


# Example usage
if __name__ == "__main__":
    client = PVGISClient()
    
    # Example: Get PV calculation for a location
    try:
        result = client.get_pv_calculation(lat=45.0, lon=8.0, peakpower=5.0)
        print("PV Calculation Results:")
        print(f"Annual energy production: {result['outputs']['totals']['fixed']['E_y']} kWh/year")
        
        # Convert to DataFrame if hourly data available
        if 'hourly' in result['outputs']:
            df = parse_pv_output_to_dataframe(result)
            print(f"Data shape: {df.shape}")
            
    except Exception as e:
        print(f"Error: {e}")
