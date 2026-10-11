"""
Data Collection Module for Solar Power Prediction
Handles fetching solar radiation data from PVGIS API and user input processing
"""

import requests
import pandas as pd
import numpy as np
from typing import Optional, Dict, Any
import os
from dotenv import load_dotenv

load_dotenv()


class SolarDataCollector:
    """Collects solar radiation and weather data for Lebanon"""
    
    def __init__(self):
        self.pvgis_base_url = "https://re.jrc.ec.europa.eu/api/v5_2/"
        
    def get_optimal_tilt_angle(self, latitude: Optional[float], 
                             longitude: Optional[float]) -> Optional[float]:
        """
        Get optimal tilt angle from PVGIS API
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            
        Returns:
            Optimal tilt angle in degrees, or None if location missing
        """
        if latitude is None or longitude is None:
            return None
            
        url = f"{self.pvgis_base_url}seriescalc"
        params = {
            'lat': latitude,
            'lon': longitude,
            'outputformat': 'json',
            'radiation': '1',
            'usehorizon': '1',
            'userhorizon': '0',
            'pvtechchoice': 'crystSi',
            'mountingplace': 'free',
            'trackingtype': '0',
            'optimalinclination': '1',
            'components': '1'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract optimal tilt angle from metadata
            metadata = data.get('meta', {})
            optimal_tilt = metadata.get('inclination', {}).get('optimal', {}).get('fixed')
            
            if optimal_tilt is not None:
                return float(optimal_tilt)
            else:
                # Fallback: use latitude-based approximation
                return abs(latitude)
                
        except requests.exceptions.RequestException:
            return None
        except (KeyError, ValueError, TypeError):
            return None

    def fetch_pvgis_data(self, latitude: float, longitude: float, 
                         start_year: int = 2019, end_year: int = 2020) -> pd.DataFrame:
        """
        Fetch solar radiation data from PVGIS API
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees  
            start_year: Start year for data collection (2005-2020)
            end_year: End year for data collection (2005-2020)
            
        Returns:
            DataFrame with daily solar data
        """
        # Validate year range (PVGIS API limitation)
        if not (2005 <= start_year <= 2020):
            raise ValueError(f"start_year must be between 2005 and 2020, got {start_year}")
        if not (2005 <= end_year <= 2020):
            raise ValueError(f"end_year must be between 2005 and 2020, got {end_year}")
        if start_year > end_year:
            raise ValueError(f"start_year ({start_year}) cannot be greater than end_year ({end_year})")
        
        url = f"{self.pvgis_base_url}seriescalc"
        params = {
            'lat': latitude,
            'lon': longitude,
            'startyear': start_year,
            'endyear': end_year,
            'outputformat': 'json',
            'radiation': '1',
            'usehorizon': '1',
            'userhorizon': '0',
            'pvtechchoice': 'crystSi',
            'mountingplace': 'free',
            'trackingtype': '0',
            'optimalinclination': '1',
            'components': '1'
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract hourly data and aggregate to daily
            hourly_data = data.get('outputs', {}).get('hourly', [])
            
            if not hourly_data:
                raise ValueError("No hourly data returned from PVGIS API")
            
            # Convert to DataFrame
            df = pd.DataFrame(hourly_data)
            
            # Convert date column
            df['date'] = pd.to_datetime(df['time'].str.split(':').str[0], format='%Y%m%d')
            
            # Aggregate hourly data to daily values
            daily_columns = {
                'Gb(i)': 'beam_irradiance',
                'Gd(i)': 'diffuse_irradiance', 
                'Gr(i)': 'reflected_irradiance',
                'G(i)': 'global_irradiance',
                'H_sun': 'sunshine_duration',
                'T2m': 'temperature',
                'WS10m': 'wind_speed',
                'Int': 'irradiance_intensity'
            }
            
            # Calculate global irradiance if not present
            if 'G(i)' not in df.columns and all(col in df.columns for col in ['Gb(i)', 'Gd(i)', 'Gr(i)']):
                df['G(i)'] = df['Gb(i)'] + df['Gd(i)'] + df['Gr(i)']
            
            # Aggregate by date
            daily_df = df.groupby('date').agg({
                'Gb(i)': 'sum',
                'Gd(i)': 'sum', 
                'Gr(i)': 'sum',
                'G(i)': 'sum' if 'G(i)' in df.columns else lambda x: (x['Gb(i)'] + x['Gd(i)'] + x['Gr(i)']).sum(),
                'H_sun': 'sum',
                'T2m': 'mean',
                'WS10m': 'mean',
                'Int': 'first'
            }).reset_index()
            
            # Rename columns for clarity
            column_mapping = {
                'Gb(i)': 'beam_irradiance',
                'Gd(i)': 'diffuse_irradiance',
                'Gr(i)': 'reflected_irradiance', 
                'G(i)': 'global_irradiance',
                'H_sun': 'sunshine_duration',
                'T2m': 'temperature',
                'WS10m': 'wind_speed',
                'Int': 'irradiance_intensity'
            }
            
            daily_df = daily_df.rename(columns=column_mapping)
            daily_df = daily_df.set_index('date')
            
            return daily_df
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch data from PVGIS: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error processing PVGIS data: {e}")
    
    def process_user_inputs(self, **kwargs) -> Dict[str, Any]:
        """
        Process optional user inputs
        
        Args:
            **kwargs: Optional parameters including:
                - latitude: Geographic latitude
                - longitude: Geographic longitude
                - battery_capacity: Battery capacity in kWh
                - battery_type: Battery type (lithium, lead_acid, etc.)
                - load: Daily load profile in kWh
                - inverter_logs: Inverter performance data
                
        Returns:
            Dictionary with processed inputs
        """
        processed_inputs = {}
        
        # Lebanon default coordinates (Beirut)
        default_lat, default_lon = 33.8938, 35.5018
        
        processed_inputs['latitude'] = kwargs.get('latitude', default_lat)
        processed_inputs['longitude'] = kwargs.get('longitude', default_lon)
        
        # Optional parameters
        optional_params = ['battery_capacity', 'battery_type', 'load', 'inverter_logs']
        for param in optional_params:
            if param in kwargs and kwargs[param] is not None:
                processed_inputs[param] = kwargs[param]
        
        return processed_inputs


class WeatherDataCollector:
    """Collects weather data for solar prediction"""
    
    def __init__(self):
        self.solcast_api_key = os.getenv('SOLCAST_API_KEY')
        
    def fetch_solcast_historical_data(self, latitude: float, longitude: float,
                                  start_date: str, end_date: str) -> pd.DataFrame:
        """
        Fetch historical daily data from Solcast API
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            
        Returns:
            DataFrame with historical daily data (GHI, cloud opacity, temperature)
        """
        if not self.solcast_api_key:
            print("⚠️ SOLCAST_API_KEY not found, skipping Solcast data fetch")
            return pd.DataFrame()
        
        url = "https://api.solcast.com.au/data/historic/radiation_and_weather"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'start': start_date,
            'end': end_date,
            'format': 'json',
            'api_key': self.solcast_api_key,
            'time_aggregation': 'daily',
            'aggregate': 'daily'  # Ensure daily aggregation
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract historical data
            historical_data = data.get('estimated_actuals', [])
            
            if not historical_data:
                raise ValueError("No historical data returned from Solcast API")
            
            df = pd.DataFrame(historical_data)
            
            # Convert period_end to datetime and set as index
            df['period_end'] = pd.to_datetime(df['period_end'])
            df = df.set_index('period_end')
            
            # Check if data is sub-daily and aggregate to daily
            if len(df) > 0:
                # Check if we have more than one data point per day (indicating sub-daily data)
                date_range = df.index.date
                unique_dates = len(set(date_range))
                total_points = len(df)
                
                if total_points > unique_dates:  # We have sub-daily data
                    # Build aggregation dictionary based on available columns
                    agg_dict = {}
                    if 'ghi' in df.columns:
                        agg_dict['ghi'] = 'sum'
                    if 'cloud_opacity' in df.columns:
                        agg_dict['cloud_opacity'] = 'mean'
                    if 'air_temp' in df.columns:
                        agg_dict['air_temp'] = 'mean'
                    
                    if agg_dict:
                        df = df.resample('D').agg(agg_dict)
            
            # Select and rename relevant columns
            relevant_columns = {
                'ghi': 'GHI',
                'cloud_opacity': 'cloud_opacity',
                'air_temp': 'temperature'
            }
            
            # Filter for available columns
            available_columns = {k: v for k, v in relevant_columns.items() if k in df.columns}
            
            if available_columns:
                df = df[list(available_columns.keys())].rename(columns=available_columns)
                return df
            else:
                print("⚠️ No relevant columns found in Solcast response")
                return pd.DataFrame()
            
        except requests.exceptions.RequestException as e:
            print(f"⚠️ Failed to fetch Solcast historical data: {e}")
            return pd.DataFrame()
        except (KeyError, ValueError) as e:
            print(f"⚠️ Error processing Solcast historical data: {e}")
            return pd.DataFrame()

    def fetch_solcast_forecast_data(self, latitude: float, longitude: float, 
                          days: int = 30) -> pd.DataFrame:
        """
        Fetch weather forecast data from Solcast API
        
        Args:
            latitude: Latitude in decimal degrees
            longitude: Longitude in decimal degrees
            days: Number of days to forecast
            
        Returns:
            DataFrame with weather forecast data
        """
        if not self.solcast_api_key:
            raise ValueError("SOLCAST_API_KEY not found in environment variables")
        
        url = "https://api.solcast.com.au/data/forecast/radiation_and_weather"
        params = {
            'latitude': latitude,
            'longitude': longitude,
            'format': 'json',
            'api_key': self.solcast_api_key,
            'hours': days * 24
        }
        
        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            # Extract forecast data
            forecasts = data.get('forecasts', [])
            
            if not forecasts:
                raise ValueError("No forecast data returned from Solcast API")
            
            df = pd.DataFrame(forecasts)
            df['period_end'] = pd.to_datetime(df['period_end'])
            df = df.set_index('period_end')
            
            return df
            
        except requests.exceptions.RequestException as e:
            raise ConnectionError(f"Failed to fetch Solcast data: {e}")
        except (KeyError, ValueError) as e:
            raise ValueError(f"Error processing Solcast data: {e}")


def fetch_solcast_historical_standalone(latitude: float, longitude: float,
                                       api_key: Optional[str],
                                       start_date: str, end_date: str) -> pd.DataFrame:
    """
    Standalone function to fetch historical daily data from Solcast API
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        api_key: Solcast API key (optional)
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with historical daily data (GHI, cloud opacity, temperature)
    """
    if not api_key:
        print("⚠️ API key missing, skipping Solcast data fetch")
        return pd.DataFrame()
    
    url = "https://api.solcast.com.au/data/historic/radiation_and_weather"
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start': start_date,
        'end': end_date,
        'format': 'json',
        'api_key': api_key,
        'time_aggregation': 'daily',
        'aggregate': 'daily'  # Ensure daily aggregation
    }
    
    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract historical data
        historical_data = data.get('estimated_actuals', [])
        
        if not historical_data:
            print("⚠️ No historical data returned from Solcast API")
            return pd.DataFrame()
        
        df = pd.DataFrame(historical_data)
        
        # Convert period_end to datetime and set as index
        df['period_end'] = pd.to_datetime(df['period_end'])
        df = df.set_index('period_end')
        
        # Check if data is sub-daily and aggregate to daily
        if len(df) > 0:
            # Check if we have more than one data point per day (indicating sub-daily data)
            date_range = df.index.date
            unique_dates = len(set(date_range))
            total_points = len(df)
            
            if total_points > unique_dates:  # We have sub-daily data
                # Build aggregation dictionary based on available columns
                agg_dict = {}
                if 'ghi' in df.columns:
                    agg_dict['ghi'] = 'sum'
                if 'cloud_opacity' in df.columns:
                    agg_dict['cloud_opacity'] = 'mean'
                if 'air_temp' in df.columns:
                    agg_dict['air_temp'] = 'mean'
                
                if agg_dict:
                    df = df.resample('D').agg(agg_dict)
        
        # Select and rename relevant columns
        relevant_columns = {
            'ghi': 'GHI',
            'cloud_opacity': 'cloud_opacity', 
            'air_temp': 'temperature'
        }
        
        # Filter for available columns
        available_columns = {k: v for k, v in relevant_columns.items() if k in df.columns}
        
        if available_columns:
            df = df[list(available_columns.keys())].rename(columns=available_columns)
            return df
        else:
            print("⚠️ No relevant columns found in Solcast response")
            return pd.DataFrame()
        
    except requests.exceptions.RequestException as e:
        print(f"⚠️ Failed to fetch Solcast historical data: {e}")
        return pd.DataFrame()
    except (KeyError, ValueError) as e:
        print(f"⚠️ Error processing Solcast historical data: {e}")
        return pd.DataFrame()


def get_optimal_tilt_angle_standalone(latitude: Optional[float], 
                                      longitude: Optional[float]) -> Optional[float]:
    """
    Standalone function to get optimal tilt angle from PVGIS API
    
    Args:
        latitude: Latitude in decimal degrees
        longitude: Longitude in decimal degrees
        
    Returns:
        Optimal tilt angle in degrees, or None if location missing
    """
    if latitude is None or longitude is None:
        return None
        
    base_url = "https://re.jrc.ec.europa.eu/api/v5_2/seriescalc"
    params = {
        'lat': latitude,
        'lon': longitude,
        'outputformat': 'json',
        'radiation': '1',
        'usehorizon': '1',
        'userhorizon': '0',
        'pvtechchoice': 'crystSi',
        'mountingplace': 'free',
        'trackingtype': '0',
        'optimalinclination': '1',
        'components': '1'
    }
    
    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        data = response.json()
        
        # Extract optimal tilt angle from metadata
        metadata = data.get('meta', {})
        optimal_tilt = metadata.get('inclination', {}).get('optimal', {}).get('fixed')
        
        if optimal_tilt is not None:
            return float(optimal_tilt)
        else:
            # Fallback: use latitude-based approximation
            return abs(latitude)
            
    except requests.exceptions.RequestException:
        return None
    except (KeyError, ValueError, TypeError):
        return None


def collect_training_data(latitude: float = 33.8938, longitude: float = 35.5018,
                         start_year: int = 2020, end_year: int = 2023) -> pd.DataFrame:
    """
    Main function to collect all training data
    
    Args:
        latitude: Geographic latitude (default: Beirut, Lebanon)
        longitude: Geographic longitude (default: Beirut, Lebanon)
        start_year: Start year for historical data
        end_year: End year for historical data
        
    Returns:
        Combined DataFrame with all collected data
    """
    collector = SolarDataCollector()
    
    # Fetch historical solar data
    solar_data = collector.fetch_pvgis_data(latitude, longitude, start_year, end_year)
    
    print(f"Collected {len(solar_data)} days of solar data")
    print(f"Date range: {solar_data.index.min()} to {solar_data.index.max()}")
    
    return solar_data


if __name__ == "__main__":
    # Example usage
    try:
        # Collect data for Beirut, Lebanon
        data = collect_training_data()
        print("Data collection completed successfully")
        print(f"Data shape: {data.shape}")
        print(f"Columns: {list(data.columns)}")
        
    except Exception as e:
        print(f"Error during data collection: {e}")
