"""
Feature Engineering Module for Solar Power Prediction
Creates features from raw solar and weather data for ML model training
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')


class SolarFeatureEngineer:
    """Engineers features for solar power prediction model"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_columns = []
        
    def create_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create time-based features from datetime index
        
        Args:
            df: DataFrame with datetime index
            
        Returns:
            DataFrame with added time features
        """
        df = df.copy()
        
        # Basic time features
        df['year'] = df.index.year
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['day_of_year'] = df.index.dayofyear
        df['week_of_year'] = df.index.isocalendar().week
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding for seasonal patterns
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Season indicators
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
        
        return df
    
    def create_weather_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create weather-related features
        
        Args:
            df: DataFrame with weather data
            
        Returns:
            DataFrame with added weather features
        """
        df = df.copy()
        
        # Temperature-based features
        if 'temperature' in df.columns:
            df['temp_squared'] = df['temperature'] ** 2
            df['temp_cubed'] = df['temperature'] ** 3
            
            # Temperature bins
            df['temp_category'] = pd.cut(df['temperature'], 
                                       bins=[-np.inf, 10, 20, 30, np.inf],
                                       labels=['cold', 'mild', 'warm', 'hot'])
            df = pd.get_dummies(df, columns=['temp_category'], prefix='temp')
        
        # Wind speed features
        if 'wind_speed' in df.columns:
            df['wind_speed_squared'] = df['wind_speed'] ** 2
            df['is_windy'] = (df['wind_speed'] > 5).astype(int)
        
        # Sunshine duration features
        if 'sunshine_duration' in df.columns:
            df['sunshine_ratio'] = df['sunshine_duration'] / 24  # Ratio of day
            df['is_sunny'] = (df['sunshine_duration'] > 8).astype(int)
        
        return df
    
    def create_lag_features(self, df: pd.DataFrame, target_col: str = 'global_irradiance',
                           lags: List[int] = [1, 2, 3, 7]) -> pd.DataFrame:
        """
        Create lag features for time series modeling
        
        Args:
            df: DataFrame with time series data
            target_col: Target column for lag features
            lags: List of lag periods in days
            
        Returns:
            DataFrame with lag features
        """
        df = df.copy()
        
        for lag in lags:
            df[f'{target_col}_lag_{lag}'] = df[target_col].shift(lag)
        
        # Rolling statistics
        for window in [3, 7, 14]:
            df[f'{target_col}_rolling_mean_{window}'] = (
                df[target_col].rolling(window=window).mean()
            )
            df[f'{target_col}_rolling_std_{window}'] = (
                df[target_col].rolling(window=window).std()
            )
            df[f'{target_col}_rolling_max_{window}'] = (
                df[target_col].rolling(window=window).max()
            )
            df[f'{target_col}_rolling_min_{window}'] = (
                df[target_col].rolling(window=window).min()
            )
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between variables
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Weather interactions
        if 'temperature' in df.columns and 'global_irradiance' in df.columns:
            df['temp_irradiance'] = df['temperature'] * df['global_irradiance']
        
        if 'wind_speed' in df.columns and 'temperature' in df.columns:
            df['wind_temp'] = df['wind_speed'] * df['temperature']
        
        # Seasonal interactions
        if 'month_sin' in df.columns and 'global_irradiance' in df.columns:
            df['season_irradiance'] = df['month_sin'] * df['global_irradiance']
        
        return df
    
    def create_battery_features(self, df: pd.DataFrame, battery_capacity: Optional[float] = None,
                              battery_type: Optional[str] = None) -> pd.DataFrame:
        """
        Create battery-related features if battery data is available
        
        Args:
            df: DataFrame with solar data
            battery_capacity: Battery capacity in kWh
            battery_type: Type of battery (lithium, lead_acid, etc.)
            
        Returns:
            DataFrame with battery features
        """
        df = df.copy()
        
        if battery_capacity is not None:
            df['battery_capacity'] = battery_capacity
            df['energy_storage_ratio'] = df['global_irradiance'] / battery_capacity
            
            # Battery efficiency based on type
            efficiency_map = {
                'lithium': 0.95,
                'lead_acid': 0.80,
                'nickel': 0.85,
                'flow': 0.90
            }
            
            if battery_type:
                efficiency = efficiency_map.get(battery_type.lower(), 0.85)
                df['battery_efficiency'] = efficiency
                df['usable_capacity'] = battery_capacity * efficiency
        
        return df
    
    def create_load_features(self, df: pd.DataFrame, load_profile: Optional[float] = None) -> pd.DataFrame:
        """
        Create load-related features if load data is available
        
        Args:
            df: DataFrame with solar data
            load_profile: Daily load in kWh
            
        Returns:
            DataFrame with load features
        """
        df = df.copy()
        
        if load_profile is not None:
            df['daily_load'] = load_profile
            df['solar_load_ratio'] = df['global_irradiance'] / load_profile
            df['excess_energy'] = np.maximum(0, df['global_irradiance'] - load_profile)
            df['energy_deficit'] = np.maximum(0, load_profile - df['global_irradiance'])
        
        return df
    
    def engineer_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Main feature engineering pipeline
        
        Args:
            df: Input DataFrame with raw data
            **kwargs: Optional parameters (battery_capacity, battery_type, load)
            
        Returns:
            DataFrame with engineered features
        """
        print("Starting feature engineering...")
        
        # Create time features
        df = self.create_time_features(df)
        print("✓ Time features created")
        
        # Create weather features
        df = self.create_weather_features(df)
        print("✓ Weather features created")
        
        # Create lag features
        if 'global_irradiance' in df.columns:
            df = self.create_lag_features(df)
            print("✓ Lag features created")
        
        # Create interaction features
        df = self.create_interaction_features(df)
        print("✓ Interaction features created")
        
        # Create battery features if provided
        if 'battery_capacity' in kwargs:
            df = self.create_battery_features(df, kwargs['battery_capacity'], 
                                            kwargs.get('battery_type'))
            print("✓ Battery features created")
        
        # Create load features if provided
        if 'load' in kwargs:
            df = self.create_load_features(df, kwargs['load'])
            print("✓ Load features created")
        
        # Remove rows with NaN values (from lag features)
        initial_rows = len(df)
        df = df.dropna()
        final_rows = len(df)
        print(f"✓ Removed {initial_rows - final_rows} rows with NaN values")
        
        print(f"Feature engineering completed. Final shape: {df.shape}")
        return df
    
    def prepare_features_and_target(self, df: pd.DataFrame, 
                                  target_col: str = 'global_irradiance') -> tuple:
        """
        Prepare features and target for ML training
        
        Args:
            df: DataFrame with engineered features
            target_col: Target column name
            
        Returns:
            Tuple of (X, y) where X is features and y is target
        """
        # Remove non-feature columns
        exclude_cols = [target_col, 'year', 'month', 'day', 'day_of_year', 
                       'week_of_year', 'day_of_week', 'quarter']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_columns = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        print(f"Prepared {X.shape[1]} features for {len(X)} samples")
        
        return X, y
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> np.ndarray:
        """
        Scale features using StandardScaler
        
        Args:
            X: Feature matrix
            fit: Whether to fit the scaler
            
        Returns:
            Scaled feature matrix
        """
        if fit:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        return X_scaled


def analyze_feature_correlation(df: pd.DataFrame, target_col: str = 'global_irradiance',
                               threshold: float = 0.1) -> pd.DataFrame:
    """
    Analyze correlation between features and target
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        threshold: Minimum absolute correlation threshold
        
    Returns:
        DataFrame with feature correlations
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Get correlations with target
    target_corr = corr_matrix[target_col].abs().sort_values(ascending=False)
    
    # Filter by threshold
    significant_features = target_corr[target_corr >= threshold]
    
    print(f"Found {len(significant_features)} features with correlation >= {threshold}")
    
    return significant_features


def calculate_correlation_matrix(df: pd.DataFrame, target_col: str = 'global_irradiance') -> pd.DataFrame:
    """
    Calculate full correlation matrix for visualization
    
    Args:
        df: DataFrame with features and target
        target_col: Target column name
        
    Returns:
        Full correlation matrix
    """
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    print(f"Calculated correlation matrix with shape: {corr_matrix.shape}")
    
    return corr_matrix


def calculate_feature_covariance(X: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate covariance matrix for features
    
    Args:
        X: Feature matrix
        
    Returns:
        Covariance matrix
    """
    cov_matrix = X.cov()
    return cov_matrix


if __name__ == "__main__":
    # Example usage
    from data_collection import collect_training_data
    
    try:
        # Collect sample data
        print("Collecting sample data...")
        data = collect_training_data()
        
        # Initialize feature engineer
        engineer = SolarFeatureEngineer()
        
        # Engineer features
        features_df = engineer.engineer_features(data)
        
        # Analyze correlations
        correlations = analyze_feature_correlation(features_df)
        print("\nTop correlations with target:")
        print(correlations.head(10))
        
        # Prepare features and target
        X, y = engineer.prepare_features_and_target(features_df)
        
        print(f"\nFeature matrix shape: {X.shape}")
        print(f"Target vector shape: {y.shape}")
        
    except Exception as e:
        print(f"Error in feature engineering: {e}")
