"""
Dataset Builder Module for Solar Power Prediction
Integrates multiple data sources and creates comprehensive datasets
"""

import pandas as pd
import numpy as np
import os
from typing import Optional, Tuple, Dict, Any
import matplotlib.pyplot as plt
import seaborn as sns
from data_collection import fetch_solcast_historical_standalone, get_optimal_tilt_angle_standalone
from feature_engineering import SolarFeatureEngineer
import warnings
warnings.filterwarnings('ignore')


class DatasetBuilder:
    """
    Builds comprehensive datasets for solar power prediction
    """
    
    def __init__(self):
        self.feature_engineer = SolarFeatureEngineer()
        self.original_dataset = None
        self.cleaned_dataset = None
        self.covariance_matrix = None
        self.correlation_matrix = None
        
    def build_dataset(self, latitude: float, longitude: float,
                     start_date: str, end_date: str,
                     api_key: Optional[str] = None,
                     save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build comprehensive dataset by merging multiple data sources
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            api_key: Solcast API key (optional)
            save_path: Path to save cleaned dataset (optional)
            
        Returns:
            Dictionary with dataset and analysis results
        """
        print("🏗️ Building comprehensive solar dataset...")
        print("=" * 50)
        
        # Step 1: Fetch Solcast weather data
        print("📡 Step 1: Fetching Solcast weather data...")
        weather_data = self._fetch_weather_data(latitude, longitude, start_date, end_date, api_key)
        
        if weather_data.empty:
            raise ValueError("No weather data available. Cannot build dataset.")
        
        print(f"✓ Fetched {len(weather_data)} days of weather data")
        
        # Step 2: Get optimal tilt angle
        print("📐 Step 2: Calculating optimal tilt angle...")
        tilt_angle = self._get_tilt_angle(latitude, longitude)
        
        if tilt_angle is not None:
            print(f"✓ Optimal tilt angle: {tilt_angle:.1f}°")
        else:
            print("⚠️ Could not determine optimal tilt angle, using latitude")
            tilt_angle = abs(latitude)
        
        # Step 3: Add tilt angle to dataset
        weather_data['tilt_angle'] = tilt_angle
        
        # Step 4: Add time features
        print("⏰ Step 4: Creating time features...")
        weather_data = self._add_time_features(weather_data)
        print(f"✓ Added time features, dataset shape: {weather_data.shape}")
        
        # Step 5: Store original copy
        print("💾 Step 5: Storing original dataset...")
        self.original_dataset = weather_data.copy()
        print(f"✓ Original dataset stored: {self.original_dataset.shape}")
        
        # Step 6: Clean missing values
        print("🧹 Step 6: Cleaning missing values...")
        self.cleaned_dataset = self._clean_missing_values(weather_data)
        print(f"✓ Cleaned dataset: {self.cleaned_dataset.shape}")
        
        # Step 7: Compute covariance matrix
        print("📊 Step 7: Computing covariance matrix...")
        self.covariance_matrix = self._compute_covariance_matrix(self.cleaned_dataset)
        print(f"✓ Covariance matrix: {self.covariance_matrix.shape}")
        
        # Step 8: Compute correlation matrix
        print("🔗 Step 8: Computing correlation matrix...")
        self.correlation_matrix = self._compute_correlation_matrix(self.cleaned_dataset)
        print(f"✓ Correlation matrix: {self.correlation_matrix.shape}")
        
        # Step 9: Save cleaned dataset
        if save_path:
            print("💾 Step 9: Saving cleaned dataset...")
            self._save_dataset(self.cleaned_dataset, save_path)
            print(f"✓ Dataset saved to: {save_path}")
        
        # Step 10: Generate summary report
        print("📋 Step 10: Generating summary report...")
        summary = self._generate_summary()
        
        print("\n🎉 Dataset building completed successfully!")
        
        return {
            'original_dataset': self.original_dataset,
            'cleaned_dataset': self.cleaned_dataset,
            'covariance_matrix': self.covariance_matrix,
            'correlation_matrix': self.correlation_matrix,
            'summary': summary,
            'tilt_angle': tilt_angle
        }
    
    def _fetch_weather_data(self, latitude: float, longitude: float,
                          start_date: str, end_date: str,
                          api_key: Optional[str]) -> pd.DataFrame:
        """Fetch weather data from Solcast API"""
        
        if api_key is None:
            api_key = os.getenv('SOLCAST_API_KEY')
        
        weather_data = fetch_solcast_historical_standalone(
            latitude, longitude, api_key, start_date, end_date
        )
        
        return weather_data
    
    def _get_tilt_angle(self, latitude: float, longitude: float) -> Optional[float]:
        """Get optimal tilt angle"""
        return get_optimal_tilt_angle_standalone(latitude, longitude)
    
    def _add_time_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add time features to dataset"""
        df = df.copy()
        
        # Basic time features
        df['day_of_year'] = df.index.dayofyear
        df['month'] = df.index.month
        df['day'] = df.index.day
        df['year'] = df.index.year
        df['week_of_year'] = df.index.isocalendar().week
        df['day_of_week'] = df.index.dayofweek
        df['quarter'] = df.index.quarter
        
        # Cyclical encoding
        df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
        df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
        df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
        df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
        df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
        df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
        
        # Season indicators
        df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
        df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
        df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
        df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
        
        return df
    
    def _clean_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean missing values from dataset"""
        df_clean = df.copy()
        
        # Report missing values before cleaning
        missing_before = df_clean.isnull().sum()
        print(f"   Missing values before cleaning:")
        for col, missing in missing_before.items():
            if missing > 0:
                print(f"     {col}: {missing} ({missing/len(df_clean)*100:.1f}%)")
        
        # Strategy for different types of columns
        numeric_columns = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numeric_columns:
            if df_clean[col].isnull().sum() > 0:
                if col in ['GHI', 'temperature']:
                    # For key weather variables, use interpolation
                    df_clean[col] = df_clean[col].interpolate(method='linear')
                elif col in ['day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']:
                    # For cyclical features, use forward fill then backward fill
                    df_clean[col] = df_clean[col].fillna(method='ffill').fillna(method='bfill')
                else:
                    # For other features, use median
                    df_clean[col] = df_clean[col].fillna(df_clean[col].median())
        
        # Remove any remaining rows with missing values
        initial_rows = len(df_clean)
        df_clean = df_clean.dropna()
        final_rows = len(df_clean)
        
        if initial_rows != final_rows:
            print(f"   Removed {initial_rows - final_rows} rows with remaining missing values")
        
        # Report missing values after cleaning
        missing_after = df_clean.isnull().sum()
        if missing_after.sum() == 0:
            print("   ✓ No missing values remaining")
        else:
            print(f"   Missing values after cleaning:")
            for col, missing in missing_after.items():
                if missing > 0:
                    print(f"     {col}: {missing}")
        
        return df_clean
    
    def _compute_covariance_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute covariance matrix"""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.cov()
    
    def _compute_correlation_matrix(self, df: pd.DataFrame) -> pd.DataFrame:
        """Compute correlation matrix"""
        numeric_df = df.select_dtypes(include=[np.number])
        return numeric_df.corr()
    
    def _save_dataset(self, df: pd.DataFrame, save_path: str):
        """Save dataset to file"""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        
        # Save as CSV
        df.to_csv(save_path)
        
        # Also save as pickle for preserving data types
        pickle_path = save_path.replace('.csv', '.pkl')
        df.to_pickle(pickle_path)
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary statistics"""
        if self.cleaned_dataset is None:
            return {}
        
        summary = {
            'dataset_shape': self.cleaned_dataset.shape,
            'date_range': {
                'start': self.cleaned_dataset.index.min(),
                'end': self.cleaned_dataset.index.max()
            },
            'columns': list(self.cleaned_dataset.columns),
            'numeric_columns': list(self.cleaned_dataset.select_dtypes(include=[np.number]).columns),
            'missing_values': self.cleaned_dataset.isnull().sum().to_dict(),
            'data_types': self.cleaned_dataset.dtypes.to_dict()
        }
        
        # Add basic statistics for numeric columns
        numeric_cols = self.cleaned_dataset.select_dtypes(include=[np.number]).columns
        summary['statistics'] = self.cleaned_dataset[numeric_cols].describe().to_dict()
        
        return summary
    
    def plot_correlation_heatmap(self, figsize: Tuple[int, int] = (12, 10),
                               save_path: Optional[str] = None):
        """Plot correlation heatmap"""
        if self.correlation_matrix is None:
            print("❌ Correlation matrix not computed. Run build_dataset() first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(self.correlation_matrix, dtype=bool))
        
        # Draw heatmap
        sns.heatmap(self.correlation_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_covariance_heatmap(self, figsize: Tuple[int, int] = (12, 10),
                               save_path: Optional[str] = None):
        """Plot covariance heatmap"""
        if self.covariance_matrix is None:
            print("❌ Covariance matrix not computed. Run build_dataset() first.")
            return
        
        plt.figure(figsize=figsize)
        
        # Draw heatmap
        sns.heatmap(self.covariance_matrix, 
                   annot=True, 
                   cmap='viridis', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8})
        
        plt.title('Feature Covariance Heatmap')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def get_high_correlations(self, threshold: float = 0.7) -> pd.DataFrame:
        """Get pairs of highly correlated features"""
        if self.correlation_matrix is None:
            print("❌ Correlation matrix not computed. Run build_dataset() first.")
            return pd.DataFrame()
        
        # Find correlations above threshold
        high_corr = []
        
        for i in range(len(self.correlation_matrix.columns)):
            for j in range(i+1, len(self.correlation_matrix.columns)):
                corr_value = self.correlation_matrix.iloc[i, j]
                if abs(corr_value) >= threshold:
                    high_corr.append({
                        'feature_1': self.correlation_matrix.columns[i],
                        'feature_2': self.correlation_matrix.columns[j],
                        'correlation': corr_value
                    })
        
        return pd.DataFrame(high_corr).sort_values('correlation', key=abs, ascending=False)


def build_dataset(latitude: float, longitude: float,
                 start_date: str, end_date: str,
                 api_key: Optional[str] = None,
                 save_path: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to build comprehensive dataset
    
    Args:
        latitude: Geographic latitude
        longitude: Geographic longitude
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        api_key: Solcast API key (optional)
        save_path: Path to save cleaned dataset (optional)
        
    Returns:
        Dictionary with dataset and analysis results
    """
    builder = DatasetBuilder()
    return builder.build_dataset(latitude, longitude, start_date, end_date, api_key, save_path)


if __name__ == "__main__":
    # Example usage
    try:
        print("🏗️ Testing Dataset Builder")
        print("=" * 40)
        
        # Build dataset for Beirut
        results = build_dataset(
            latitude=33.8938,
            longitude=35.5018,
            start_date='2023-01-01',
            end_date='2023-01-31',
            save_path='data/beirut_solar_dataset.csv'
        )
        
        print("\n📊 Dataset Summary:")
        print(f"Shape: {results['summary']['dataset_shape']}")
        print(f"Date range: {results['summary']['date_range']}")
        print(f"Columns: {len(results['summary']['columns'])}")
        
        # Show high correlations
        high_corr = results['builder'].get_high_correlations(threshold=0.7)
        if len(high_corr) > 0:
            print(f"\n🔗 High correlations (|r| ≥ 0.7):")
            print(high_corr)
        
        print("\n✅ Dataset builder test completed!")
        
    except Exception as e:
        print(f"❌ Error: {e}")
