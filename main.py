"""
Main Module for Solar Power Prediction System
Integrates all components for daily solar panel output prediction in Lebanon
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from typing import Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')

# Import custom modules
from data_collection import SolarDataCollector, collect_training_data, get_optimal_tilt_angle_standalone, fetch_solcast_historical_standalone
from feature_engineering import SolarFeatureEngineer, calculate_correlation_matrix, analyze_feature_correlation, calculate_feature_covariance
from model_training import train_solar_power_model
from battery_model import create_battery_from_user_input, BatterySpecs
from dataset_builder import build_dataset, DatasetBuilder


class SolarPowerPredictionSystem:
    """
    Main system for solar power prediction with battery integration
    """
    
    def __init__(self):
        self.data_collector = SolarDataCollector()
        self.feature_engineer = SolarFeatureEngineer()
        self.model = None
        self.trainer = None
        self.battery_model = None
        self.data = None
        self.features = None
        self.X = None
        self.y = None
        
    def fetch_solcast_data(self, latitude: float, longitude: float,
                           start_date: str, end_date: str, api_key: Optional[str] = None) -> pd.DataFrame:
        """
        Fetch historical daily data from Solcast API
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            api_key: Solcast API key (optional, will use environment if not provided)
            
        Returns:
            DataFrame with historical daily data (GHI, cloud opacity, temperature)
        """
        if api_key is None:
            api_key = os.getenv('SOLCAST_API_KEY')
        
        return fetch_solcast_historical_standalone(latitude, longitude, api_key, start_date, end_date)

    def build_comprehensive_dataset(self, latitude: float, longitude: float,
                                   start_date: str, end_date: str,
                                   api_key: Optional[str] = None,
                                   save_path: Optional[str] = None) -> Dict[str, Any]:
        """
        Build comprehensive dataset using the dataset builder
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            start_date: Start date in 'YYYY-MM-DD' format
            end_date: End date in 'YYYY-MM-DD' format
            api_key: Solcast API key (optional)
            save_path: Path to save dataset (optional)
            
        Returns:
            Dictionary with dataset and analysis results
        """
        return build_dataset(latitude, longitude, start_date, end_date, api_key, save_path)

    def get_optimal_tilt(self, latitude: Optional[float] = None,
                         longitude: Optional[float] = None) -> Optional[float]:
        """
        Get optimal tilt angle for solar panels
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            
        Returns:
            Optimal tilt angle in degrees, or None if location missing
        """
        return get_optimal_tilt_angle_standalone(latitude, longitude)

    def setup_system(self, latitude: Optional[float] = None,
                    longitude: Optional[float] = None,
                    battery_capacity: Optional[float] = None,
                    battery_type: Optional[str] = None,
                    load: Optional[float] = None,
                    inverter_logs: Optional[str] = None):
        """
        Setup the solar power prediction system
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            battery_capacity: Battery capacity in kWh
            battery_type: Type of battery
            load: Daily load profile in kWh
            inverter_logs: Path to inverter log files
        """
        print("🔧 Setting up Solar Power Prediction System...")
        
        # Process user inputs
        user_inputs = self.data_collector.process_user_inputs(
            latitude=latitude,
            longitude=longitude,
            battery_capacity=battery_capacity,
            battery_type=battery_type,
            load=load,
            inverter_logs=inverter_logs
        )
        
        # Handle both file path and DataFrame
        if isinstance(inverter_logs, str):
            # It's a file path - use process_inverter_logs function
            print("📖 Processing inverter logs from file...")
            inverter_data = process_inverter_logs(
                inverter_file_path=inverter_logs,
                dataset=self.X,
                date_column='date'
            )
        elif hasattr(inverter_logs, 'to_csv'):
            # It's a DataFrame with to_csv method
            print("📊 Processing inverter logs from DataFrame...")
            # Save to temporary file first
            temp_file = 'temp_inverter_data.csv'
            inverter_logs.to_csv(temp_file, index=False)
            inverter_data = process_inverter_logs(
                inverter_file_path=temp_file,
                dataset=self.X,
                date_column='date'
            )
            # Clean up temp file
            import os
            os.remove(temp_file)
        elif isinstance(inverter_logs, pd.DataFrame):
            # It's already a DataFrame - use it directly
            print("📊 Using provided inverter DataFrame...")
            inverter_data = inverter_logs
        else:
            print("⚠️ Invalid inverter_logs format, skipping inverter data")
            inverter_data = None
        
        print(f"✓ Location: {user_inputs['latitude']:.4f}, {user_inputs['longitude']:.4f}")
        
        # Get and display optimal tilt angle
        optimal_tilt = self.get_optimal_tilt(user_inputs['latitude'], user_inputs['longitude'])
        if optimal_tilt is not None:
            print(f"✓ Optimal tilt angle: {optimal_tilt:.1f}°")
        else:
            print("⚠️ Could not determine optimal tilt angle")
        
        if battery_capacity:
            print(f"✓ Battery: {battery_capacity} kWh {battery_type or 'lithium'}")
            self.battery_model = create_battery_from_user_input(battery_capacity, battery_type)
        
        if load:
            print(f"✓ Daily Load: {load} kWh")
        
        return user_inputs
    
    def collect_data(self, latitude: float, longitude: float,
                    start_year: int = 2020, end_year: int = 2023):
        """
        Collect training data
        
        Args:
            latitude: Geographic latitude
            longitude: Geographic longitude
            start_year: Start year for data
            end_year: End year for data
        """
        print("\n📊 Collecting solar data...")
        
        try:
            self.data = collect_training_data(latitude, longitude, start_year, end_year)
            print(f"✓ Collected {len(self.data)} days of data")
            return self.data
        except Exception as e:
            print(f"❌ Error collecting data: {e}")
            raise
    
    def engineer_features(self, **kwargs):
        """
        Engineer features from collected data
        
        Args:
            **kwargs: Additional parameters for feature engineering
        """
        print("\n⚙️ Engineering features...")
        
        try:
            self.features = self.feature_engineer.engineer_features(self.data, **kwargs)
            self.X, self.y = self.feature_engineer.prepare_features_and_target(self.features)
            print(f"✓ Created {self.X.shape[1]} features for {len(self.X)} samples")
            return self.features
        except Exception as e:
            print(f"❌ Error engineering features: {e}")
            raise
    
    def train_model(self, learning_rate: float = 0.01, max_iter: int = 1000, 
                   regularization: str = 'l2', alpha: float = 0.1):
        """
        Train prediction model
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            regularization: Type of regularization ('none', 'l1', 'l2')
            alpha: Regularization strength
        """
        print("\n🤖 Training prediction model...")
        
        try:
            # Check if data has been collected
            if self.data is None:
                print("⚠️ No data found. Collecting default data for Beirut, Lebanon...")
                self.collect_data(33.8938, 35.5018, 2019, 2020)
            
            # Check if features have been engineered
            if self.X is None or self.y is None:
                print("⚠️ No features found. Engineering features...")
                self.engineer_features()
            
            results = train_solar_power_model(
                self.X, self.y, 
                learning_rate=learning_rate,
                max_iter=max_iter,
                regularization=regularization,
                alpha=alpha
            )
            self.model = results['best_model']
            self.trainer = results['trainer']
            
            print("✓ Model training completed")
            return results
        except Exception as e:
            print(f"❌ Error training model: {e}")
            raise
    
    def analyze_data(self):
        """
        Perform comprehensive data analysis
        """
        print("\n📈 Analyzing data...")
        
        # Correlation analysis
        print("Performing correlation analysis...")
        correlations = analyze_feature_correlation(self.features)
        correlation_matrix = calculate_correlation_matrix(self.features)
        print("\nTop 10 features correlated with solar irradiance:")
        print(correlations.head(10))
        
        # Covariance analysis
        print("\nCalculating covariance matrix...")
        covariance_matrix = calculate_feature_covariance(self.X)
        print(f"Covariance matrix shape: {covariance_matrix.shape}")
        
        # Plot correlation heatmap
        plt.figure(figsize=(12, 8))
        
        # Select top correlated features for heatmap
        top_features = correlations.head(15).index.tolist()
        if 'global_irradiance' in top_features:
            top_features.remove('global_irradiance')
        
        correlation_subset = self.features[top_features + ['global_irradiance']].corr()
        
        sns.heatmap(correlation_subset, annot=True, cmap='coolwarm', center=0,
                   square=True, fmt='.2f')
        plt.title('Feature Correlation Heatmap')
        plt.tight_layout()
        plt.show()
        
        return correlations, covariance_matrix, correlation_matrix
    
    def predict_daily_output(self, date: str, **kwargs) -> Dict[str, Any]:
        """
        Predict solar panel output for a specific day
        
        Args:
            date: Date in 'YYYY-MM-DD' format
            **kwargs: Additional parameters for prediction
            
        Returns:
            Dictionary with prediction results
        """
        if self.model is None:
            raise ValueError("Model not trained. Call train_model() first.")
        
        print(f"\n🔮 Predicting solar output for {date}...")
        
        # Create features for the specific date
        date_features = self._create_date_features(date, **kwargs)
        
        # Get the prepared features for error checking
        X_date, _ = self.feature_engineer.prepare_features_and_target(date_features)
        
        # Make prediction
        try:
            prediction = self.model.predict(X_date)[0]
        except ValueError as e:
            if "feature names should match" in str(e):
                # Get expected features from the model
                if hasattr(self.model, 'feature_names_in_'):
                    expected_features = self.model.feature_names_in_
                    actual_features = X_date.columns.tolist()
                    print(f"❌ Feature mismatch detected:")
                    print(f"   Expected features: {list(expected_features)}")
                    print(f"   Actual features: {actual_features}")
                    print(f"   Missing features: {set(expected_features) - set(actual_features)}")
                raise ValueError("Feature mismatch between training and prediction. Please check feature engineering.") from e
            else:
                raise
        
        # Calculate confidence intervals (simplified)
        if hasattr(self.trainer, 'metrics'):
            rmse = self.trainer.metrics.get('Custom MLR', {}).get('RMSE', 0)
            confidence_interval = 1.96 * rmse  # 95% confidence interval
        else:
            confidence_interval = 0
        
        results = {
            'date': date,
            'predicted_irradiance': prediction,
            'confidence_interval': confidence_interval,
            'lower_bound': max(0, prediction - confidence_interval),
            'upper_bound': prediction + confidence_interval
        }
        
        print(f"✓ Predicted irradiance: {prediction:.2f} ± {confidence_interval:.2f}")
        
        return results
    
    def _create_date_features(self, date: str, **kwargs) -> np.ndarray:
        """
        Create features for a specific date
        
        Args:
            date: Date string
            **kwargs: Additional parameters
            
        Returns:
            Feature array for the date
        """
        # Convert date to datetime
        date_obj = pd.to_datetime(date)
        
        # Create a single-row DataFrame with the date
        date_df = pd.DataFrame({'date': [date_obj]})
        date_df = date_df.set_index('date')
        
        # Add realistic weather data based on training data averages
        # These are typical values for Beirut, Lebanon
        date_df['global_irradiance'] = 5848.0  # Mean from training data
        date_df['beam_irradiance'] = 3972.0   # Mean from training data
        date_df['diffuse_irradiance'] = 1805.9 # Mean from training data  
        date_df['reflected_irradiance'] = 70.1 # Mean from training data
        date_df['temperature'] = kwargs.get('temperature', 25.0)
        date_df['wind_speed'] = kwargs.get('wind_speed', 3.0)
        date_df['sunshine_duration'] = kwargs.get('sunshine_duration', 8.0)
        date_df['irradiance_intensity'] = date_df['global_irradiance'] / date_df['sunshine_duration']
        
        # Engineer features with simplified approach for single date
        try:
            date_features = self.feature_engineer.engineer_features(date_df, **kwargs)
            # Check if features were actually created
            if date_features.empty or len(date_features) == 0:
                raise ValueError("No features created - likely due to insufficient data for lag features")
        except Exception as e:
            # If full feature engineering fails due to insufficient data, use basic features
            print(f"⚠️ Full feature engineering failed ({e}), using basic features for prediction")
            date_features = self._create_basic_features(date_df, **kwargs)
        
        print(f"DEBUG: date_features type: {type(date_features)}")
        if hasattr(date_features, 'shape'):
            print(f"DEBUG: date_features shape: {date_features.shape}")
        
        # Prepare features manually to avoid method issues
        print(f"DEBUG: About to manually prepare features with type: {type(date_features)}")
        if hasattr(date_features, 'columns'):
            print(f"DEBUG: date_features columns: {list(date_features.columns)}")
        
        # Manually prepare features (same as prepare_features_and_target)
        exclude_cols = ['global_irradiance', 'year', 'month', 'day', 'day_of_year', 
                       'week_of_year', 'day_of_week', 'quarter']
        
        if hasattr(date_features, 'columns'):
            feature_cols = [col for col in date_features.columns if col not in exclude_cols]
            X_date = date_features[feature_cols]
            y_date = date_features['global_irradiance']
        else:
            # Handle numpy array case
            training_features = list(self.X.columns) if hasattr(self, 'X') and self.X is not None else []
            X_date = pd.DataFrame(date_features, columns=training_features)
            y_date = np.array([date_features[:, training_features.index('global_irradiance')]] if 'global_irradiance' in training_features else [0])
        
        print(f"DEBUG: Manually prepared X_date type: {type(X_date)}, shape: {X_date.shape if hasattr(X_date, 'shape') else 'N/A'}")
        
        # Ensure features are in the same order as training
        if hasattr(self, 'X') and self.X is not None:
            training_features = list(self.X.columns)
            if isinstance(X_date, pd.DataFrame):
                if list(X_date.columns) != training_features:
                    print("⚠️ Reordering features to match training order")
                    X_date = X_date[training_features]
            else:
                # X_date is numpy array, create DataFrame with proper column order
                X_date_df = pd.DataFrame(X_date, columns=training_features)
                X_date = X_date_df
        
        # Scale features
        X_date_scaled = self.trainer.scaler.transform(X_date)
        
        return X_date_scaled
    
    def _create_basic_features(self, df: pd.DataFrame, **kwargs) -> pd.DataFrame:
        """
        Create basic features for single date prediction when full engineering fails
        
        Args:
            df: DataFrame with single date
            **kwargs: Additional parameters
            
        Returns:
            DataFrame with basic features matching training feature names
        """
        df = df.copy()
        
        # Time features (matching training feature names)
        df['day_sin'] = np.sin(2 * np.pi * df.index.dayofyear / 365.25)
        df['day_cos'] = np.cos(2 * np.pi * df.index.dayofyear / 365.25)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # Seasonal features
        df['is_summer'] = df.index.month.isin([6, 7, 8]).astype(int)
        df['is_winter'] = df.index.month.isin([12, 1, 2]).astype(int)
        df['is_spring'] = df.index.month.isin([3, 4, 5]).astype(int)
        df['is_autumn'] = df.index.month.isin([9, 10, 11]).astype(int)
        
        # Temperature features
        df['temp_squared'] = df['temperature'] ** 2
        df['temp_cubed'] = df['temperature'] ** 3
        df['temp_cold'] = (df['temperature'] < 10).astype(int)
        df['temp_mild'] = ((df['temperature'] >= 10) & (df['temperature'] < 20)).astype(int)
        df['temp_warm'] = ((df['temperature'] >= 20) & (df['temperature'] < 30)).astype(int)
        df['temp_hot'] = (df['temperature'] >= 30).astype(int)
        
        # Wind features
        df['wind_speed_squared'] = df['wind_speed'] ** 2
        df['is_windy'] = (df['wind_speed'] > 5).astype(int)
        
        # Sunshine features
        df['sunshine_ratio'] = df['sunshine_duration'] / 24.0
        df['is_sunny'] = (df['sunshine_duration'] > 6).astype(int)
        
        # Lag features (use same values since we don't have history)
        for lag in [1, 2, 3, 7]:
            df[f'global_irradiance_lag_{lag}'] = df['global_irradiance']
        
        # Rolling features (use same values since we don't have history)
        for window in [3, 7, 14]:
            df[f'global_irradiance_rolling_mean_{window}'] = df['global_irradiance']
            df[f'global_irradiance_rolling_std_{window}'] = 100.0  # Dummy std
            df[f'global_irradiance_rolling_max_{window}'] = df['global_irradiance']
            df[f'global_irradiance_rolling_min_{window}'] = df['global_irradiance']
        
        # Interaction features
        df['temp_irradiance'] = df['temperature'] * df['global_irradiance'] / 1000
        df['wind_temp'] = df['wind_speed'] * df['temperature']
        df['season_irradiance'] = df['is_summer'] * df['global_irradiance']
        
        # Add target column (needed for feature preparation)
        df['global_irradiance'] = df['global_irradiance']  # Ensure target exists
        
        return df
    
    def simulate_with_battery(self, solar_prediction: float, load_profile: float,
                            hours: int = 24) -> Dict[str, Any]:
        """
        Simulate system operation with battery
        
        Args:
            solar_prediction: Predicted solar irradiance
            load_profile: Daily load in kWh
            hours: Number of hours to simulate
            
        Returns:
            Simulation results
        """
        if self.battery_model is None:
            print("⚠️ No battery model available. Skipping battery simulation.")
            return {}
        
        print(f"\n🔋 Simulating battery operation...")
        
        # Create hourly profiles
        hours_array = np.arange(hours)
        
        # Solar profile (simplified - peaks at midday)
        solar_profile = np.maximum(0, solar_prediction * 
                                 np.sin((hours_array - 6) * np.pi / 12))
        
        # Load profile (simplified - higher in morning/evening)
        hourly_load = load_profile / hours
        load_profile = hourly_load * (1 + 0.5 * np.sin(hours_array * np.pi / 12))
        
        # Simulate
        results = self.battery_model.simulate_day(solar_profile, load_profile)
        metrics = self.battery_model.calculate_metrics(results)
        
        print(f"✓ Battery simulation completed")
        print(f"  Self-consumption rate: {metrics['self_consumption_rate']:.2%}")
        print(f"  Self-sufficiency rate: {metrics['self_sufficiency_rate']:.2%}")
        
        # Plot results
        self.battery_model.plot_simulation_results(results)
        
        return {'results': results, 'metrics': metrics}
    
    def generate_report(self, prediction_date: str = None) -> Dict[str, Any]:
        """
        Generate comprehensive system report
        
        Args:
            prediction_date: Date for prediction report
            
        Returns:
            Dictionary with report data
        """
        print("\n📋 Generating system report...")
        
        report = {
            'system_info': {
                'data_points': len(self.data) if self.data is not None else 0,
                'features': self.X.shape[1] if self.X is not None else 0,
                'model_trained': self.model is not None,
                'battery_installed': self.battery_model is not None
            },
            'model_performance': {},
            'data_summary': {},
            'prediction': {}
        }
        
        # Model performance
        if self.trainer and hasattr(self.trainer, 'metrics'):
            report['model_performance'] = self.trainer.metrics
        
        # Data summary
        if self.data is not None:
            report['data_summary'] = {
                'date_range': f"{self.data.index.min()} to {self.data.index.max()}",
                'mean_irradiance': self.data['global_irradiance'].mean(),
                'max_irradiance': self.data['global_irradiance'].max(),
                'min_irradiance': self.data['global_irradiance'].min()
            }
        
        # Prediction
        if prediction_date and self.model is not None:
            prediction = self.predict_daily_output(prediction_date)
            report['prediction'] = prediction
        
        print("✓ Report generated")
        return report


def main():
    """
    Main function to run the solar power prediction system
    """
    print("🌞 Solar Power Prediction System for Lebanon")
    print("=" * 50)
    
    # Initialize system
    system = SolarPowerPredictionSystem()
    
    try:
        # Setup system with user inputs (all optional)
        user_inputs = system.setup_system(
            latitude=33.8938,  # Beirut, Lebanon
            longitude=35.5018,
            battery_capacity=10,  # 10 kWh battery
            battery_type='lithium',
            load=20  # 20 kWh daily load
        )
        
        # Collect data
        system.collect_data(
            latitude=user_inputs['latitude'],
            longitude=user_inputs['longitude'],
            start_year=2018,
            end_year=2020
        )
        
        # Engineer features
        system.engineer_features(
            battery_capacity=user_inputs.get('battery_capacity'),
            battery_type=user_inputs.get('battery_type'),
            load=user_inputs.get('load')
        )
        
        # Analyze data
        correlations, covariance_matrix, correlation_matrix = system.analyze_data()
        
        # Train model
        training_results = system.train_model()
        
        # Make prediction for a sample date within training range
        prediction = system.predict_daily_output('2020-06-15')
        
        # Simulate with battery if available
        if system.battery_model:
            battery_results = system.simulate_with_battery(
                prediction['predicted_irradiance'],
                user_inputs.get('load', 20)
            )
        
        # Generate report
        report = system.generate_report('2020-06-15')
        
        print("\n🎉 System execution completed successfully!")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        raise


if __name__ == "__main__":
    main()
