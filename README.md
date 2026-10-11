# Solar Power Prediction Project

## Overview
This project focuses on predicting daily solar panel output for Lebanon using machine learning techniques, specifically multiple linear regression with gradient descent. The system includes comprehensive data analysis, feature engineering, model training, and battery simulation capabilities.

## Project Structure
```
solar-powerdiction/
├── data/                    # Raw and processed data files
├── notebooks/               # Jupyter notebooks for analysis and modeling
├── src/                    # Source code for data processing and models
│   ├── data_fetcher.py     # Original data fetching module
│   └── pvgis_client.py     # Original PVGIS API client
├── data_collection.py      # Main data collection module
├── feature_engineering.py  # Feature engineering and analysis
├── model_training.py       # ML model training with gradient descent
├── battery_model.py        # Battery charging/discharging simulation
├── main.py                 # Main system integration
├── requirements.txt        # Python dependencies
├── .env                    # Environment variables
└── README.md              # Project documentation
```

## Features
- **Multiple Linear Regression**: Custom implementation with gradient descent
- **Feature Engineering**: Time-based, weather, lag, and interaction features
- **Statistical Analysis**: Correlation and covariance analysis
- **Battery Simulation**: Realistic battery charging/discharging modeling
- **Optional User Inputs**: Latitude, longitude, battery specs, load profiles
- **Visualization**: Comprehensive plotting for analysis and results
- **ML Best Practices**: Cross-validation, scaling, regularization

## Setup
1. Activate the virtual environment:
   ```bash
   source venv/bin/activate
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up environment variables:
   - Copy `.env` file and add your API keys if needed
   - SOLCAST_API_KEY is optional but recommended for weather forecasts

4. Run the main system:
   ```bash
   python main.py
   ```

## Dependencies
- pandas: Data manipulation and analysis
- numpy: Numerical computing
- matplotlib: Data visualization
- seaborn: Statistical visualization
- scikit-learn: Machine learning algorithms and utilities
- requests: HTTP requests for API data
- python-dotenv: Environment variable management
- jupyter: Interactive notebooks

## Usage

### Basic Usage
```python
from main import SolarPowerPredictionSystem

# Initialize system
system = SolarPowerPredictionSystem()

# Setup with optional parameters
system.setup_system(
    latitude=33.8938,  # Beirut, Lebanon
    longitude=35.5018,
    battery_capacity=10,  # kWh
    battery_type='lithium',
    load=20  # kWh daily
)

# Collect and process data
system.collect_data(33.8938, 35.5018)
system.engineer_features()

# Train model
results = system.train_model()

# Make predictions
prediction = system.predict_daily_output('2024-12-25')
```

### Individual Module Usage
```python
# Data collection
from data_collection import collect_training_data
data = collect_training_data()

# Feature engineering
from feature_engineering import SolarFeatureEngineer
engineer = SolarFeatureEngineer()
features = engineer.engineer_features(data)

# Model training
from model_training import train_solar_power_model
X, y = engineer.prepare_features_and_target(features)
results = train_solar_power_model(X, y)

# Battery simulation
from battery_model import BatteryModel, BatterySpecs
specs = BatterySpecs(capacity=10, battery_type='lithium')
battery = BatteryModel(specs)
```

## System Components

### Data Collection (`data_collection.py`)
- Fetches solar radiation data from PVGIS API
- Processes weather data from Solcast API
- Handles optional user inputs
- Default location: Beirut, Lebanon (33.8938°N, 35.5018°E)

### Feature Engineering (`feature_engineering.py`)
- Time-based features (seasonal, cyclical encoding)
- Weather-based features (temperature, wind, sunshine)
- Lag features for time series modeling
- Interaction features between variables
- Battery and load features if provided
- Correlation and covariance analysis

### Model Training (`model_training.py`)
- Custom Multiple Linear Regression with gradient descent
- Scikit-learn models for comparison
- Cross-validation and hyperparameter tuning
- Comprehensive evaluation metrics
- Feature importance analysis
- Training history visualization

### Battery Model (`battery_model.py`)
- Realistic battery charging/discharging simulation
- Support for multiple battery types (lithium, lead-acid, nickel, flow)
- Efficiency modeling and depth of discharge limits
- Performance metrics (self-consumption, self-sufficiency)
- Battery size optimization

### Main System (`main.py`)
- Integrates all components
- Handles user inputs and system configuration
- Provides end-to-end workflow
- Generates comprehensive reports
- Battery simulation integration

## ML Best Practices Implemented
- Data preprocessing and scaling
- Cross-validation for model evaluation
- Regularization (L1/L2) to prevent overfitting
- Feature selection based on correlation analysis
- Proper train/test splitting
- Comprehensive evaluation metrics
- Visualization for model interpretation

## Example Output
The system provides:
- Daily solar irradiance predictions with confidence intervals
- Feature importance rankings
- Model performance metrics (R², RMSE, MAE, MAPE)
- Battery performance metrics (self-consumption rate, self-sufficiency)
- Visualizations of predictions, correlations, and battery operation

## Configuration
All user inputs are optional:
- **latitude/longitude**: Geographic coordinates (default: Beirut)
- **battery_capacity**: Battery size in kWh
- **battery_type**: Battery chemistry type
- **load**: Daily energy consumption in kWh
- **inverter_logs**: Path to inverter performance data

## Contributing
1. Follow the existing code structure
2. Add comprehensive docstrings
3. Include error handling
4. Add unit tests for new features
5. Update documentation for changes

## Setup
Copy `.env.example` to `.env` and add your own Solcast API key. Virtual environments are not included: install from `requirements.txt`.
