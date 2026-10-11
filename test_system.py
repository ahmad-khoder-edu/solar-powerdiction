"""
Test script for Solar Power Prediction System
Quick verification of basic functionality
"""

import numpy as np
import pandas as pd
from data_collection import SolarDataCollector
from feature_engineering import SolarFeatureEngineer
from model_training import MultipleLinearRegression
from battery_model import BatteryModel, BatterySpecs

def test_data_collection():
    """Test data collection functionality"""
    print("Testing Data Collection...")
    
    collector = SolarDataCollector()
    
    # Test user input processing
    inputs = collector.process_user_inputs(
        latitude=33.8938,
        longitude=35.5018,
        battery_capacity=10,
        battery_type='lithium'
    )
    
    assert inputs['latitude'] == 33.8938
    assert inputs['battery_capacity'] == 10
    print("✓ Data collection test passed")

def test_feature_engineering():
    """Test feature engineering functionality"""
    print("Testing Feature Engineering...")
    
    # Create sample data
    dates = pd.date_range('2023-01-01', periods=100, freq='D')
    data = pd.DataFrame({
        'global_irradiance': np.random.uniform(0, 8, 100),
        'temperature': np.random.uniform(10, 35, 100),
        'wind_speed': np.random.uniform(0, 10, 100),
        'sunshine_duration': np.random.uniform(0, 12, 100)
    }, index=dates)
    
    engineer = SolarFeatureEngineer()
    features = engineer.engineer_features(data)
    
    assert len(features) > 0
    assert 'month_sin' in features.columns
    assert 'day_sin' in features.columns
    print("✓ Feature engineering test passed")

def test_model_training():
    """Test model training functionality"""
    print("Testing Model Training...")
    
    # Create sample data
    np.random.seed(42)
    X = np.random.randn(100, 5)
    y = X[:, 0] * 2 + X[:, 1] * -1 + np.random.randn(100) * 0.1
    
    # Test custom model
    model = MultipleLinearRegression(learning_rate=0.01, n_iterations=100)
    model.fit(X, y)
    
    weights, bias = model.get_coefficients()
    assert len(weights) == 5
    assert len(model.cost_history) == 100
    
    # Test prediction
    y_pred = model.predict(X[:5])
    assert len(y_pred) == 5
    
    print("✓ Model training test passed")

def test_battery_model():
    """Test battery model functionality"""
    print("Testing Battery Model...")
    
    specs = BatterySpecs(
        capacity=10,
        battery_type='lithium',
        efficiency=0,
        max_charge_rate=0,
        max_discharge_rate=0,
        depth_of_discharge=0,
        initial_soc=0.5
    )
    
    battery = BatteryModel(specs)
    
    # Test SOC calculation
    next_soc, battery_power, energy = battery.calculate_soc(
        solar_power=5, load=3, current_soc=0.5
    )
    
    assert 0 <= next_soc <= 1
    assert isinstance(battery_power, (int, float))
    assert energy >= 0
    
    # Test daily simulation
    solar_profile = np.maximum(0, 5 * np.sin(np.arange(24) * np.pi / 12))
    load_profile = np.ones(24) * 2
    
    results = battery.simulate_day(solar_profile, load_profile)
    
    assert 'soc' in results
    assert 'battery_power' in results
    assert len(results['soc']) == 24
    
    print("✓ Battery model test passed")

def run_all_tests():
    """Run all tests"""
    print("🧪 Running Solar Power Prediction System Tests")
    print("=" * 50)
    
    try:
        test_data_collection()
        test_feature_engineering()
        test_model_training()
        test_battery_model()
        
        print("\n🎉 All tests passed successfully!")
        return True
        
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        return False

if __name__ == "__main__":
    run_all_tests()
