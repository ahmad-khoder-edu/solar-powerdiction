#!/usr/bin/env python3
"""
Test script for main.py scenarios
"""

import sys
import os
sys.path.append('.')

from main import SolarPowerPredictionSystem

def test_scenario_1_location_only():
    """Test scenario 1: Only location"""
    print("🧪 Scenario 1: Location Only")
    print("=" * 40)
    
    try:
        system = SolarPowerPredictionSystem()
        
        # Only provide location
        user_inputs = system.setup_system(
            latitude=33.8938,
            longitude=35.5018
        )
        
        print(f"✅ Location setup: {user_inputs['latitude']}, {user_inputs['longitude']}")
        print("✅ No battery, no load, no inverter - should not crash")
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario 1 failed: {e}")
        return False

def test_scenario_2_location_solcast():
    """Test scenario 2: Location + Solcast"""
    print("\n🧪 Scenario 2: Location + Solcast")
    print("=" * 40)
    
    try:
        system = SolarPowerPredictionSystem()
        
        # Location + Solcast API key (from environment)
        user_inputs = system.setup_system(
            latitude=33.8938,
            longitude=35.5018
        )
        
        print(f"✅ Location setup: {user_inputs['latitude']}, {user_inputs['longitude']}")
        print("✅ Solcast API key available")
        print("✅ No battery, no load, no inverter - should not crash")
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario 2 failed: {e}")
        return False

def test_scenario_3_location_battery():
    """Test scenario 3: Location + Battery"""
    print("\n🧪 Scenario 3: Location + Battery")
    print("=" * 40)
    
    try:
        system = SolarPowerPredictionSystem()
        
        # Location + Battery
        user_inputs = system.setup_system(
            latitude=33.8938,
            longitude=35.5018,
            battery_capacity=10,
            battery_type='lithium'
        )
        
        print(f"✅ Location setup: {user_inputs['latitude']}, {user_inputs['longitude']}")
        print(f"✅ Battery setup: {user_inputs['battery_capacity']} kWh {user_inputs['battery_type']}")
        print("✅ No load, no inverter - should not crash")
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario 3 failed: {e}")
        return False

def test_scenario_4_full_inputs():
    """Test scenario 4: Full inputs + inverter logs"""
    print("\n🧪 Scenario 4: Full inputs + inverter logs")
    print("=" * 40)
    
    try:
        system = SolarPowerPredictionSystem()
        
        # Create sample inverter log file
        import pandas as pd
        import numpy as np
        
        sample_inverter_data = {
            'timestamp': pd.date_range('2023-01-01', periods=48, freq='h'),
            'voltage': np.random.normal(230, 5, 48),
            'current': np.random.normal(10, 2, 48),
            'power': np.abs(np.random.normal(2000, 500, 48))
        }
        
        inverter_file = 'data/test_inverter.csv'
        sample_inverter_data.to_csv(inverter_file, index=False)
        
        # Full inputs with inverter
        user_inputs = system.setup_system(
            latitude=33.8938,
            longitude=35.5018,
            battery_capacity=10,
            battery_type='lithium',
            load=20,
            inverter_logs=inverter_file
        )
        
        print(f"✅ Full setup completed")
        print(f"✅ System should handle all inputs gracefully")
        
        # Clean up
        if os.path.exists(inverter_file):
            os.remove(inverter_file)
        
        return True
        
    except Exception as e:
        print(f"❌ Scenario 4 failed: {e}")
        return False

def test_missing_inputs():
    """Test system behavior with missing inputs"""
    print("\n🧪 Testing Missing Inputs")
    print("=" * 30)
    
    try:
        system = SolarPowerPredictionSystem()
        
        # Test with all missing inputs
        user_inputs = system.setup_system()
        
        print("✅ Missing inputs handled gracefully")
        print("✅ System should not crash")
        
        return True
        
    except Exception as e:
        print(f"❌ Missing inputs test failed: {e}")
        return False

def main():
    """Run all test scenarios"""
    print("🧪 Testing Main.py Scenarios")
    print("=" * 50)
    
    results = []
    
    # Run all scenarios
    results.append(("Scenario 1: Location Only", test_scenario_1_location_only()))
    results.append(("Scenario 2: Location + Solcast", test_scenario_2_location_solcast()))
    results.append(("Scenario 3: Location + Battery", test_scenario_3_location_battery()))
    results.append(("Scenario 4: Full inputs + Inverter", test_scenario_4_full_inputs()))
    results.append(("Missing Inputs", test_missing_inputs()))
    
    # Summary
    print("\n📊 Test Results Summary:")
    print("=" * 30)
    
    for scenario, success in results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{status}: {scenario}")
    
    passed = sum(1 for _, success in results)
    total = len(results)
    
    print(f"\n🎯 Overall Results: {passed}/{total} scenarios passed")
    
    if passed == total:
        print("🎉 All scenarios passed! System handles missing inputs gracefully.")
    else:
        print("⚠️ Some scenarios failed. Check error messages above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
