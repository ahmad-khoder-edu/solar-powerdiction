"""
Test script for PVGIS optimal tilt angle function
"""

from data_collection import get_optimal_tilt_angle_standalone, SolarDataCollector

def test_tilt_angle_function():
    """Test the optimal tilt angle function"""
    
    print("🧪 Testing PVGIS Optimal Tilt Angle Function")
    print("=" * 50)
    
    # Test locations
    test_locations = [
        {"name": "Beirut, Lebanon", "lat": 33.8938, "lon": 35.5018},
        {"name": "Tripoli, Lebanon", "lat": 34.4333, "lon": 35.8333},
        {"name": "Sidon, Lebanon", "lat": 33.5583, "lon": 35.3747},
        {"name": "Missing coordinates", "lat": None, "lon": None},
        {"name": "Invalid coordinates", "lat": 999, "lon": 999}
    ]
    
    for location in test_locations:
        print(f"\n📍 Location: {location['name']}")
        
        # Test standalone function
        tilt = get_optimal_tilt_angle_standalone(location['lat'], location['lon'])
        print(f"  Standalone function: {tilt}°")
        
        # Test class method
        collector = SolarDataCollector()
        tilt_class = collector.get_optimal_tilt_angle(location['lat'], location['lon'])
        print(f"  Class method: {tilt_class}°")
        
        # Verify results match
        if tilt == tilt_class:
            print("  ✅ Results match")
        else:
            print("  ❌ Results don't match")

if __name__ == "__main__":
    test_tilt_angle_function()
