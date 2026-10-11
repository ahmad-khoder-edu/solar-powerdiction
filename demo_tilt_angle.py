"""
Demonstration of PVGIS optimal tilt angle function
"""

from data_collection import get_optimal_tilt_angle_standalone

def main():
    """Demonstrate the optimal tilt angle function"""
    
    print("🌞 PVGIS Optimal Tilt Angle Finder")
    print("=" * 40)
    
    # Example coordinates for major Lebanese cities
    cities = [
        ("Beirut", 33.8938, 35.5018),
        ("Tripoli", 34.4333, 35.8333),
        ("Sidon", 33.5583, 35.3747),
        ("Tyre", 33.2733, 35.1939),
        ("Jounieh", 33.9808, 35.6178),
        ("Zahle", 33.8464, 35.9022)
    ]
    
    print("📍 Optimal tilt angles for Lebanese cities:")
    print("-" * 50)
    
    for city, lat, lon in cities:
        tilt = get_optimal_tilt_angle_standalone(lat, lon)
        if tilt is not None:
            print(f"{city:12} | Lat: {lat:6.2f}° | Lon: {lon:7.2f}° | Tilt: {tilt:5.1f}°")
        else:
            print(f"{city:12} | Could not determine tilt angle")
    
    print("\n🔧 Testing edge cases:")
    
    # Test missing coordinates
    tilt_none = get_optimal_tilt_angle_standalone(None, None)
    print(f"Missing coordinates: {tilt_none}")
    
    # Test invalid coordinates
    tilt_invalid = get_optimal_tilt_angle_standalone(999, 999)
    print(f"Invalid coordinates: {tilt_invalid}")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()
