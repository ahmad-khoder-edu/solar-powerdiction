"""
Test script for Solcast historical data integration
"""

from data_collection import fetch_solcast_historical_standalone, WeatherDataCollector
import os

def test_solcast_integration():
    """Test Solcast historical data fetching"""
    
    print("🌤️ Testing Solcast Historical Data Integration")
    print("=" * 50)
    
    # Get API key from environment
    api_key = os.getenv('SOLCAST_API_KEY')
    
    # Test locations
    locations = [
        ("Beirut, Lebanon", 33.8938, 35.5018),
        ("Tripoli, Lebanon", 34.4333, 35.8333),
        ("Sidon, Lebanon", 33.5583, 35.3747)
    ]
    
    for city, lat, lon in locations:
        print(f"\n📍 Location: {city}")
        print(f"   Coordinates: {lat:.4f}, {lon:.4f}")
        
        # Test standalone function
        print("   Testing standalone function...")
        df_standalone = fetch_solcast_historical_standalone(
            lat, lon, api_key, '2023-01-01', '2023-01-07'
        )
        
        if len(df_standalone) > 0:
            print(f"   ✓ Standalone: {len(df_standalone)} days of data")
            print(f"   ✓ Columns: {list(df_standalone.columns)}")
            print(f"   ✓ Date range: {df_standalone.index.min().date()} to {df_standalone.index.max().date()}")
            print(f"   ✓ Sample GHI values: {df_standalone['GHI'].head(3).tolist()}")
        else:
            print("   ❌ Standalone: No data returned")
        
        # Test class method
        print("   Testing class method...")
        collector = WeatherDataCollector()
        df_class = collector.fetch_solcast_historical_data(
            lat, lon, '2023-01-01', '2023-01-07'
        )
        
        if len(df_class) > 0:
            print(f"   ✓ Class method: {len(df_class)} days of data")
            print(f"   ✓ Columns: {list(df_class.columns)}")
        else:
            print("   ❌ Class method: No data returned")
        
        # Verify results match
        if len(df_standalone) > 0 and len(df_class) > 0:
            if df_standalone.equals(df_class):
                print("   ✅ Results match between standalone and class method")
            else:
                print("   ⚠️ Results differ between standalone and class method")
    
    # Test edge cases
    print("\n🔧 Testing edge cases...")
    
    # Test missing API key
    print("   Testing missing API key...")
    df_no_key = fetch_solcast_historical_standalone(
        33.8938, 35.5018, None, '2023-01-01', '2023-01-07'
    )
    print(f"   Missing API key result: {len(df_no_key)} rows")
    
    # Test invalid coordinates
    print("   Testing invalid coordinates...")
    df_invalid = fetch_solcast_historical_standalone(
        999, 999, api_key, '2023-01-01', '2023-01-07'
    )
    print(f"   Invalid coordinates result: {len(df_invalid)} rows")
    
    print("\n✅ Solcast integration testing completed!")

if __name__ == "__main__":
    test_solcast_integration()
