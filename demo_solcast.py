"""
Demonstration of Solcast historical data integration
"""

from data_collection import fetch_solcast_historical_standalone
from main import SolarPowerPredictionSystem
import pandas as pd

def main():
    """Demonstrate Solcast historical data functionality"""
    
    print("🌤️ Solcast Historical Data Integration Demo")
    print("=" * 45)
    
    # Initialize system
    system = SolarPowerPredictionSystem()
    
    # Example coordinates for Lebanese cities
    cities = [
        ("Beirut", 33.8938, 35.5018),
        ("Tripoli", 34.4333, 35.8333),
        ("Sidon", 33.5583, 35.3747)
    ]
    
    print("📍 Fetching historical solar data for Lebanese cities:")
    print("-" * 60)
    
    for city, lat, lon in cities:
        print(f"\n🏙️  {city}")
        print(f"   Coordinates: {lat:.4f}°N, {lon:.4f}°E")
        
        # Fetch 7 days of historical data
        df = system.fetch_solcast_data(lat, lon, '2023-01-01', '2023-01-07')
        
        if len(df) > 0:
            print(f"   ✅ Successfully fetched {len(df)} days of data")
            print(f"   📊 Available columns: {list(df.columns)}")
            
            # Calculate statistics
            if 'GHI' in df.columns:
                avg_ghi = df['GHI'].mean()
                max_ghi = df['GHI'].max()
                min_ghi = df['GHI'].min()
                print(f"   ☀️  GHI Statistics:")
                print(f"      Average: {avg_ghi:.0f} Wh/m²/day")
                print(f"      Maximum: {max_ghi:.0f} Wh/m²/day")
                print(f"      Minimum: {min_ghi:.0f} Wh/m²/day")
            
            if 'temperature' in df.columns:
                avg_temp = df['temperature'].mean()
                max_temp = df['temperature'].max()
                min_temp = df['temperature'].min()
                print(f"   🌡️  Temperature Statistics:")
                print(f"      Average: {avg_temp:.1f}°C")
                print(f"      Maximum: {max_temp:.1f}°C")
                print(f"      Minimum: {min_temp:.1f}°C")
            
            if 'cloud_opacity' in df.columns:
                avg_cloud = df['cloud_opacity'].mean()
                print(f"   ☁️  Average Cloud Opacity: {avg_cloud:.3f}")
            
        else:
            print("   ❌ No data returned")
    
    print("\n🔧 Testing standalone function...")
    
    # Test standalone function directly
    df_standalone = fetch_solcast_historical_standalone(
        33.8938, 35.5018, 
        None,  # Will use environment variable
        '2023-01-01', 
        '2023-01-03'
    )
    
    if len(df_standalone) > 0:
        print(f"✅ Standalone function: {len(df_standalone)} days of data")
        print(f"📋 Sample data:")
        print(df_standalone.to_string())
    else:
        print("❌ Standalone function: No data returned")
    
    print("\n🎯 Key Features:")
    print("• Fetches historical daily GHI (Global Horizontal Irradiance)")
    print("• Provides temperature data")
    print("• Includes cloud opacity when available")
    print("• Automatic daily aggregation from sub-daily data")
    print("• Graceful handling of missing API keys")
    print("• Error handling for invalid coordinates")
    print("• Integration with main solar prediction system")
    
    print("\n✅ Demo completed!")

if __name__ == "__main__":
    main()
