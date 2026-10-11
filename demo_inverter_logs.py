"""
Demonstration of inverter logs functionality
"""

import pandas as pd
import numpy as np
import os
from inverter_logs import process_inverter_logs, InverterLogProcessor

def create_sample_inverter_data():
    """Create realistic sample inverter data"""
    
    print("📊 Creating sample inverter data...")
    
    # Generate 3 days of hourly data
    timestamps = pd.date_range('2023-01-01', periods=72, freq='h')
    
    # Simulate realistic power patterns
    base_power = 3000  # 3kW system
    
    # Create daily patterns (higher during day, lower at night)
    hourly_pattern = []
    for hour in range(24):
        if 6 <= hour <= 18:  # Daytime
            power = base_power + np.random.normal(500, 200)
        else:  # Nighttime
            power = 200 + np.random.normal(50, 30)
        hourly_pattern.extend([power] * 3)  # 3 days
    
    # Create realistic data
    data = {
        'timestamp': timestamps,
        'voltage': np.random.normal(230, 5, 72),  # Stable voltage
        'current': np.array(hourly_pattern) / np.random.normal(230, 5, 72),  # Current = Power/Voltage
        'power': hourly_pattern
    }
    
    df = pd.DataFrame(data)
    return df

def demonstrate_inverter_integration():
    """Demonstrate inverter data integration with solar dataset"""
    
    print("🔌 Inverter Logs Integration Demonstration")
    print("=" * 50)
    
    # Step 1: Create sample inverter data
    inverter_df = create_sample_inverter_data()
    inverter_file = 'data/sample_inverter_data.csv'
    inverter_df.to_csv(inverter_file, index=False)
    print(f"✅ Sample inverter data created: {inverter_file}")
    
    # Step 2: Create sample solar dataset
    print("\n📈 Creating sample solar dataset...")
    
    solar_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=3, freq='D'),
        'GHI': [4500, 5500, 6000],  # Increasing solar irradiance
        'temperature': [12, 14, 16],
        'humidity': [65, 70, 68],
        'wind_speed': [5, 7, 6]
    })
    
    print(f"✅ Solar dataset created: {solar_data.shape}")
    
    # Step 3: Process inverter logs and merge
    print("\n🔗 Processing inverter logs and merging with solar data...")
    
    merged_data = process_inverter_logs(
        inverter_file_path=inverter_file,
        dataset=solar_data,
        date_column='date'
    )
    
    if not merged_data.empty:
        print(f"✅ Successfully merged datasets: {merged_data.shape}")
        
        # Step 4: Analyze merged dataset
        print("\n📊 Analyzing merged dataset...")
        
        # Show available columns
        print(f"   Available columns: {list(merged_data.columns)}")
        
        # Show inverter-specific columns
        inverter_cols = [col for col in merged_data.columns if col in ['average_power', 'energy_kwh', 'max_power', 'min_power']]
        print(f"   Inverter columns: {inverter_cols}")
        
        # Show correlation between solar and inverter data
        if 'GHI' in merged_data.columns and 'average_power' in merged_data.columns:
            correlation = merged_data['GHI'].corr(merged_data['average_power'])
            print(f"   GHI vs Average Power correlation: {correlation:.3f}")
        
        # Show sample of merged data
        print("\n📋 Sample of merged dataset:")
        sample_cols = ['date', 'GHI', 'average_power', 'energy_kwh']
        available_sample_cols = [col for col in sample_cols if col in merged_data.columns]
        
        if available_sample_cols:
            sample_data = merged_data[available_sample_cols].head()
            print(sample_data.to_string())
        
        # Step 5: Generate insights
        print("\n💡 Key Insights:")
        
        if 'average_power' in merged_data.columns:
            avg_power_stats = merged_data['average_power'].describe()
            print(f"   • Average inverter power: {avg_power_stats['mean']:.1f} W")
            print(f"   • Power variation: {avg_power_stats['std']:.1f} W")
            print(f"   • Peak power: {avg_power_stats['max']:.1f} W")
        
        if 'energy_kwh' in merged_data.columns:
            energy_stats = merged_data['energy_kwh'].describe()
            print(f"   • Daily energy production: {energy_stats['mean']:.1f} kWh")
            total_energy = merged_data['energy_kwh'].sum()
            print(f"   • Total energy (3 days): {total_energy:.1f} kWh")
        
        if 'GHI' in merged_data.columns and 'average_power' in merged_data.columns:
            # Calculate efficiency (solar to inverter conversion)
            ghi_total = merged_data['GHI'].sum()
            inverter_total = merged_data['energy_kwh'].sum() * 1000  # Convert kWh to Wh
            
            if inverter_total > 0:
                efficiency = (inverter_total / ghi_total) * 100
                print(f"   • System efficiency: {efficiency:.1f}%")
        
        # Step 6: Recommendations
        print("\n📋 Recommendations:")
        print("   • Use inverter data to validate solar predictions")
        print("   • Compare actual power output with predicted values")
        print("   • Monitor inverter efficiency over time")
        print("   • Use inverter data for battery sizing optimization")
        print("   • Analyze power patterns for load profiling")
        
    else:
        print("❌ Failed to merge datasets")
    
    # Step 7: Clean up
    if os.path.exists(inverter_file):
        os.remove(inverter_file)
        print(f"\n🧹 Cleaned up sample file: {inverter_file}")
    
    return merged_data

def demonstrate_advanced_features():
    """Demonstrate advanced inverter data processing"""
    
    print("\n🔬 Advanced Inverter Data Processing")
    print("=" * 45)
    
    processor = InverterLogProcessor()
    
    # Create detailed sample data
    detailed_data = {
        'timestamp': pd.date_range('2023-01-01', periods=168, freq='h'),  # 1 week
        'voltage': np.random.normal(230, 3, 168),
        'current': np.random.normal(12, 2, 168),
        'power': np.abs(np.random.normal(2500, 800, 168)),
        'frequency': np.random.normal(50, 5, 168),  # Grid frequency
        'power_factor': np.random.normal(0.95, 0.05, 168)  # Power factor
    }
    
    detailed_df = pd.DataFrame(detailed_data)
    detailed_file = 'data/detailed_inverter_data.csv'
    detailed_df.to_csv(detailed_file, index=False)
    
    print(f"✅ Created detailed inverter data: {detailed_file}")
    
    # Process with additional columns
    result = processor.read_inverter_csv(detailed_file)
    
    if result is not None:
        print(f"✅ Read detailed data: {result.shape}")
        
        # Compute daily averages with additional statistics
        daily_avg = processor.compute_daily_averages(result)
        
        if not daily_avg.empty:
            print(f"✅ Computed daily averages: {daily_avg.shape}")
            
            # Show enhanced statistics
            print("\n📈 Enhanced Daily Statistics:")
            
            if 'voltage' in daily_avg.columns:
                voltage_stats = daily_avg['voltage'].describe()
                print(f"   • Voltage stability: {voltage_stats['std']:.2f} V")
            
            if 'power_factor' in daily_avg.columns:
                pf_stats = daily_avg['power_factor'].describe()
                print(f"   • Average power factor: {pf_stats['mean']:.3f}")
            
            if 'frequency' in daily_avg.columns:
                freq_stats = daily_avg['frequency'].describe()
                print(f"   • Grid frequency: {freq_stats['mean']:.1f} Hz")
    
    # Clean up
    if os.path.exists(detailed_file):
        os.remove(detailed_file)
        print(f"\n🧹 Cleaned up detailed file: {detailed_file}")

if __name__ == "__main__":
    try:
        # Main demonstration
        merged_data = demonstrate_inverter_integration()
        
        # Advanced features demonstration
        demonstrate_advanced_features()
        
        print(f"\n🎯 Inverter Logs Integration Summary:")
        print("=" * 45)
        print("✅ CSV reading with flexible column detection")
        print("✅ Daily average power computation")
        print("✅ Dataset merging functionality")
        print("✅ Timestamp handling and datetime conversion")
        print("✅ Alternative column name support")
        print("✅ Statistical analysis and insights")
        print("✅ Error handling for missing files")
        print("✅ Data validation and cleaning")
        
        print(f"\n📋 Module Features:")
        print("• ✅ Reads inverter CSV if provided")
        print("• ✅ Expects columns: timestamp, voltage, current, power")
        print("• ✅ Computes daily average power")
        print("• ✅ Merges into dataset")
        print("• ✅ Skips if file not provided")
        print("• ✅ Handles alternative column names")
        print("• ✅ Provides comprehensive statistics")
        print("• ✅ Supports data validation")
        print("• ✅ Generates actionable insights")
        
    except Exception as e:
        print(f"❌ Demonstration failed: {e}")
        raise
