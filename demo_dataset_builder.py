"""
Demonstration of Dataset Builder functionality
"""

from dataset_builder import build_dataset, DatasetBuilder
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def main():
    """Demonstrate dataset builder capabilities"""
    
    print("🏗️ Dataset Builder Demonstration")
    print("=" * 40)
    
    # Build a comprehensive dataset
    print("📊 Building comprehensive solar dataset for Beirut...")
    
    results = build_dataset(
        latitude=33.8938,
        longitude=35.5018,
        start_date='2023-01-01',
        end_date='2023-01-31',
        save_path='data/beirut_comprehensive_dataset.csv'
    )
    
    # Display results
    print(f"\n✅ Dataset built successfully!")
    print(f"   📅 Date range: {results['summary']['date_range']['start'].date()} to {results['summary']['date_range']['end'].date()}")
    print(f"   📏 Dataset shape: {results['summary']['dataset_shape']}")
    print(f"   📋 Total features: {len(results['summary']['columns'])}")
    print(f"   🔢 Numeric features: {len(results['summary']['numeric_columns'])}")
    print(f"   📐 Optimal tilt angle: {results['tilt_angle']:.1f}°")
    
    # Show feature categories
    print(f"\n📂 Feature Categories:")
    
    original = results['original_dataset']
    cleaned = results['cleaned_dataset']
    
    # Weather features
    weather_features = ['GHI', 'temperature', 'cloud_opacity']
    weather_present = [f for f in weather_features if f in cleaned.columns]
    print(f"   🌤️  Weather: {weather_present}")
    
    # Time features
    time_features = ['day_of_year', 'month', 'day', 'year', 'week_of_year', 'day_of_week', 'quarter']
    time_present = [f for f in time_features if f in cleaned.columns]
    print(f"   ⏰  Time: {time_present}")
    
    # Cyclical features
    cyclical_features = ['day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']
    cyclical_present = [f for f in cyclical_features if f in cleaned.columns]
    print(f"   🔄  Cyclical: {cyclical_present}")
    
    # Seasonal features
    seasonal_features = ['is_summer', 'is_winter', 'is_spring', 'is_autumn']
    seasonal_present = [f for f in seasonal_features if f in cleaned.columns]
    print(f"   🌺  Seasonal: {seasonal_present}")
    
    # System features
    system_features = ['tilt_angle']
    system_present = [f for f in system_features if f in cleaned.columns]
    print(f"   ⚙️  System: {system_present}")
    
    # Show data statistics
    print(f"\n📊 Data Statistics:")
    
    if 'GHI' in cleaned.columns:
        ghi_stats = cleaned['GHI'].describe()
        print(f"   ☀️  GHI (Global Horizontal Irradiance):")
        print(f"      Mean: {ghi_stats['mean']:.0f} Wh/m²/day")
        print(f"      Range: {ghi_stats['min']:.0f} - {ghi_stats['max']:.0f} Wh/m²/day")
        print(f"      Std Dev: {ghi_stats['std']:.0f} Wh/m²/day")
    
    if 'temperature' in cleaned.columns:
        temp_stats = cleaned['temperature'].describe()
        print(f"   🌡️  Temperature:")
        print(f"      Mean: {temp_stats['mean']:.1f}°C")
        print(f"      Range: {temp_stats['min']:.1f} - {temp_stats['max']:.1f}°C")
        print(f"      Std Dev: {temp_stats['std']:.1f}°C")
    
    # Show correlation insights
    print(f"\n🔗 Correlation Insights:")
    
    corr_matrix = results['correlation_matrix']
    
    # Find correlations with GHI
    if 'GHI' in corr_matrix.columns:
        ghi_corr = corr_matrix['GHI'].sort_values(key=abs, ascending=False)
        top_corr = ghi_corr.drop('GHI').head(3)
        print(f"   Top 3 features correlated with GHI:")
        for feature, corr in top_corr.items():
            print(f"      {feature}: {corr:.3f}")
    
    # Show high correlations
    builder = DatasetBuilder()
    builder.cleaned_dataset = cleaned
    builder.correlation_matrix = corr_matrix
    
    high_corr = builder.get_high_correlations(threshold=0.8)
    if len(high_corr) > 0:
        print(f"   Highly correlated pairs (|r| ≥ 0.8):")
        for _, row in high_corr.head(3).iterrows():
            print(f"      {row['feature_1']} ↔ {row['feature_2']}: {row['correlation']:.3f}")
    
    # Show missing values handling
    print(f"\n🧹 Data Cleaning:")
    missing_before = original.isnull().sum().sum()
    missing_after = cleaned.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}")
    print(f"   Missing values after: {missing_after}")
    print(f"   Rows removed: {len(original) - len(cleaned)}")
    
    # Show file saving info
    print(f"\n💾 File Saving:")
    import os
    csv_file = 'data/beirut_comprehensive_dataset.csv'
    pkl_file = 'data/beirut_comprehensive_dataset.pkl'
    
    if os.path.exists(csv_file):
        file_size = os.path.getsize(csv_file)
        print(f"   CSV file: {csv_file} ({file_size:,} bytes)")
    
    if os.path.exists(pkl_file):
        file_size = os.path.getsize(pkl_file)
        print(f"   Pickle file: {pkl_file} ({file_size:,} bytes)")
    
    # Show sample data
    print(f"\n📋 Sample Data (first 3 rows):")
    sample_cols = ['GHI', 'temperature', 'tilt_angle', 'day_of_year', 'month']
    available_cols = [col for col in sample_cols if col in cleaned.columns]
    print(cleaned[available_cols].head(3).to_string())
    
    print(f"\n🎯 Key Features of Dataset Builder:")
    print("• ✅ Merges Solcast weather data with optimal tilt angle")
    print("• ✅ Creates comprehensive time-based features")
    print("• ✅ Handles missing values intelligently")
    print("• ✅ Computes covariance and correlation matrices")
    print("• ✅ Saves datasets in multiple formats")
    print("• ✅ Preserves original dataset copy")
    print("• ✅ Provides detailed summary statistics")
    print("• ✅ Identifies highly correlated features")
    
    print(f"\n🎉 Dataset Builder demonstration completed!")

if __name__ == "__main__":
    main()
