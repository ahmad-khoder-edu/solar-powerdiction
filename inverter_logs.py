"""
Inverter Logs Module for Solar Power Prediction
Reads and processes inverter CSV data for integration with solar datasets
"""

import pandas as pd
import numpy as np
from typing import Optional, Dict, Any, Union
import os
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')


class InverterLogProcessor:
    """
    Processes inverter log data and integrates with solar datasets
    """
    
    def __init__(self):
        self.inverter_data = None
        self.daily_averages = None
        self.expected_columns = ['timestamp', 'voltage', 'current', 'power']
        
    def read_inverter_csv(self, file_path: Optional[str] = None) -> Optional[pd.DataFrame]:
        """
        Read inverter CSV file
        
        Args:
            file_path: Path to inverter CSV file
            
        Returns:
            DataFrame with inverter data or None if file not provided
        """
        if file_path is None:
            print("⚠️ No inverter log file provided, skipping inverter data processing")
            return None
        
        if not os.path.exists(file_path):
            print(f"❌ Inverter log file not found: {file_path}")
            return None
        
        try:
            print(f"📖 Reading inverter log file: {file_path}")
            
            # Read CSV file
            df = pd.read_csv(file_path)
            
            # Check if required columns are present
            missing_columns = [col for col in self.expected_columns if col not in df.columns]
            
            if missing_columns:
                print(f"⚠️ Missing expected columns: {missing_columns}")
                print(f"   Available columns: {list(df.columns)}")
                
                # Try to find alternative column names
                column_mapping = self._find_column_alternatives(df.columns)
                if column_mapping:
                    print(f"   Using alternative columns: {column_mapping}")
                    df = df.rename(columns=column_mapping)
                else:
                    print("   ❌ Cannot find suitable column alternatives")
                    return None
            else:
                print("✅ All expected columns found")
            
            # Convert timestamp to datetime
            timestamp_col = self._find_timestamp_column(df)
            if timestamp_col:
                df[timestamp_col] = pd.to_datetime(df[timestamp_col])
                df = df.set_index(timestamp_col)
                print(f"✅ Timestamp column converted: {timestamp_col}")
            
            # Sort by timestamp
            df = df.sort_index()
            
            # Remove duplicates
            initial_rows = len(df)
            df = df.drop_duplicates()
            final_rows = len(df)
            
            if initial_rows != final_rows:
                print(f"   Removed {initial_rows - final_rows} duplicate rows")
            
            self.inverter_data = df
            
            print(f"✅ Inverter data loaded: {df.shape}")
            print(f"   Date range: {df.index.min()} to {df.index.max()}")
            print(f"   Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            print(f"❌ Error reading inverter log file: {e}")
            return None
    
    def _find_column_alternatives(self, columns: list) -> Dict[str, str]:
        """
        Find alternative column names for expected columns
        
        Args:
            columns: List of available column names
            
        Returns:
            Dictionary mapping expected columns to found alternatives
        """
        # Common alternative names for expected columns
        alternatives = {
            'timestamp': ['time', 'datetime', 'date', 'timestamp_utc', 'time_utc', 'datetime_utc'],
            'voltage': ['v', 'volts', 'voltage_v', 'volts_v', 'bus_voltage'],
            'current': ['i', 'amps', 'current_a', 'amps_a', 'load_current'],
            'power': ['p', 'watts', 'power_w', 'watts_w', 'active_power', 'real_power']
        }
        
        column_mapping = {}
        columns_lower = [col.lower() for col in columns]
        
        for expected_col, alt_names in alternatives.items():
            for alt_name in alt_names:
                for col in columns:
                    if alt_name.lower() in col.lower():
                        column_mapping[expected_col] = col
                        break
                if expected_col in column_mapping:
                    break
        
        return column_mapping
    
    def _find_timestamp_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the timestamp column in the DataFrame
        
        Args:
            df: DataFrame to search
            
        Returns:
            Name of timestamp column or None
        """
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['time', 'date', 'timestamp']):
                return col
        return None
    
    def compute_daily_averages(self, data: Optional[pd.DataFrame] = None) -> pd.DataFrame:
        """
        Compute daily average power from inverter data
        
        Args:
            data: Inverter data DataFrame
            
        Returns:
            DataFrame with daily average power
        """
        if data is None:
            data = self.inverter_data
        
        if data is None or data.empty:
            print("⚠️ No inverter data available for daily averaging")
            return pd.DataFrame()
        
        print("📊 Computing daily average power...")
        
        # Ensure we have a datetime index
        if not isinstance(data.index, pd.DatetimeIndex):
            print("❌ Inverter data must have datetime index")
            return pd.DataFrame()
        
        # Find power column
        power_col = self._find_power_column(data)
        if power_col is None:
            print("❌ No power column found in inverter data")
            return pd.DataFrame()
        
        # Resample to daily averages
        daily_avg = data.resample('D').agg({
            power_col: 'mean',
            'voltage': 'mean' if 'voltage' in data.columns else 'first',
            'current': 'mean' if 'current' in data.columns else 'first'
        })
        
        # Rename columns for clarity
        column_rename = {power_col: 'average_power'}
        daily_avg = daily_avg.rename(columns=column_rename)
        
        # Add additional statistics
        if power_col in data.columns:
            daily_avg['max_power'] = data[power_col].resample('D').max()
            daily_avg['min_power'] = data[power_col].resample('D').min()
            daily_avg['std_power'] = data[power_col].resample('D').std()
            daily_avg['energy_kwh'] = data[power_col].resample('D').sum() / 1000  # Convert W to kWh
        
        self.daily_averages = daily_avg
        
        print(f"✅ Daily averages computed: {daily_avg.shape}")
        print(f"   Date range: {daily_avg.index.min().date()} to {daily_avg.index.max().date()}")
        print(f"   Average power range: {daily_avg['average_power'].min():.1f} - {daily_avg['average_power'].max():.1f} W")
        
        return daily_avg
    
    def _find_power_column(self, df: pd.DataFrame) -> Optional[str]:
        """
        Find the power column in the DataFrame
        
        Args:
            df: DataFrame to search
            
        Returns:
            Name of power column or None
        """
        for col in df.columns:
            if any(keyword in col.lower() for keyword in ['power', 'watt', 'p_', 'real_power', 'active_power']):
                return col
        return None
    
    def merge_with_dataset(self, dataset: pd.DataFrame, 
                        date_column: str = 'date') -> pd.DataFrame:
        """
        Merge inverter daily averages with existing dataset
        
        Args:
            dataset: Existing dataset to merge with
            date_column: Name of date column in dataset
            
        Returns:
            Merged dataset with inverter data
        """
        if self.daily_averages is None:
            print("⚠️ No daily averages available for merging")
            return dataset
        
        if dataset.empty:
            print("⚠️ Empty dataset provided for merging")
            return dataset
        
        print("🔗 Merging inverter data with dataset...")
        
        # Ensure both DataFrames have datetime index
        if not isinstance(dataset.index, pd.DatetimeIndex):
            if date_column in dataset.columns:
                dataset[date_column] = pd.to_datetime(dataset[date_column])
                dataset = dataset.set_index(date_column)
            else:
                print("❌ Dataset must have date column for merging")
                return dataset
        
        # Merge datasets
        merged_dataset = dataset.merge(
            self.daily_averages,
            left_index=True,
            right_index=True,
            how='left',
            suffixes=('', '_inverter')
        )
        
        # Fill missing values for inverter data
        inverter_cols = [col for col in merged_dataset.columns if col.endswith('_inverter')]
        for col in inverter_cols:
            merged_dataset[col] = merged_dataset[col].fillna(0)
        
        print(f"✅ Datasets merged: {merged_dataset.shape}")
        print(f"   Original dataset: {dataset.shape}")
        print(f"   Inverter data: {self.daily_averages.shape}")
        print(f"   Merged dataset: {merged_dataset.shape}")
        
        # Count merged rows
        merged_rows = len(merged_dataset[merged_dataset['average_power'].notna()])
        print(f"   Rows with inverter data: {merged_rows}/{len(merged_dataset)}")
        
        return merged_dataset
    
    def get_inverter_summary(self) -> Dict[str, Any]:
        """
        Get summary statistics of inverter data
        
        Returns:
            Dictionary with summary statistics
        """
        if self.daily_averages is None:
            return {}
        
        summary = {
            'total_days': len(self.daily_averages),
            'date_range': {
                'start': self.daily_averages.index.min().date(),
                'end': self.daily_averages.index.max().date()
            },
            'average_power_stats': {
                'mean': self.daily_averages['average_power'].mean(),
                'std': self.daily_averages['average_power'].std(),
                'min': self.daily_averages['average_power'].min(),
                'max': self.daily_averages['average_power'].max()
            },
            'energy_stats': {
                'total_energy_kwh': self.daily_averages['energy_kwh'].sum(),
                'daily_average_energy_kwh': self.daily_averages['energy_kwh'].mean(),
                'max_daily_energy_kwh': self.daily_averages['energy_kwh'].max()
            } if 'energy_kwh' in self.daily_averages.columns else {}
        }
        
        return summary
    
    def save_processed_data(self, output_path: str, format: str = 'csv'):
        """
        Save processed inverter data
        
        Args:
            output_path: Output file path
            format: Output format ('csv' or 'excel')
        """
        if self.daily_averages is None:
            print("⚠️ No data to save")
            return
        
        try:
            print(f"💾 Saving processed inverter data to {output_path}")
            
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            
            if format.lower() == 'csv':
                self.daily_averages.to_csv(output_path)
            elif format.lower() == 'excel':
                self.daily_averages.to_excel(output_path)
            else:
                print(f"❌ Unsupported format: {format}")
                return
            
            print(f"✅ Data saved successfully in {format} format")
            
        except Exception as e:
            print(f"❌ Error saving data: {e}")


def process_inverter_logs(inverter_file_path: Optional[str] = None,
                         dataset: Optional[pd.DataFrame] = None,
                         date_column: str = 'date') -> pd.DataFrame:
    """
    Standalone function to process inverter logs and merge with dataset
    
    Args:
        inverter_file_path: Path to inverter CSV file
        dataset: Existing dataset to merge with
        date_column: Name of date column in dataset
        
    Returns:
        Merged dataset or original dataset if no inverter data
    """
    processor = InverterLogProcessor()
    
    # Read inverter data
    inverter_data = processor.read_inverter_csv(inverter_file_path)
    
    if inverter_data is None:
        print("⚠️ No inverter data to process")
        return dataset if dataset is not None else pd.DataFrame()
    
    # Compute daily averages
    daily_averages = processor.compute_daily_averages(inverter_data)
    
    if daily_averages.empty:
        print("⚠️ No daily averages computed")
        return dataset if dataset is not None else pd.DataFrame()
    
    # Merge with dataset if provided
    if dataset is not None:
        merged_dataset = processor.merge_with_dataset(dataset, date_column)
        
        # Print summary
        summary = processor.get_inverter_summary()
        if summary:
            print(f"\n📊 Inverter Data Summary:")
            print(f"   Total days: {summary['total_days']}")
            print(f"   Date range: {summary['date_range']['start']} to {summary['date_range']['end']}")
            if 'average_power_stats' in summary:
                stats = summary['average_power_stats']
                print(f"   Average power: {stats['mean']:.1f} ± {stats['std']:.1f} W")
                print(f"   Power range: {stats['min']:.1f} - {stats['max']:.1f} W")
            if 'energy_stats' in summary:
                energy_stats = summary['energy_stats']
                print(f"   Total energy: {energy_stats['total_energy_kwh']:.1f} kWh")
                print(f"   Daily average: {energy_stats['daily_average_energy_kwh']:.1f} kWh")
        
        return merged_dataset
    else:
        print("⚠️ No dataset provided for merging")
        return daily_averages


if __name__ == "__main__":
    # Example usage
    print("🔌 Inverter Logs Processor Example")
    print("=" * 45)
    
    # Create sample inverter data for demonstration
    sample_data = {
        'timestamp': pd.date_range('2023-01-01', periods=72, freq='h'),
        'voltage': np.random.normal(230, 5, 72),
        'current': np.random.normal(10, 2, 72),
        'power': np.random.normal(2000, 500, 72)
    }
    sample_df = pd.DataFrame(sample_data)
    sample_file = 'data/sample_inverter_data.csv'
    sample_df.to_csv(sample_file, index=False)
    print(f"✅ Created sample inverter data: {sample_file}")
    
    # Create sample solar dataset
    sample_solar_data = pd.DataFrame({
        'date': pd.date_range('2023-01-01', periods=10, freq='D'),
        'GHI': np.random.uniform(1000, 7000, 10),
        'temperature': np.random.uniform(10, 25, 10)
    })
    
    # Process inverter logs
    print("\n📖 Processing inverter logs...")
    merged_data = process_inverter_logs(
        inverter_file_path=sample_file,
        dataset=sample_solar_data,
        date_column='date'
    )
    
    if not merged_data.empty:
        print(f"\n✅ Processing completed!")
        print(f"   Merged dataset shape: {merged_data.shape}")
        print(f"   New columns: {[col for col in merged_data.columns if 'inverter' in col]}")
        
        # Show sample of merged data
        print(f"\n📋 Sample of merged data:")
        sample_cols = ['date', 'GHI', 'average_power', 'energy_kwh']
        available_cols = [col for col in sample_cols if col in merged_data.columns]
        if available_cols:
            print(merged_data[available_cols].head().to_string())
    else:
        print("❌ No merged data created")
    
    # Clean up sample file
    if os.path.exists(sample_file):
        os.remove(sample_file)
        print(f"\n🧹 Cleaned up sample file: {sample_file}")
    
    print("\n✅ Inverter logs processor example completed!")
