"""
Battery Calculator Module for Solar Power Systems
Calculates battery runtime and state of charge based on capacity and load
"""

from typing import Optional, Tuple, Dict, Any
import pandas as pd
import numpy as np


class BatteryCalculator:
    """
    Battery calculator for solar power systems with different battery types
    """
    
    def __init__(self):
        # Battery specifications by type
        self.battery_specs = {
            'lead-acid': {
                'depth_of_discharge': 0.5,  # 50% DoD
                'efficiency': 0.85,  # 85% round-trip efficiency
                'self_discharge_rate': 0.05  # 5% per month
            },
            'lithium': {
                'depth_of_discharge': 0.8,  # 80% DoD
                'efficiency': 0.95,  # 95% round-trip efficiency
                'self_discharge_rate': 0.02  # 2% per month
            }
        }
    
    def calculate_battery_metrics(self, battery_capacity_kwh: Optional[float] = None,
                               daily_load_kwh: Optional[float] = None,
                               battery_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate battery runtime and end of day state of charge
        
        Args:
            battery_capacity_kwh: Battery capacity in kWh
            daily_load_kwh: Daily load in kWh
            battery_type: Type of battery ('lead-acid' or 'lithium')
            
        Returns:
            Dictionary with battery metrics or empty dict if inputs missing
        """
        # Check for missing inputs
        if battery_capacity_kwh is None or daily_load_kwh is None or battery_type is None:
            print("⚠️ Missing battery parameters, skipping calculation")
            return {}
        
        # Validate battery type
        if battery_type.lower() not in self.battery_specs:
            print(f"❌ Unsupported battery type: {battery_type}")
            print(f"   Supported types: {list(self.battery_specs.keys())}")
            return {}
        
        # Get battery specifications
        specs = self.battery_specs[battery_type.lower()]
        dod = specs['depth_of_discharge']
        efficiency = specs['efficiency']
        
        # Calculate usable battery capacity
        usable_capacity = battery_capacity_kwh * dod
        
        # Calculate runtime in hours (assuming constant load)
        if daily_load_kwh > 0:
            runtime_hours = (usable_capacity * efficiency) / (daily_load_kwh / 24)
        else:
            runtime_hours = float('inf')
        
        # Calculate end of day state of charge
        # Assuming battery starts full (100% SoC)
        daily_consumption = daily_load_kwh / efficiency  # Account for efficiency losses
        end_of_day_soc = max(0, (battery_capacity_kwh - daily_consumption) / battery_capacity_kwh)
        
        # Apply DoD limit (battery cannot go below this level)
        min_soc = 1 - dod
        end_of_day_soc = max(min_soc, end_of_day_soc)
        
        # Calculate days of autonomy
        days_of_autonomy = usable_capacity / daily_load_kwh if daily_load_kwh > 0 else float('inf')
        
        # Compile results
        results = {
            'battery_capacity_kwh': battery_capacity_kwh,
            'daily_load_kwh': daily_load_kwh,
            'battery_type': battery_type.lower(),
            'depth_of_discharge': dod,
            'efficiency': efficiency,
            'usable_capacity_kwh': usable_capacity,
            'runtime_hours': runtime_hours,
            'end_of_day_soc_percent': end_of_day_soc * 100,
            'days_of_autonomy': days_of_autonomy,
            'can_cover_daily_load': usable_capacity >= daily_load_kwh
        }
        
        return results
    
    def print_battery_analysis(self, battery_capacity_kwh: Optional[float] = None,
                             daily_load_kwh: Optional[float] = None,
                             battery_type: Optional[str] = None):
        """
        Print comprehensive battery analysis
        
        Args:
            battery_capacity_kwh: Battery capacity in kWh
            daily_load_kwh: Daily load in kWh
            battery_type: Type of battery ('lead-acid' or 'lithium')
        """
        results = self.calculate_battery_metrics(
            battery_capacity_kwh, daily_load_kwh, battery_type
        )
        
        if not results:
            return
        
        print(f"🔋 Battery Analysis Results")
        print("=" * 40)
        print(f"Battery Type: {results['battery_type'].title()}")
        print(f"Battery Capacity: {results['battery_capacity_kwh']:.1f} kWh")
        print(f"Daily Load: {results['daily_load_kwh']:.1f} kWh/day")
        print(f"Depth of Discharge: {results['depth_of_discharge']*100:.0f}%")
        print(f"Round-trip Efficiency: {results['efficiency']*100:.0f}%")
        print(f"Usable Capacity: {results['usable_capacity_kwh']:.1f} kWh")
        print(f"Runtime (constant load): {results['runtime_hours']:.1f} hours")
        print(f"End of Day SoC: {results['end_of_day_soc_percent']:.1f}%")
        print(f"Days of Autonomy: {results['days_of_autonomy']:.1f} days")
        
        if results['can_cover_daily_load']:
            print("✅ Battery can cover daily load")
        else:
            print("❌ Battery cannot cover daily load")
            deficit = results['daily_load_kwh'] - results['usable_capacity_kwh']
            print(f"   Daily deficit: {deficit:.1f} kWh")
    
    def compare_battery_types(self, battery_capacity_kwh: Optional[float] = None,
                            daily_load_kwh: Optional[float] = None) -> pd.DataFrame:
        """
        Compare performance between lead-acid and lithium batteries
        
        Args:
            battery_capacity_kwh: Battery capacity in kWh
            daily_load_kwh: Daily load in kWh
            
        Returns:
            DataFrame with comparison results
        """
        if battery_capacity_kwh is None or daily_load_kwh is None:
            print("⚠️ Missing parameters for comparison")
            return pd.DataFrame()
        
        comparison_results = []
        
        for battery_type in ['lead-acid', 'lithium']:
            results = self.calculate_battery_metrics(
                battery_capacity_kwh, daily_load_kwh, battery_type
            )
            if results:
                comparison_results.append(results)
        
        if comparison_results:
            df = pd.DataFrame(comparison_results)
            return df
        else:
            return pd.DataFrame()
    
    def size_battery_for_load(self, daily_load_kwh: Optional[float] = None,
                             days_of_autonomy: float = 1.0,
                             battery_type: str = 'lithium') -> Dict[str, Any]:
        """
        Calculate required battery size for given load and autonomy
        
        Args:
            daily_load_kwh: Daily load in kWh
            days_of_autonomy: Number of days battery should last
            battery_type: Type of battery ('lead-acid' or 'lithium')
            
        Returns:
            Dictionary with sizing results
        """
        if daily_load_kwh is None:
            print("⚠️ Missing daily load for sizing calculation")
            return {}
        
        if battery_type.lower() not in self.battery_specs:
            print(f"❌ Unsupported battery type: {battery_type}")
            return {}
        
        specs = self.battery_specs[battery_type.lower()]
        dod = specs['depth_of_discharge']
        efficiency = specs['efficiency']
        
        # Calculate required usable capacity
        required_usable = daily_load_kwh * days_of_autonomy
        
        # Calculate total capacity needed (accounting for DoD)
        required_capacity = required_usable / dod
        
        # Account for efficiency losses
        required_capacity = required_capacity / efficiency
        
        results = {
            'daily_load_kwh': daily_load_kwh,
            'days_of_autonomy': days_of_autonomy,
            'battery_type': battery_type.lower(),
            'required_capacity_kwh': required_capacity,
            'required_usable_capacity_kwh': required_usable,
            'depth_of_discharge': dod,
            'efficiency': efficiency
        }
        
        return results


def calculate_battery_runtime(battery_capacity_kwh: Optional[float] = None,
                           daily_load_kwh: Optional[float] = None,
                           battery_type: Optional[str] = None) -> Dict[str, Any]:
    """
    Standalone function to calculate battery runtime and SoC
    
    Args:
        battery_capacity_kwh: Battery capacity in kWh
        daily_load_kwh: Daily load in kWh
        battery_type: Type of battery ('lead-acid' or 'lithium')
        
    Returns:
        Dictionary with runtime and SoC information
    """
    calculator = BatteryCalculator()
    return calculator.calculate_battery_metrics(
        battery_capacity_kwh, daily_load_kwh, battery_type
    )


if __name__ == "__main__":
    # Example usage
    print("🔋 Battery Calculator Examples")
    print("=" * 40)
    
    calculator = BatteryCalculator()
    
    # Example 1: Lithium battery
    print("\n📊 Example 1: Lithium Battery")
    print("-" * 30)
    calculator.print_battery_analysis(
        battery_capacity_kwh=10,
        daily_load_kwh=5,
        battery_type='lithium'
    )
    
    # Example 2: Lead-acid battery
    print("\n📊 Example 2: Lead-Acid Battery")
    print("-" * 35)
    calculator.print_battery_analysis(
        battery_capacity_kwh=10,
        daily_load_kwh=5,
        battery_type='lead-acid'
    )
    
    # Example 3: Comparison
    print("\n📊 Example 3: Battery Type Comparison")
    print("-" * 42)
    comparison = calculator.compare_battery_types(
        battery_capacity_kwh=10,
        daily_load_kwh=5
    )
    
    if not comparison.empty:
        print(comparison[['battery_type', 'usable_capacity_kwh', 'runtime_hours', 
                        'end_of_day_soc_percent', 'days_of_autonomy']].to_string())
    
    # Example 4: Missing inputs
    print("\n📊 Example 4: Missing Inputs")
    print("-" * 30)
    result = calculator.calculate_battery_metrics(
        battery_capacity_kwh=None,
        daily_load_kwh=5,
        battery_type='lithium'
    )
    print(f"Result with missing inputs: {result}")
    
    # Example 5: Battery sizing
    print("\n📊 Example 5: Battery Sizing")
    print("-" * 30)
    sizing = calculator.size_battery_for_load(
        daily_load_kwh=5,
        days_of_autonomy=2,
        battery_type='lithium'
    )
    
    if sizing:
        print(f"Required battery capacity: {sizing['required_capacity_kwh']:.1f} kWh")
        print(f"For {sizing['days_of_autonomy']} days of autonomy")
    
    print("\n✅ Battery calculator examples completed!")
