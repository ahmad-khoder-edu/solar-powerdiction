"""
Battery Model Module for Solar Power Prediction
Models battery charging/discharging behavior and energy storage
"""

import numpy as np
import pandas as pd
from typing import Dict, Any, Optional, Tuple
import matplotlib.pyplot as plt
from dataclasses import dataclass


@dataclass
class BatterySpecs:
    """Battery specifications"""
    capacity: float  # kWh
    battery_type: str  # 'lithium', 'lead_acid', 'nickel', 'flow'
    efficiency: float  # Round-trip efficiency (0-1)
    max_charge_rate: float  # kW
    max_discharge_rate: float  # kW
    depth_of_discharge: float  # Maximum DOD (0-1)
    initial_soc: float  # Initial state of charge (0-1)


class BatteryModel:
    """
    Models battery charging and discharging behavior for solar power systems
    """
    
    def __init__(self, specs: BatterySpecs):
        """
        Initialize battery model
        
        Args:
            specs: Battery specifications
        """
        self.specs = specs
        self.soc_history = []
        self.power_history = []
        self.energy_throughput = 0
        
        # Battery type specific parameters
        self._setup_battery_parameters()
    
    def _setup_battery_parameters(self):
        """Setup battery-specific parameters"""
        # Default efficiencies by battery type
        default_efficiencies = {
            'lithium': 0.95,
            'lead_acid': 0.80,
            'nickel': 0.85,
            'flow': 0.90
        }
        
        # Update efficiency if not specified
        if self.specs.efficiency == 0:
            self.specs.efficiency = default_efficiencies.get(
                self.specs.battery_type.lower(), 0.85
            )
        
        # Default charge/discharge rates if not specified
        if self.specs.max_charge_rate == 0:
            self.specs.max_charge_rate = self.specs.capacity * 0.5  # C/2 rate
        
        if self.specs.max_discharge_rate == 0:
            self.specs.max_discharge_rate = self.specs.capacity * 0.5  # C/2 rate
        
        # Default depth of discharge
        if self.specs.depth_of_discharge == 0:
            dod_map = {
                'lithium': 0.9,
                'lead_acid': 0.5,
                'nickel': 0.8,
                'flow': 1.0
            }
            self.specs.depth_of_discharge = dod_map.get(
                self.specs.battery_type.lower(), 0.8
            )
    
    def calculate_soc(self, solar_power: float, load: float, 
                     current_soc: float, timestep: float = 1.0) -> Tuple[float, float, float]:
        """
        Calculate battery state of charge for next timestep
        
        Args:
            solar_power: Solar power generation (kW)
            load: Power demand (kW)
            current_soc: Current state of charge (0-1)
            timestep: Time step in hours
            
        Returns:
            Tuple of (next_soc, battery_power, energy_stored/released)
        """
        # Power balance
        net_power = solar_power - load
        
        # Initialize battery power
        battery_power = 0
        
        if net_power > 0:
            # Excess power - charge battery
            max_charge_power = min(
                self.specs.max_charge_rate,
                (self.specs.capacity - current_soc * self.specs.capacity) / timestep / self.specs.efficiency
            )
            
            battery_power = min(net_power, max_charge_power)
            energy_stored = battery_power * timestep * self.specs.efficiency
            
            # Update SOC
            next_soc = current_soc + energy_stored / self.specs.capacity
            
        else:
            # Power deficit - discharge battery
            max_discharge_power = min(
                self.specs.max_discharge_rate,
                (current_soc * self.specs.capacity - self.specs.capacity * (1 - self.specs.depth_of_discharge)) / timestep
            )
            
            battery_power = max(net_power, -max_discharge_power)  # Negative for discharge
            energy_released = abs(battery_power) * timestep / self.specs.efficiency
            
            # Update SOC
            next_soc = current_soc - energy_released / self.specs.capacity
        
        # Ensure SOC stays within bounds
        next_soc = np.clip(next_soc, 1 - self.specs.depth_of_discharge, 1.0)
        
        # Track energy throughput
        self.energy_throughput += abs(battery_power) * timestep
        
        return next_soc, battery_power, abs(battery_power) * timestep
    
    def simulate_day(self, solar_profile: np.ndarray, load_profile: np.ndarray,
                    initial_soc: Optional[float] = None) -> Dict[str, np.ndarray]:
        """
        Simulate battery operation for a day
        
        Args:
            solar_profile: Hourly solar power generation (kW)
            load_profile: Hourly power demand (kW)
            initial_soc: Initial state of charge
            
        Returns:
            Dictionary with simulation results
        """
        if initial_soc is None:
            initial_soc = self.specs.initial_soc
        
        n_hours = len(solar_profile)
        soc = np.zeros(n_hours + 1)
        battery_power = np.zeros(n_hours)
        grid_power = np.zeros(n_hours)
        
        soc[0] = initial_soc
        
        for i in range(n_hours):
            soc[i+1], battery_power[i], _ = self.calculate_soc(
                solar_profile[i], load_profile[i], soc[i], timestep=1.0
            )
            
            # Grid power (positive = import, negative = export)
            grid_power[i] = load_profile[i] - solar_profile[i] - battery_power[i]
        
        # Store history
        self.soc_history.extend(soc[1:])
        self.power_history.extend(battery_power)
        
        return {
            'soc': soc[1:],
            'battery_power': battery_power,
            'grid_power': grid_power,
            'solar_power': solar_profile,
            'load_power': load_profile
        }
    
    def calculate_metrics(self, results: Dict[str, np.ndarray]) -> Dict[str, float]:
        """
        Calculate performance metrics
        
        Args:
            results: Simulation results
            
        Returns:
            Dictionary with performance metrics
        """
        solar_power = results['solar_power']
        load_power = results['load_power']
        battery_power = results['battery_power']
        grid_power = results['grid_power']
        
        # Energy metrics
        total_solar = np.sum(solar_power)
        total_load = np.sum(load_power)
        total_battery_charge = np.sum(np.maximum(battery_power, 0))
        total_battery_discharge = np.sum(np.maximum(-battery_power, 0))
        total_grid_import = np.sum(np.maximum(grid_power, 0))
        total_grid_export = np.sum(np.maximum(-grid_power, 0))
        
        # Self-consumption rate
        direct_consumption = np.sum(np.minimum(solar_power, load_power))
        total_solar_consumption = direct_consumption + total_battery_discharge
        self_consumption_rate = total_solar_consumption / total_solar if total_solar > 0 else 0
        
        # Self-sufficiency rate
        self_sufficiency_rate = total_solar_consumption / total_load if total_load > 0 else 0
        
        # Battery utilization
        avg_soc = np.mean(results['soc'])
        soc_range = np.max(results['soc']) - np.min(results['soc'])
        
        # Round-trip efficiency
        round_trip_efficiency = total_battery_discharge / total_battery_charge if total_battery_charge > 0 else 0
        
        metrics = {
            'total_solar_energy': total_solar,
            'total_load_energy': total_load,
            'battery_charge_energy': total_battery_charge,
            'battery_discharge_energy': total_battery_discharge,
            'grid_import_energy': total_grid_import,
            'grid_export_energy': total_grid_export,
            'self_consumption_rate': self_consumption_rate,
            'self_sufficiency_rate': self_sufficiency_rate,
            'average_soc': avg_soc,
            'soc_range': soc_range,
            'round_trip_efficiency': round_trip_efficiency,
            'energy_throughput': self.energy_throughput
        }
        
        return metrics
    
    def plot_simulation_results(self, results: Dict[str, np.ndarray], 
                              save_path: Optional[str] = None):
        """
        Plot simulation results
        
        Args:
            results: Simulation results
            save_path: Path to save the plot
        """
        hours = np.arange(len(results['solar_power']))
        
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Power flows
        ax1 = axes[0]
        ax1.plot(hours, results['solar_power'], label='Solar Power', color='orange', linewidth=2)
        ax1.plot(hours, results['load_power'], label='Load', color='red', linewidth=2)
        ax1.plot(hours, results['battery_power'], label='Battery Power', color='blue', linewidth=2)
        ax1.plot(hours, results['grid_power'], label='Grid Power', color='green', linewidth=2)
        ax1.set_xlabel('Hour of Day')
        ax1.set_ylabel('Power (kW)')
        ax1.set_title('Power Flows Throughout the Day')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        ax1.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # State of charge
        ax2 = axes[1]
        ax2.plot(hours, results['soc'], color='purple', linewidth=2)
        ax2.set_xlabel('Hour of Day')
        ax2.set_ylabel('State of Charge')
        ax2.set_title('Battery State of Charge')
        ax2.set_ylim([0, 1])
        ax2.grid(True, alpha=0.3)
        
        # Add SOC zones
        ax2.axhspan(0, 1 - self.specs.depth_of_discharge, alpha=0.2, color='red', label='Min SOC')
        ax2.axhspan(0.8, 1.0, alpha=0.2, color='green', label='High SOC')
        ax2.legend()
        
        # Energy balance
        ax3 = axes[2]
        energy_balance = np.cumsum(results['solar_power'] - results['load_power'])
        ax3.plot(hours, energy_balance, color='brown', linewidth=2)
        ax3.set_xlabel('Hour of Day')
        ax3.set_ylabel('Cumulative Energy (kWh)')
        ax3.set_title('Cumulative Energy Balance')
        ax3.grid(True, alpha=0.3)
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def optimize_battery_size(self, solar_profiles: np.ndarray, 
                            load_profiles: np.ndarray,
                            capacity_range: Tuple[float, float] = (1, 50),
                            step: float = 1) -> Dict[str, Any]:
        """
        Optimize battery size for given solar and load profiles
        
        Args:
            solar_profiles: Array of daily solar profiles
            load_profiles: Array of daily load profiles
            capacity_range: Range of battery capacities to test
            step: Step size for capacity search
            
        Returns:
            Dictionary with optimization results
        """
        capacities = np.arange(capacity_range[0], capacity_range[1] + step, step)
        results = []
        
        for capacity in capacities:
            # Create temporary battery specs
            temp_specs = BatterySpecs(
                capacity=capacity,
                battery_type=self.specs.battery_type,
                efficiency=0,  # Will be set automatically
                max_charge_rate=0,  # Will be set automatically
                max_discharge_rate=0,  # Will be set automatically
                depth_of_discharge=0,  # Will be set automatically
                initial_soc=0.5
            )
            
            temp_battery = BatteryModel(temp_specs)
            
            # Simulate for all days
            total_self_consumption = 0
            total_self_sufficiency = 0
            
            for day in range(len(solar_profiles)):
                day_results = temp_battery.simulate_day(
                    solar_profiles[day], load_profiles[day]
                )
                metrics = temp_battery.calculate_metrics(day_results)
                total_self_consumption += metrics['self_consumption_rate']
                total_self_sufficiency += metrics['self_sufficiency_rate']
            
            avg_self_consumption = total_self_consumption / len(solar_profiles)
            avg_self_sufficiency = total_self_sufficiency / len(solar_profiles)
            
            results.append({
                'capacity': capacity,
                'self_consumption_rate': avg_self_consumption,
                'self_sufficiency_rate': avg_self_sufficiency,
                'combined_score': (avg_self_consumption + avg_self_sufficiency) / 2
            })
        
        # Find optimal capacity
        results_df = pd.DataFrame(results)
        optimal_idx = results_df['combined_score'].idxmax()
        optimal_capacity = results_df.loc[optimal_idx, 'capacity']
        
        return {
            'results': results_df,
            'optimal_capacity': optimal_capacity,
            'optimal_metrics': results_df.loc[optimal_idx].to_dict()
        }


def create_battery_from_user_input(battery_capacity: Optional[float] = None,
                                  battery_type: Optional[str] = None) -> BatteryModel:
    """
    Create battery model from user input
    
    Args:
        battery_capacity: Battery capacity in kWh
        battery_type: Type of battery
        
    Returns:
        BatteryModel instance or None if no input provided
    """
    if battery_capacity is None:
        return None
    
    specs = BatterySpecs(
        capacity=battery_capacity,
        battery_type=battery_type or 'lithium',
        efficiency=0,  # Will be set automatically
        max_charge_rate=0,  # Will be set automatically
        max_discharge_rate=0,  # Will be set automatically
        depth_of_discharge=0,  # Will be set automatically
        initial_soc=0.5
    )
    
    return BatteryModel(specs)


if __name__ == "__main__":
    # Example usage
    try:
        # Create a lithium battery
        specs = BatterySpecs(
            capacity=10,  # 10 kWh
            battery_type='lithium',
            efficiency=0.95,
            max_charge_rate=5,  # 5 kW
            max_discharge_rate=5,  # 5 kW
            depth_of_discharge=0.9,
            initial_soc=0.5
        )
        
        battery = BatteryModel(specs)
        
        # Create sample solar and load profiles
        hours = np.arange(24)
        solar_profile = np.maximum(0, 5 * np.sin((hours - 6) * np.pi / 12))
        load_profile = 2 + np.sin(hours * np.pi / 12) + 0.5 * np.random.randn(24)
        
        # Simulate one day
        results = battery.simulate_day(solar_profile, load_profile)
        metrics = battery.calculate_metrics(results)
        
        print("Battery Simulation Results:")
        for key, value in metrics.items():
            print(f"  {key}: {value:.4f}")
        
        # Plot results
        battery.plot_simulation_results(results)
        
    except Exception as e:
        print(f"Error in battery simulation: {e}")
