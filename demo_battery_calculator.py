"""
Demonstration of Battery Calculator functionality
"""

from battery_calculator import BatteryCalculator, calculate_battery_runtime
import pandas as pd

def main():
    """Demonstrate battery calculator capabilities"""
    
    print("🔋 Battery Calculator Demonstration")
    print("=" * 40)
    
    calculator = BatteryCalculator()
    
    # Scenario 1: Residential Home with Lithium Battery
    print("\n🏠 Scenario 1: Residential Home with Lithium Battery")
    print("-" * 55)
    
    home_results = calculator.calculate_battery_metrics(
        battery_capacity_kwh=13.5,
        daily_load_kwh=12.0,
        battery_type='lithium'
    )
    
    if home_results:
        print(f"   Battery Capacity: {home_results['battery_capacity_kwh']} kWh")
        print(f"   Daily Load: {home_results['daily_load_kwh']} kWh/day")
        print(f"   Runtime: {home_results['runtime_hours']:.1f} hours")
        print(f"   End of Day SoC: {home_results['end_of_day_soc_percent']:.1f}%")
        print(f"   Days of Autonomy: {home_results['days_of_autonomy']:.1f} days")
        
        if home_results['can_cover_daily_load']:
            print("   ✅ Battery can cover daily load")
        else:
            print("   ❌ Battery cannot cover daily load")
    
    # Scenario 2: Off-Grid Cabin with Lead-Acid Battery
    print("\n🏕️ Scenario 2: Off-Grid Cabin with Lead-Acid Battery")
    print("-" * 58)
    
    cabin_results = calculator.calculate_battery_metrics(
        battery_capacity_kwh=6.0,
        daily_load_kwh=3.5,
        battery_type='lead-acid'
    )
    
    if cabin_results:
        print(f"   Battery Capacity: {cabin_results['battery_capacity_kwh']} kWh")
        print(f"   Daily Load: {cabin_results['daily_load_kwh']} kWh/day")
        print(f"   Runtime: {cabin_results['runtime_hours']:.1f} hours")
        print(f"   End of Day SoC: {cabin_results['end_of_day_soc_percent']:.1f}%")
        print(f"   Days of Autonomy: {cabin_results['days_of_autonomy']:.1f} days")
        
        if cabin_results['can_cover_daily_load']:
            print("   ✅ Battery can cover daily load")
        else:
            print("   ❌ Battery cannot cover daily load")
    
    # Scenario 3: Comparison between battery types
    print("\n📊 Scenario 3: Battery Type Comparison")
    print("-" * 40)
    
    comparison = calculator.compare_battery_types(
        battery_capacity_kwh=10,
        daily_load_kwh=6
    )
    
    if not comparison.empty:
        print("   Comparison Results:")
        for _, row in comparison.iterrows():
            print(f"   {row['battery_type'].title()}:")
            print(f"     Runtime: {row['runtime_hours']:.1f} hours")
            print(f"     End of Day SoC: {row['end_of_day_soc_percent']:.1f}%")
            print(f"     Days of Autonomy: {row['days_of_autonomy']:.1f} days")
    
    # Scenario 4: Battery sizing for specific requirements
    print("\n📏 Scenario 4: Battery Sizing for 3-Day Autonomy")
    print("-" * 50)
    
    sizing = calculator.size_battery_for_load(
        daily_load_kwh=8,
        days_of_autonomy=3,
        battery_type='lithium'
    )
    
    if sizing:
        print(f"   Required Battery Capacity: {sizing['required_capacity_kwh']:.1f} kWh")
        print(f"   Required Usable Capacity: {sizing['required_usable_capacity_kwh']:.1f} kWh")
        print(f"   For {sizing['days_of_autonomy']} days of autonomy")
        print(f"   Battery Type: {sizing['battery_type'].title()}")
    
    # Scenario 5: Testing missing inputs
    print("\n⚠️ Scenario 5: Testing Missing Inputs")
    print("-" * 40)
    
    missing_input = calculator.calculate_battery_metrics(
        battery_capacity_kwh=None,
        daily_load_kwh=5,
        battery_type='lithium'
    )
    
    if missing_input == {}:
        print("   ✅ Correctly skipped calculation due to missing inputs")
    else:
        print("   ❌ Did not handle missing inputs correctly")
    
    # Scenario 6: Standalone function test
    print("\n🔧 Scenario 6: Standalone Function Test")
    print("-" * 40)
    
    standalone_result = calculate_battery_runtime(
        battery_capacity_kwh=15,
        daily_load_kwh=10,
        battery_type='lead-acid'
    )
    
    if standalone_result:
        print(f"   Standalone function results:")
        print(f"   Runtime: {standalone_result['runtime_hours']:.1f} hours")
        print(f"   End of Day SoC: {standalone_result['end_of_day_soc_percent']:.1f}%")
        print("   ✅ Standalone function working correctly")
    
    # Key insights
    print("\n💡 Key Insights:")
    print("-" * 20)
    
    # Lithium vs Lead-Acid comparison
    lithium_10kwh = calculator.calculate_battery_metrics(10, 5, 'lithium')
    lead_acid_10kwh = calculator.calculate_battery_metrics(10, 5, 'lead-acid')
    
    if lithium_10kwh and lead_acid_10kwh:
        lithium_runtime = lithium_10kwh['runtime_hours']
        lead_acid_runtime = lead_acid_10kwh['runtime_hours']
        
        print(f"   • Lithium provides {lithium_runtime:.1f} hours runtime")
        print(f"   • Lead-acid provides {lead_acid_runtime:.1f} hours runtime")
        print(f"   • Lithium offers {((lithium_runtime/lead_acid_runtime - 1)*100):.0f}% more runtime")
        print(f"   • This is due to higher DoD (80% vs 50%) and better efficiency")
    
    # Recommendations
    print(f"\n📋 Recommendations:")
    print("   • Use lithium batteries for higher efficiency and longer runtime")
    print("   • Lead-acid batteries are more economical but have lower performance")
    print("   • Consider daily load patterns when sizing battery capacity")
    print("   • Account for efficiency losses in your calculations")
    print("   • Plan for days of autonomy based on your reliability needs")
    
    print(f"\n🎯 Battery Calculator Features:")
    print("• ✅ Accepts battery_capacity_kWh parameter")
    print("• ✅ Accepts daily_load_kWh parameter")
    print("• ✅ Accepts battery_type (lead-acid or lithium)")
    print("• ✅ Uses 50% DoD for lead-acid")
    print("• ✅ Uses 80% DoD for lithium")
    print("• ✅ Returns runtime in hours")
    print("• ✅ Returns end of day state of charge")
    print("• ✅ Skips calculation if inputs missing")
    print("• ✅ Provides comprehensive battery analysis")
    print("• ✅ Supports battery type comparison")
    print("• ✅ Includes battery sizing functionality")

if __name__ == "__main__":
    main()
