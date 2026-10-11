"""
Example usage of the model training script with real solar data
"""

import pandas as pd
import numpy as np
from train_model import train_solar_model, SolarModelTrainer

def main():
    """Example of training model with real solar data"""
    
    print("🌞 Solar Power Prediction Model Training Example")
    print("=" * 55)
    
    # Try to load real dataset
    try:
        # Load the comprehensive dataset created earlier
        df = pd.read_csv('data/beirut_comprehensive_dataset.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded real dataset: {df.shape}")
        
        # Ensure we have the target column
        if 'GHI' not in df.columns:
            print("❌ GHI column not found in dataset")
            return
        
        # Get numeric columns only
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        print(f"📊 Dataset features: {list(df.columns)}")
        print(f"📅 Date range: {df.index.min().date()} to {df.index.max().date()}")
        
        # Basic statistics
        print(f"\n📈 Target Variable Statistics (GHI):")
        print(f"   Mean: {df['GHI'].mean():.0f} Wh/m²/day")
        print(f"   Std: {df['GHI'].std():.0f} Wh/m²/day")
        print(f"   Range: {df['GHI'].min():.0f} - {df['GHI'].max():.0f} Wh/m²/day")
        
        # Train model using the standalone function
        print(f"\n🚀 Training Model with Standalone Function...")
        results = train_solar_model(
            df=df,
            target_column='GHI',
            learning_rate=0.001,  # Lower learning rate for better convergence
            max_iter=2000,
            test_size=0.3,
            random_state=42
        )
        
        # Display results
        print(f"\n🎯 Training Results:")
        print(f"   Algorithm: SGDRegressor with StandardScaler")
        print(f"   Features used: {len(results['coefficients'])}")
        print(f"   R² Score: {results['r2']:.4f}")
        print(f"   MSE: {results['mse']:.2f}")
        print(f"   RMSE: {np.sqrt(results['mse']):.2f}")
        
        # Feature importance
        print(f"\n🔍 Top 10 Most Important Features:")
        top_features = results['coefficients'].head(10)
        for i, (_, row) in enumerate(top_features.iterrows()):
            sign = "+" if row['Coefficient'] >= 0 else ""
            print(f"   {i+1:2d}. {row['Feature']:25s}: {sign}{row['Coefficient']:8.4f}")
        
        # Model interpretation
        print(f"\n💡 Model Interpretation:")
        
        # Check most important feature
        most_important = results['coefficients'].iloc[0]
        print(f"   • Most important feature: {most_important['Feature']}")
        print(f"   • Impact: {most_important['Coefficient']:.2f}")
        
        # Performance assessment
        r2 = results['r2']
        if r2 > 0.7:
            print(f"   • Model performance: Excellent (R² = {r2:.3f})")
        elif r2 > 0.5:
            print(f"   • Model performance: Good (R² = {r2:.3f})")
        elif r2 > 0.3:
            print(f"   • Model performance: Moderate (R² = {r2:.3f})")
        else:
            print(f"   • Model performance: Needs improvement (R² = {r2:.3f})")
        
        # Feature categories analysis
        print(f"\n📂 Feature Categories Performance:")
        
        # Time-based features
        time_features = ['day_of_year', 'month', 'day', 'week_of_year', 'day_of_week']
        time_coefs = results['coefficients'][results['coefficients']['Feature'].isin(time_features)]
        if len(time_coefs) > 0:
            max_time = time_coefs.loc[time_coefs['Absolute_Coefficient'].idxmax()]
            print(f"   • Best time feature: {max_time['Feature']} ({max_time['Coefficient']:.2f})")
        
        # Cyclical features
        cyclical_features = ['day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']
        cyclical_coefs = results['coefficients'][results['coefficients']['Feature'].isin(cyclical_features)]
        if len(cyclical_coefs) > 0:
            max_cyclical = cyclical_coefs.loc[cyclical_coefs['Absolute_Coefficient'].idxmax()]
            print(f"   • Best cyclical feature: {max_cyclical['Feature']} ({max_cyclical['Coefficient']:.2f})")
        
        # Weather features
        weather_features = ['temperature', 'cloud_opacity']
        weather_coefs = results['coefficients'][results['coefficients']['Feature'].isin(weather_features)]
        if len(weather_coefs) > 0:
            max_weather = weather_coefs.loc[weather_coefs['Absolute_Coefficient'].idxmax()]
            print(f"   • Best weather feature: {max_weather['Feature']} ({max_weather['Coefficient']:.2f})")
        
        # Recommendations
        print(f"\n📋 Recommendations:")
        if r2 < 0.5:
            print("   • Consider collecting more training data")
            print("   • Try feature engineering to create more predictive features")
            print("   • Experiment with different model hyperparameters")
            print("   • Consider non-linear models for comparison")
        
        if 'temperature' in results['coefficients']['Feature'].values:
            temp_coef = results['coefficients'][results['coefficients']['Feature'] == 'temperature'].iloc[0]['Coefficient']
            if abs(temp_coef) < 50:
                print("   • Temperature has low impact, consider additional weather variables")
        
        print(f"\n✅ Model training example completed successfully!")
        
        return results
        
    except FileNotFoundError:
        print("❌ Dataset not found. Please run dataset_builder.py first to create the dataset.")
        print("   You can create a dataset using:")
        print("   python dataset_builder.py")
        return None
        
    except Exception as e:
        print(f"❌ Error during training: {e}")
        return None

def demonstrate_hyperparameter_tuning():
    """Demonstrate hyperparameter tuning"""
    
    print(f"\n🔧 Hyperparameter Tuning Example")
    print("=" * 40)
    
    try:
        # Load dataset
        df = pd.read_csv('data/beirut_comprehensive_dataset.csv', index_col=0, parse_dates=True)
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        # Test different learning rates
        learning_rates = [0.0001, 0.001, 0.01, 0.1]
        results = []
        
        print("📊 Testing different learning rates:")
        
        for lr in learning_rates:
            trainer = SolarModelTrainer(random_state=42)
            trainer.prepare_data(df, 'GHI')
            trainer.train_model(learning_rate=lr, max_iter=1000)
            mse, r2 = trainer.evaluate_model()
            
            results.append({
                'learning_rate': lr,
                'mse': mse,
                'r2': r2
            })
            
            print(f"   LR {lr:6.4f}: R² = {r2:7.4f}, MSE = {mse:8.0f}")
        
        # Find best learning rate
        best_result = max(results, key=lambda x: x['r2'])
        print(f"\n🏆 Best learning rate: {best_result['learning_rate']}")
        print(f"   R²: {best_result['r2']:.4f}")
        print(f"   MSE: {best_result['mse']:.2f}")
        
        return results
        
    except Exception as e:
        print(f"❌ Error in hyperparameter tuning: {e}")
        return None

if __name__ == "__main__":
    # Run main example
    results = main()
    
    # Run hyperparameter tuning
    if results:
        demonstrate_hyperparameter_tuning()
    
    print(f"\n🎯 Model Training Script Features:")
    print("• ✅ Uses SGDRegressor with gradient descent")
    print("• ✅ Splits data into train/test sets")
    print("• ✅ Computes MSE and R² metrics")
    print("• ✅ Generates correlation heatmap")
    print("• ✅ Plots actual vs predicted values")
    print("• ✅ Creates residual analysis plots")
    print("• ✅ Prints model coefficients")
    print("• ✅ Supports hyperparameter tuning")
    print("• ✅ Provides comprehensive evaluation")
