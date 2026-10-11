"""
Demonstration of Model Training Script
"""

import pandas as pd
import numpy as np
from train_model import SolarModelTrainer, train_solar_model
import matplotlib.pyplot as plt

def create_sample_dataset(n_samples=100):
    """Create a sample dataset for demonstration"""
    
    print("📊 Creating sample dataset for demonstration...")
    
    # Generate dates
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    # Generate synthetic features
    np.random.seed(42)
    
    # Base GHI with seasonal pattern
    day_of_year = dates.dayofyear
    base_ghi = 5000 + 2000 * np.sin(2 * np.pi * (day_of_year - 80) / 365)
    
    # Add weather effects
    temperature = 15 + 10 * np.sin(2 * np.pi * (day_of_year - 80) / 365) + np.random.normal(0, 2, n_samples)
    cloud_opacity = np.random.beta(2, 5, n_samples)  # Skewed distribution for clouds
    
    # Apply weather effects on GHI
    ghi = base_ghi * (1 - 0.7 * cloud_opacity) + np.random.normal(0, 500, n_samples)
    ghi = np.maximum(0, ghi)  # Ensure non-negative
    
    # Create time features
    df = pd.DataFrame({
        'date': dates,
        'GHI': ghi,
        'temperature': temperature,
        'cloud_opacity': cloud_opacity,
        'day_of_year': day_of_year,
        'month': dates.month,
        'day': dates.day,
        'year': dates.year,
        'week_of_year': dates.isocalendar().week,
        'day_of_week': dates.dayofweek,
        'quarter': dates.quarter
    })
    
    # Add cyclical features
    df['day_of_year_sin'] = np.sin(2 * np.pi * df['day_of_year'] / 365)
    df['day_of_year_cos'] = np.cos(2 * np.pi * df['day_of_year'] / 365)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)
    df['day_of_week_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['day_of_week_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    
    # Add seasonal indicators
    df['is_summer'] = df['month'].isin([6, 7, 8]).astype(int)
    df['is_winter'] = df['month'].isin([12, 1, 2]).astype(int)
    df['is_spring'] = df['month'].isin([3, 4, 5]).astype(int)
    df['is_autumn'] = df['month'].isin([9, 10, 11]).astype(int)
    
    # Add tilt angle (constant for simplicity)
    df['tilt_angle'] = 33.9
    
    df = df.set_index('date')
    
    print(f"✅ Created dataset with {len(df)} samples and {len(df.columns)} features")
    print(f"   GHI range: {df['GHI'].min():.0f} - {df['GHI'].max():.0f} Wh/m²/day")
    print(f"   Temperature range: {df['temperature'].min():.1f} - {df['temperature'].max():.1f}°C")
    
    return df

def demonstrate_model_training():
    """Demonstrate the model training functionality"""
    
    print("🤖 Model Training Demonstration")
    print("=" * 40)
    
    # Create sample dataset
    df = create_sample_dataset(n_samples=100)
    
    # Initialize trainer
    trainer = SolarModelTrainer(random_state=42)
    
    print("\n🚀 Running Complete Training Pipeline...")
    
    # Run complete training with optimized parameters
    results = trainer.run_complete_training(
        df=df,
        target_column='GHI',
        plot_correlation=True,
        plot_predictions=True,
        plot_residuals=True,
        top_features=15
    )
    
    # Display results summary
    print(f"\n📊 Training Results Summary:")
    print(f"   Model: SGDRegressor with StandardScaler")
    print(f"   Dataset: {df.shape}")
    print(f"   Training samples: {len(trainer.X_train)}")
    print(f"   Test samples: {len(trainer.X_test)}")
    print(f"   R² Score: {results['r2']:.4f}")
    print(f"   MSE: {results['mse']:.2f}")
    print(f"   RMSE: {np.sqrt(results['mse']):.2f}")
    
    # Show feature importance
    print(f"\n🔍 Top 5 Most Important Features:")
    top_features = results['coefficients'].head(5)
    for i, (_, row) in enumerate(top_features.iterrows()):
        sign = "+" if row['Coefficient'] >= 0 else ""
        print(f"   {i+1}. {row['Feature']}: {sign}{row['Coefficient']:.4f}")
    
    # Model insights
    print(f"\n💡 Model Insights:")
    
    # Check if temperature is important
    temp_coef = results['coefficients'][results['coefficients']['Feature'] == 'temperature']
    if len(temp_coef) > 0:
        temp_val = temp_coef.iloc[0]['Coefficient']
        print(f"   • Temperature impact: {'Positive' if temp_val > 0 else 'Negative'} ({temp_val:.2f})")
    
    # Check seasonal patterns
    seasonal_features = ['day_of_year_sin', 'day_of_year_cos', 'month_sin', 'month_cos']
    seasonal_coefs = results['coefficients'][results['coefficients']['Feature'].isin(seasonal_features)]
    if len(seasonal_coefs) > 0:
        max_seasonal = seasonal_coefs.loc[seasonal_coefs['Absolute_Coefficient'].idxmax()]
        print(f"   • Strongest seasonal feature: {max_seasonal['Feature']} ({max_seasonal['Coefficient']:.2f})")
    
    # Performance interpretation
    r2 = results['r2']
    if r2 > 0.8:
        print(f"   • Model performance: Excellent (R² = {r2:.3f})")
    elif r2 > 0.6:
        print(f"   • Model performance: Good (R² = {r2:.3f})")
    elif r2 > 0.4:
        print(f"   • Model performance: Moderate (R² = {r2:.3f})")
    else:
        print(f"   • Model performance: Needs improvement (R² = {r2:.3f})")
    
    return results

def test_hyperparameters():
    """Test different hyperparameters"""
    
    print("\n🔧 Testing Different Hyperparameters")
    print("=" * 45)
    
    # Create dataset
    df = create_sample_dataset(n_samples=100)
    
    # Test different learning rates
    learning_rates = [0.001, 0.01, 0.1]
    results = []
    
    for lr in learning_rates:
        print(f"\n📊 Testing learning rate: {lr}")
        
        trainer = SolarModelTrainer(random_state=42)
        trainer.prepare_data(df, 'GHI')
        trainer.train_model(learning_rate=lr, max_iter=1000)
        mse, r2 = trainer.evaluate_model()
        
        results.append({
            'learning_rate': lr,
            'mse': mse,
            'r2': r2
        })
        
        print(f"   R²: {r2:.4f}, MSE: {mse:.2f}")
    
    # Find best learning rate
    best_result = max(results, key=lambda x: x['r2'])
    print(f"\n🏆 Best learning rate: {best_result['learning_rate']} (R² = {best_result['r2']:.4f})")
    
    return results

if __name__ == "__main__":
    try:
        # Run main demonstration
        results = demonstrate_model_training()
        
        # Test hyperparameters
        hyperparameter_results = test_hyperparameters()
        
        print(f"\n🎉 Model Training Demonstration Completed!")
        print(f"   ✅ Successfully trained SGDRegressor model")
        print(f"   ✅ Generated all required plots")
        print(f"   ✅ Displayed model coefficients")
        print(f"   ✅ Computed MSE and R² metrics")
        print(f"   ✅ Tested hyperparameter variations")
        
    except Exception as e:
        print(f"❌ Error during demonstration: {e}")
        raise
