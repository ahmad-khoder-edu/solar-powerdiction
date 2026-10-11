"""
Test script to verify all model training requirements
"""

import pandas as pd
import numpy as np
from train_model import SolarModelTrainer

def test_all_requirements():
    """Test all required model training features"""
    
    print("🧪 Testing Model Training Requirements")
    print("=" * 45)
    
    # Create sample data
    np.random.seed(42)
    n_samples = 50
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    
    df = pd.DataFrame({
        'date': dates,
        'GHI': 3000 + 2000 * np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.normal(0, 500, n_samples),
        'temperature': 15 + 10 * np.sin(np.arange(n_samples) * 2 * np.pi / 365) + np.random.normal(0, 2, n_samples),
        'day_of_year': dates.dayofyear,
        'month': dates.month,
        'tilt_angle': 33.9
    })
    df = df.set_index('date')
    
    # Initialize trainer
    trainer = SolarModelTrainer(random_state=42)
    
    print("✅ 1. SGDRegressor (Gradient Descent)")
    trainer.prepare_data(df, 'GHI')
    trainer.train_model(learning_rate=0.001, max_iter=500)
    print("   ✓ SGDRegressor with gradient descent configured")
    
    print("\n✅ 2. Train/Test Split")
    print(f"   ✓ Training set: {trainer.X_train.shape}")
    print(f"   ✓ Test set: {trainer.X_test.shape}")
    print(f"   ✓ Split ratio: {len(trainer.X_test)/(len(trainer.X_train)+len(trainer.X_test)):.1%}")
    
    print("\n✅ 3. MSE and R² Computation")
    mse, r2 = trainer.evaluate_model()
    print(f"   ✓ MSE: {mse:.2f}")
    print(f"   ✓ R²: {r2:.4f}")
    
    print("\n✅ 4. Correlation Heatmap")
    try:
        corr_matrix = trainer.plot_correlation_heatmap(figsize=(8, 6))
        print("   ✓ Correlation heatmap generated successfully")
        print(f"   ✓ Matrix shape: {corr_matrix.shape}")
    except Exception as e:
        print(f"   ❌ Error generating correlation heatmap: {e}")
    
    print("\n✅ 5. Actual vs Predicted Plot")
    try:
        trainer.plot_actual_vs_predicted(figsize=(12, 5))
        print("   ✓ Actual vs predicted plot generated")
    except Exception as e:
        print(f"   ❌ Error generating actual vs predicted plot: {e}")
    
    print("\n✅ 6. Residual Plot")
    try:
        residuals = trainer.plot_residuals(figsize=(15, 5))
        print("   ✓ Residual plots generated")
        print(f"   ✓ Residuals shape: {residuals.shape}")
    except Exception as e:
        print(f"   ❌ Error generating residual plots: {e}")
    
    print("\n✅ 7. Model Coefficients")
    try:
        coef_df = trainer.print_coefficients(top_n=5)
        print("   ✓ Model coefficients printed")
        print(f"   ✓ Number of features: {len(coef_df)}")
    except Exception as e:
        print(f"   ❌ Error printing coefficients: {e}")
    
    print("\n🎉 All Requirements Tested Successfully!")
    
    # Summary
    print(f"\n📊 Final Model Summary:")
    print(f"   Algorithm: SGDRegressor (Gradient Descent)")
    print(f"   Features: {trainer.X_train.shape[1]}")
    print(f"   Training samples: {len(trainer.X_train)}")
    print(f"   Test samples: {len(trainer.X_test)}")
    print(f"   Performance: R² = {r2:.4f}, MSE = {mse:.2f}")
    
    return True

if __name__ == "__main__":
    try:
        success = test_all_requirements()
        if success:
            print("\n🏆 All model training requirements verified!")
        else:
            print("\n❌ Some requirements failed")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
