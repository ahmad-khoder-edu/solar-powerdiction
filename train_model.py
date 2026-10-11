"""
Model Training Script for Solar Power Prediction
Uses SGDRegressor with comprehensive evaluation and visualization
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
import warnings
warnings.filterwarnings('ignore')

# Set style for better plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


class SolarModelTrainer:
    """
    Comprehensive model training class for solar power prediction
    """
    
    def __init__(self, random_state: int = 42):
        """
        Initialize the model trainer
        
        Args:
            random_state: Random seed for reproducibility
        """
        self.random_state = random_state
        self.model = None
        self.scaler = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.y_pred = None
        self.mse = None
        self.r2 = None
        
    def prepare_data(self, df: pd.DataFrame, target_column: str = 'GHI',
                    test_size: float = 0.2) -> tuple:
        """
        Prepare data for training
        
        Args:
            df: Input DataFrame
            target_column: Name of target column
            test_size: Proportion of data for testing
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        print("📊 Preparing data for training...")
        
        # Separate features and target
        X = df.drop(columns=[target_column])
        y = df[target_column]
        
        # Get numeric columns only
        numeric_columns = X.select_dtypes(include=[np.number]).columns
        X = X[numeric_columns]
        
        print(f"   Features: {list(X.columns)}")
        print(f"   Target: {target_column}")
        print(f"   Dataset shape: {X.shape}")
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=test_size, random_state=self.random_state
        )
        
        print(f"   Training set: {self.X_train.shape}")
        print(f"   Test set: {self.X_test.shape}")
        
        return self.X_train, self.X_test, self.y_train, self.y_test
    
    def train_model(self, learning_rate: float = 0.01, max_iter: int = 1000,
                   tol: float = 1e-3, penalty: str = 'l2', alpha: float = 0.0001):
        """
        Train SGDRegressor model
        
        Args:
            learning_rate: Learning rate for gradient descent
            max_iter: Maximum number of iterations
            tol: Tolerance for stopping criterion
            penalty: Regularization type ('l2', 'l1', 'elasticnet', 'none')
            alpha: Regularization strength
        """
        print("\n🤖 Training SGDRegressor model...")
        
        # Create pipeline with scaling and model
        self.model = Pipeline([
            ('scaler', StandardScaler()),
            ('sgd', SGDRegressor(
                learning_rate='constant',
                eta0=learning_rate,
                max_iter=max_iter,
                tol=tol,
                penalty=penalty,
                alpha=alpha,
                random_state=self.random_state,
                verbose=0
            ))
        ])
        
        # Train model
        self.model.fit(self.X_train, self.y_train)
        
        print(f"   ✓ Model trained successfully!")
        print(f"   ✓ Learning rate: {learning_rate}")
        print(f"   ✓ Max iterations: {max_iter}")
        print(f"   ✓ Penalty: {penalty}")
        print(f"   ✓ Alpha: {alpha}")
        
        return self.model
    
    def evaluate_model(self):
        """
        Evaluate the trained model
        """
        print("\n📈 Evaluating model performance...")
        
        # Make predictions
        self.y_pred = self.model.predict(self.X_test)
        
        # Calculate metrics
        self.mse = mean_squared_error(self.y_test, self.y_pred)
        self.r2 = r2_score(self.y_test, self.y_pred)
        rmse = np.sqrt(self.mse)
        
        print(f"   Mean Squared Error (MSE): {self.mse:.2f}")
        print(f"   Root Mean Squared Error (RMSE): {rmse:.2f}")
        print(f"   R² Score: {self.r2:.4f}")
        
        # Additional metrics
        mae = np.mean(np.abs(self.y_test - self.y_pred))
        mape = np.mean(np.abs((self.y_test - self.y_pred) / self.y_test)) * 100
        
        print(f"   Mean Absolute Error (MAE): {mae:.2f}")
        print(f"   Mean Absolute Percentage Error (MAPE): {mape:.2f}%")
        
        return self.mse, self.r2
    
    def print_coefficients(self, top_n: int = 10):
        """
        Print model coefficients
        
        Args:
            top_n: Number of top coefficients to show
        """
        print(f"\n🔍 Model Coefficients (Top {top_n}):")
        print("-" * 50)
        
        # Get coefficients from the SGDRegressor step
        sgd_model = self.model.named_steps['sgd']
        coefficients = sgd_model.coef_
        feature_names = self.X_train.columns
        
        # Create DataFrame for better display
        coef_df = pd.DataFrame({
            'Feature': feature_names,
            'Coefficient': coefficients,
            'Absolute_Coefficient': np.abs(coefficients)
        }).sort_values('Absolute_Coefficient', ascending=False)
        
        # Print top coefficients
        for i, (_, row) in enumerate(coef_df.head(top_n).iterrows()):
            sign = "+" if row['Coefficient'] >= 0 else ""
            coef_val = row['Coefficient']
            print(f"   {i+1:2d}. {row['Feature']:25s}: {sign}{coef_val:8.4f}")
        
        # Print intercept
        intercept = sgd_model.intercept_[0] if hasattr(sgd_model.intercept_, '__len__') else sgd_model.intercept_
        sign = "+" if intercept >= 0 else ""
        print(f"\n   Intercept: {sign}{intercept:.4f}")
        
        return coef_df
    
    def plot_correlation_heatmap(self, figsize: tuple = (12, 10)):
        """
        Plot correlation heatmap of features
        
        Args:
            figsize: Figure size
        """
        print("\n📊 Generating correlation heatmap...")
        
        # Combine training features and target for correlation
        train_data = self.X_train.copy()
        train_data['target'] = self.y_train
        
        # Calculate correlation matrix
        corr_matrix = train_data.corr()
        
        # Create figure
        plt.figure(figsize=figsize)
        
        # Create mask for upper triangle
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        
        # Draw heatmap
        sns.heatmap(corr_matrix, 
                   mask=mask,
                   annot=True, 
                   cmap='coolwarm', 
                   center=0,
                   square=True, 
                   fmt='.2f',
                   cbar_kws={"shrink": .8},
                   annot_kws={"size": 8})
        
        plt.title('Feature Correlation Heatmap', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()
        
        return corr_matrix
    
    def plot_actual_vs_predicted(self, figsize: tuple = (15, 6)):
        """
        Plot actual vs predicted values
        
        Args:
            figsize: Figure size
        """
        print("\n📈 Generating actual vs predicted plot...")
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
        
        # Scatter plot
        ax1.scatter(self.y_test, self.y_pred, alpha=0.6, s=50)
        
        # Perfect prediction line
        min_val = min(self.y_test.min(), self.y_pred.min())
        max_val = max(self.y_test.max(), self.y_pred.max())
        ax1.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')
        
        ax1.set_xlabel('Actual Values', fontsize=12)
        ax1.set_ylabel('Predicted Values', fontsize=12)
        ax1.set_title('Actual vs Predicted', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Add R² annotation
        ax1.text(0.05, 0.95, f'R² = {self.r2:.4f}', 
                transform=ax1.transAxes, fontsize=12,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        # Time series plot
        test_indices = range(len(self.y_test))
        ax2.plot(test_indices, self.y_test, 'o-', label='Actual', alpha=0.7, markersize=4)
        ax2.plot(test_indices, self.y_pred, 's-', label='Predicted', alpha=0.7, markersize=4)
        
        ax2.set_xlabel('Sample Index', fontsize=12)
        ax2.set_ylabel('GHI (Wh/m²/day)', fontsize=12)
        ax2.set_title('Actual vs Predicted Over Test Set', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_residuals(self, figsize: tuple = (15, 5)):
        """
        Plot residual analysis
        
        Args:
            figsize: Figure size
        """
        print("\n📉 Generating residual plots...")
        
        # Calculate residuals
        residuals = self.y_test - self.y_pred
        
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=figsize)
        
        # Residuals vs Predicted
        ax1.scatter(self.y_pred, residuals, alpha=0.6, s=50)
        ax1.axhline(y=0, color='r', linestyle='--', lw=2)
        ax1.set_xlabel('Predicted Values', fontsize=12)
        ax1.set_ylabel('Residuals', fontsize=12)
        ax1.set_title('Residuals vs Predicted', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # Residuals histogram
        ax2.hist(residuals, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--', lw=2)
        ax2.set_xlabel('Residuals', fontsize=12)
        ax2.set_ylabel('Frequency', fontsize=12)
        ax2.set_title('Residuals Distribution', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # Q-Q plot
        from scipy import stats
        stats.probplot(residuals, dist="norm", plot=ax3)
        ax3.set_title('Q-Q Plot', fontsize=14, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        return residuals
    
    def run_complete_training(self, df: pd.DataFrame, target_column: str = 'GHI',
                           plot_correlation: bool = True, plot_predictions: bool = True,
                           plot_residuals: bool = True, top_features: int = 10):
        """
        Run complete training pipeline
        
        Args:
            df: Input DataFrame
            target_column: Target column name
            plot_correlation: Whether to plot correlation heatmap
            plot_predictions: Whether to plot actual vs predicted
            plot_residuals: Whether to plot residual analysis
            top_features: Number of top features to show
            
        Returns:
            Dictionary with training results
        """
        print("🚀 Starting Complete Model Training Pipeline")
        print("=" * 60)
        
        # Prepare data
        self.prepare_data(df, target_column)
        
        # Train model
        self.train_model()
        
        # Evaluate model
        mse, r2 = self.evaluate_model()
        
        # Print coefficients
        coef_df = self.print_coefficients(top_features)
        
        # Generate plots
        if plot_correlation:
            corr_matrix = self.plot_correlation_heatmap()
        
        if plot_predictions:
            self.plot_actual_vs_predicted()
        
        if plot_residuals:
            residuals = self.plot_residuals()
        
        # Compile results
        results = {
            'model': self.model,
            'mse': mse,
            'r2': r2,
            'coefficients': coef_df,
            'predictions': self.y_pred,
            'residuals': residuals if plot_residuals else None,
            'correlation_matrix': corr_matrix if plot_correlation else None
        }
        
        print(f"\n🎉 Training completed successfully!")
        print(f"   Final R² Score: {r2:.4f}")
        print(f"   Final MSE: {mse:.2f}")
        
        return results


def train_solar_model(df: pd.DataFrame, target_column: str = 'GHI',
                     learning_rate: float = 0.01, max_iter: int = 1000,
                     test_size: float = 0.2, random_state: int = 42):
    """
    Standalone function to train solar prediction model
    
    Args:
        df: Input DataFrame
        target_column: Target column name
        learning_rate: Learning rate for SGD
        max_iter: Maximum iterations
        test_size: Test set proportion
        random_state: Random seed
        
    Returns:
        Dictionary with training results
    """
    trainer = SolarModelTrainer(random_state=random_state)
    
    return trainer.run_complete_training(
        df=df,
        target_column=target_column,
        plot_correlation=True,
        plot_predictions=True,
        plot_residuals=True,
        top_features=10
    )


if __name__ == "__main__":
    # Example usage with sample data
    print("🌞 Solar Power Prediction Model Training")
    print("=" * 50)
    
    # Load sample dataset (if available)
    try:
        # Try to load a previously created dataset
        df = pd.read_csv('data/beirut_comprehensive_dataset.csv', index_col=0, parse_dates=True)
        print(f"✅ Loaded dataset: {df.shape}")
        
        # Remove any non-numeric columns except the index
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df = df[numeric_cols]
        
        # Train model
        results = train_solar_model(df, target_column='GHI')
        
        print(f"\n📊 Final Results:")
        print(f"   Model: SGDRegressor with StandardScaler")
        print(f"   R² Score: {results['r2']:.4f}")
        print(f"   MSE: {results['mse']:.2f}")
        print(f"   Top feature: {results['coefficients'].iloc[0]['Feature']}")
        
    except FileNotFoundError:
        print("❌ No dataset found. Please run dataset_builder.py first to create a dataset.")
    except Exception as e:
        print(f"❌ Error during training: {e}")
