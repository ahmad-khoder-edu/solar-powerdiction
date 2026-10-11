"""
Model Training Module for Solar Power Prediction
Implements multiple linear regression with gradient descent and ML best practices
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression, SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Tuple, Dict, Any, Optional
import warnings
warnings.filterwarnings('ignore')


class MultipleLinearRegression:
    """
    Custom implementation of Multiple Linear Regression with Gradient Descent
    """
    
    def __init__(self, learning_rate: float = 0.01, n_iterations: int = 1000,
                 regularization: str = 'none', alpha: float = 0.1):
        """
        Initialize the model
        
        Args:
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations
            regularization: Type of regularization ('none', 'l1', 'l2')
            alpha: Regularization strength
        """
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.regularization = regularization
        self.alpha = alpha
        self.weights = None
        self.bias = None
        self.cost_history = []
        
    def fit(self, X: np.ndarray, y: np.ndarray) -> None:
        """
        Train the model using gradient descent
        
        Args:
            X: Feature matrix
            y: Target vector
        """
        n_samples, n_features = X.shape
        
        # Initialize parameters
        self.weights = np.zeros(n_features)
        self.bias = 0
        self.cost_history = []
        
        # Gradient descent
        for i in range(self.n_iterations):
            # Forward pass
            y_predicted = np.dot(X, self.weights) + self.bias
            
            # Calculate cost
            cost = self._calculate_cost(y, y_predicted)
            self.cost_history.append(cost)
            
            # Calculate gradients
            dw = (1 / n_samples) * np.dot(X.T, (y_predicted - y))
            db = (1 / n_samples) * np.sum(y_predicted - y)
            
            # Add regularization
            if self.regularization == 'l2':
                dw += (self.alpha / n_samples) * self.weights
            elif self.regularization == 'l1':
                dw += (self.alpha / n_samples) * np.sign(self.weights)
            
            # Update parameters
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
            
            # Print progress
            if i % 100 == 0:
                print(f"Iteration {i}: Cost = {cost:.4f}")
    
    def _calculate_cost(self, y: np.ndarray, y_predicted: np.ndarray) -> float:
        """Calculate mean squared error with regularization"""
        n_samples = len(y)
        mse = (1 / (2 * n_samples)) * np.sum((y_predicted - y) ** 2)
        
        # Add regularization
        if self.regularization == 'l2':
            mse += (self.alpha / (2 * n_samples)) * np.sum(self.weights ** 2)
        elif self.regularization == 'l1':
            mse += (self.alpha / n_samples) * np.sum(np.abs(self.weights))
        
        return mse
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Make predictions"""
        return np.dot(X, self.weights) + self.bias
    
    def get_coefficients(self) -> Tuple[np.ndarray, float]:
        """Get model coefficients"""
        return self.weights, self.bias


class SolarPowerModelTrainer:
    """
    Handles training and evaluation of solar power prediction models
    """
    
    def __init__(self):
        self.custom_model = None
        self.sklearn_model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.metrics = {}
        
    def prepare_data(self, X: pd.DataFrame, y: pd.Series, 
                    test_size: float = 0.2, random_state: int = 42) -> Tuple:
        """
        Prepare data for training
        
        Args:
            X: Feature matrix
            y: Target vector
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        self.feature_names = X.columns.tolist()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state, shuffle=False
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        print(f"Data prepared:")
        print(f"  Training set: {X_train_scaled.shape}")
        print(f"  Test set: {X_test_scaled.shape}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test
    
    def train_custom_model(self, X_train: np.ndarray, y_train: np.ndarray,
                          learning_rate: float = 0.01, n_iterations: int = 1000,
                          regularization: str = 'l2', alpha: float = 0.1) -> Dict[str, Any]:
        """
        Train custom multiple linear regression model
        
        Args:
            X_train: Training features
            y_train: Training target
            learning_rate: Learning rate for gradient descent
            n_iterations: Number of iterations
            regularization: Regularization type
            alpha: Regularization strength
            
        Returns:
            Training results
        """
        print("Training custom Multiple Linear Regression model...")
        
        self.custom_model = MultipleLinearRegression(
            learning_rate=learning_rate,
            n_iterations=n_iterations,
            regularization=regularization,
            alpha=alpha
        )
        
        self.custom_model.fit(X_train, y_train)
        
        # Get coefficients
        weights, bias = self.custom_model.get_coefficients()
        
        # Create coefficient DataFrame
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': weights
        }).sort_values('coefficient', key=abs, ascending=False)
        
        results = {
            'model': self.custom_model,
            'coefficients': coef_df,
            'bias': bias,
            'cost_history': self.custom_model.cost_history
        }
        
        print("✓ Custom model training completed")
        return results
    
    def train_sklearn_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Dict[str, Any]:
        """
        Train scikit-learn linear regression model for comparison
        
        Args:
            X_train: Training features
            y_train: Training target
            
        Returns:
            Training results
        """
        print("Training scikit-learn Linear Regression model...")
        
        # Try both LinearRegression and SGDRegressor
        models = {
            'LinearRegression': LinearRegression(),
            'SGDRegressor': SGDRegressor(random_state=42, max_iter=1000)
        }
        
        best_model = None
        best_score = -np.inf
        best_name = None
        
        for name, model in models.items():
            # Cross-validation
            cv_scores = cross_val_score(model, X_train, y_train, 
                                       cv=5, scoring='r2')
            avg_score = cv_scores.mean()
            
            print(f"  {name}: CV R² = {avg_score:.4f}")
            
            if avg_score > best_score:
                best_score = avg_score
                best_model = model
                best_name = name
        
        # Train best model
        best_model.fit(X_train, y_train)
        
        # Get coefficients
        coef_df = pd.DataFrame({
            'feature': self.feature_names,
            'coefficient': best_model.coef_
        }).sort_values('coefficient', key=abs, ascending=False)
        
        self.sklearn_model = best_model
        
        results = {
            'model': best_model,
            'model_name': best_name,
            'coefficients': coef_df,
            'cv_score': best_score
        }
        
        print(f"✓ Best sklearn model: {best_name}")
        return results
    
    def evaluate_model(self, model, X_test: np.ndarray, y_test: np.ndarray,
                       model_name: str = "Model") -> Dict[str, float]:
        """
        Evaluate model performance
        
        Args:
            model: Trained model
            X_test: Test features
            y_test: Test target
            model_name: Name of the model
            
        Returns:
            Evaluation metrics
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Calculate MAPE (Mean Absolute Percentage Error)
        mape = np.mean(np.abs((y_test - y_pred) / y_test)) * 100
        
        metrics = {
            'MSE': mse,
            'RMSE': rmse,
            'MAE': mae,
            'R²': r2,
            'MAPE': mape
        }
        
        self.metrics[model_name] = metrics
        
        print(f"\n{model_name} Performance:")
        print(f"  MSE: {mse:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  MAE: {mae:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MAPE: {mape:.2f}%")
        
        return metrics
    
    def plot_training_history(self, cost_history: list, save_path: Optional[str] = None):
        """
        Plot training cost history
        
        Args:
            cost_history: List of cost values during training
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 6))
        plt.plot(cost_history)
        plt.title('Training Cost History')
        plt.xlabel('Iteration')
        plt.ylabel('Cost (MSE)')
        plt.grid(True, alpha=0.3)
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_predictions(self, y_true: np.ndarray, y_pred: np.ndarray,
                        model_name: str = "Model", save_path: Optional[str] = None):
        """
        Plot predictions vs actual values
        
        Args:
            y_true: True values
            y_pred: Predicted values
            model_name: Name of the model
            save_path: Path to save the plot
        """
        plt.figure(figsize=(12, 5))
        
        # Scatter plot
        plt.subplot(1, 2, 1)
        plt.scatter(y_true, y_pred, alpha=0.6)
        plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'{model_name}: Actual vs Predicted')
        plt.grid(True, alpha=0.3)
        
        # Time series plot
        plt.subplot(1, 2, 2)
        plt.plot(y_true.values, label='Actual', alpha=0.7)
        plt.plot(y_pred, label='Predicted', alpha=0.7)
        plt.xlabel('Time')
        plt.ylabel('Solar Irradiance')
        plt.title(f'{model_name}: Time Series Comparison')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def plot_feature_importance(self, coefficients: pd.DataFrame, 
                               top_n: int = 15, save_path: Optional[str] = None):
        """
        Plot feature importance based on coefficients
        
        Args:
            coefficients: DataFrame with feature coefficients
            top_n: Number of top features to show
            save_path: Path to save the plot
        """
        plt.figure(figsize=(10, 8))
        
        # Get top features
        top_features = coefficients.head(top_n)
        
        # Create horizontal bar plot
        colors = ['red' if x < 0 else 'blue' for x in top_features['coefficient']]
        plt.barh(range(len(top_features)), top_features['coefficient'], color=colors, alpha=0.7)
        plt.yticks(range(len(top_features)), top_features['feature'])
        plt.xlabel('Coefficient Value')
        plt.title(f'Top {top_n} Feature Coefficients')
        plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
        plt.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()


def train_solar_power_model(X: pd.DataFrame, y: pd.Series, 
                          learning_rate: float = 0.01, max_iter: int = 1000,
                          regularization: str = 'l2', alpha: float = 0.1) -> Dict[str, Any]:
    """
    Main function to train solar power prediction model
    
    Args:
        X: Feature matrix
        y: Target vector
        learning_rate: Learning rate for gradient descent
        max_iter: Maximum number of iterations
        regularization: Type of regularization ('none', 'l1', 'l2')
        alpha: Regularization strength
        
    Returns:
        Dictionary with trained models and results
    """
    # Initialize trainer
    trainer = SolarPowerModelTrainer()
    
    # Prepare data
    X_train, X_test, y_train, y_test = trainer.prepare_data(X, y)
    
    # Train custom model with user-specified parameters
    custom_results = trainer.train_custom_model(
        X_train, y_train,
        learning_rate=learning_rate,
        n_iterations=max_iter,
        regularization=regularization,
        alpha=alpha
    )
    
    # Train sklearn model
    sklearn_results = trainer.train_sklearn_model(X_train, y_train)
    
    # Evaluate models
    custom_metrics = trainer.evaluate_model(
        custom_results['model'], X_test, y_test, "Custom MLR"
    )
    
    sklearn_metrics = trainer.evaluate_model(
        sklearn_results['model'], X_test, y_test, sklearn_results['model_name']
    )
    
    # Plot results
    trainer.plot_training_history(custom_results['cost_history'])
    trainer.plot_feature_importance(custom_results['coefficients'])
    
    # Plot predictions for best model
    if custom_metrics['R²'] > sklearn_metrics['R²']:
        y_pred = custom_results['model'].predict(X_test)
        trainer.plot_predictions(y_test, y_pred, "Custom MLR")
        best_model = custom_results['model']
        coefficients = custom_results['coefficients']
    else:
        y_pred = sklearn_results['model'].predict(X_test)
        trainer.plot_predictions(y_test, y_pred, sklearn_results['model_name'])
        best_model = sklearn_results['model']
        # For sklearn models, create coefficients from model coefficients
        if hasattr(sklearn_results['model'], 'coef_'):
            coef_df = pd.DataFrame({
                'feature': trainer.feature_names,
                'coefficient': sklearn_results['model'].coef_
            }).sort_values('coefficient', key=abs, ascending=False)
        else:
            coef_df = pd.DataFrame({
                'feature': ['No coefficients available'],
                'coefficient': [0.0]
            })
        coefficients = coef_df
    
    results = {
        'trainer': trainer,
        'custom_results': custom_results,
        'sklearn_results': sklearn_results,
        'custom_metrics': custom_metrics,
        'sklearn_metrics': sklearn_metrics,
        'best_model': best_model,
        'coefficients': coefficients  # Always include coefficients
    }
    
    return results


if __name__ == "__main__":
    # Example usage
    from feature_engineering import SolarFeatureEngineer
    from data_collection import collect_training_data
    
    try:
        # Collect and prepare data
        print("Collecting and preparing data...")
        data = collect_training_data()
        
        engineer = SolarFeatureEngineer()
        features_df = engineer.engineer_features(data)
        X, y = engineer.prepare_features_and_target(features_df)
        
        # Train model
        print("\nTraining solar power prediction model...")
        results = train_solar_power_model(X, y)
        
        print("\n✓ Model training completed successfully!")
        
    except Exception as e:
        print(f"Error during model training: {e}")
