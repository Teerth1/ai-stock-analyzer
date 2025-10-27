# ML Price Prediction Model
# Uses Random Forest to predict stock price movements

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle
import os
import numpy as np

# ============================================================================
# ML MODEL CLASS
# ============================================================================

class PricePredictor:
    """
    Random Forest model to predict stock price movement percentages.

    How it works:
    1. Train on historical data (features � future returns)
    2. Learn patterns (e.g., "when RSI > 70 and MACD negative, price usually drops")
    3. Predict future price changes based on current features
    """

    def __init__(self, n_estimators=100, max_depth=20, random_state=42):
        """
        Initialize the Random Forest model.

        Parameters:
        - n_estimators: Number of decision trees in the forest (default: 100)
        - max_depth: Maximum depth of each tree (default: 20)
        - random_state: Random seed for reproducibility (default: 42)
        """

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=5,
            min_samples_leaf=2,
            random_state=random_state,
            n_jobs=-1  # Use all CPU cores
        )

        self.is_trained = False
        self.feature_names = None
        self.training_score = None


    def train(self, X, y, test_size=0.2):
        """
        Train the Random Forest model on historical data.

        Parameters:
        - X: Features (DataFrame with technical indicators, sentiment, etc.)
        - y: Target (Series with future returns to predict)
        - test_size: Portion of data to use for testing (default: 0.2 = 20%)

        Returns:
        - Dictionary with training metrics
        """

        print("Training Random Forest model...")

        # Store feature names
        self.feature_names = list(X.columns)

        # Split data into training and testing sets
        # test_size=0.2 means 80% train, 20% test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, shuffle=False
        )

        print(f"Training set: {len(X_train)} samples")
        print(f"Test set: {len(X_test)} samples")

        # Train the model
        # The model learns patterns from X_train � y_train
        self.model.fit(X_train, y_train)

        # Make predictions on test set
        y_pred = self.model.predict(X_test)

        # Calculate performance metrics
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(np.mean((y_test - y_pred) ** 2))  # Calculate RMSE manually
        r2 = r2_score(y_test, y_pred)

        # Store metrics
        metrics = {
            'mae': mae,
            'rmse': rmse,
            'r2': r2,
            'train_samples': len(X_train),
            'test_samples': len(X_test)
        }

        self.training_score = metrics
        self.is_trained = True

        print(f"\nModel Training Complete!")
        print(f"  MAE (Mean Absolute Error): {mae:.4f}%")
        print(f"  RMSE (Root Mean Squared Error): {rmse:.4f}%")
        print(f"  R2 Score: {r2:.4f}")

        # R2 interpretation
        if r2 > 0.5:
            print(f"  Model explains {r2*100:.1f}% of price variance (Good!)")
        elif r2 > 0.2:
            print(f"  Model explains {r2*100:.1f}% of price variance (Fair)")
        else:
            print(f"  Model explains {r2*100:.1f}% of price variance (Weak)")

        return metrics


    def predict(self, X):
        """
        Predict future price change percentage.

        Parameters:
        - X: Features (single row or DataFrame)

        Returns:
        - Predicted price change percentage
        """

        if not self.is_trained:
            raise ValueError("Model not trained yet! Call train() first.")

        # Make prediction
        prediction = self.model.predict(X)

        # Return single value if single prediction
        if len(prediction) == 1:
            return prediction[0]

        return prediction


    def get_feature_importance(self):
        """
        Get the importance of each feature in making predictions.

        Returns:
        - Dictionary mapping feature names to importance scores
        """

        if not self.is_trained:
            return None

        # Get feature importances from the model
        importances = self.model.feature_importances_

        # Create dictionary of feature � importance
        feature_importance = dict(zip(self.feature_names, importances))

        # Sort by importance (highest first)
        feature_importance = dict(sorted(feature_importance.items(),
                                        key=lambda x: x[1],
                                        reverse=True))

        return feature_importance


    def print_feature_importance(self):
        """Print feature importances in a readable format."""

        importance = self.get_feature_importance()

        if importance is None:
            print("Model not trained yet!")
            return

        print("\nFeature Importance (most important first):")
        print("="*50)

        for feature, score in importance.items():
            bar = "#" * int(score * 100)
            print(f"{feature:20s} {score:6.4f}  {bar}")


    def save(self, filepath='models/price_predictor.pkl'):
        """
        Save the trained model to a file.

        Parameters:
        - filepath: Where to save the model (default: 'models/price_predictor.pkl')
        """

        if not self.is_trained:
            print("Warning: Saving untrained model")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)

        # Save model using pickle
        with open(filepath, 'wb') as f:
            pickle.dump(self, f)

        print(f"Model saved to {filepath}")


    @staticmethod
    def load(filepath='models/price_predictor.pkl'):
        """
        Load a trained model from a file.

        Parameters:
        - filepath: Where to load the model from

        Returns:
        - Loaded PricePredictor instance
        """

        if not os.path.exists(filepath):
            print(f"Error: Model file not found at {filepath}")
            return None

        with open(filepath, 'rb') as f:
            model = pickle.load(f)

        print(f"Model loaded from {filepath}")
        return model


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def train_model_for_stock(symbol, sentiment_score=0.0):
    """
    Train a price prediction model for a specific stock.

    Parameters:
    - symbol: Stock ticker
    - sentiment_score: Average sentiment to use for historical data

    Returns:
    - Trained PricePredictor instance
    """

    from data_collector import get_stock_data
    from feature_engineer import add_technical_indicators, create_ml_features

    print(f"\n{'='*70}")
    print(f"Training ML Model for {symbol}")
    print(f"{'='*70}")

    # Step 1: Get historical data
    df = get_stock_data(symbol)

    if df is None or len(df) < 100:
        print(f"Not enough data for {symbol}")
        return None

    # Step 2: Add technical indicators
    df = add_technical_indicators(df)

    # Step 3: Create features
    features = create_ml_features(df, sentiment_score=sentiment_score)

    # Step 4: Prepare X (features) and y (target)
    # We need to align features with future returns
    # Get corresponding future returns for each feature row
    y = df.loc[features.index, 'Future_Return']

    # Remove any NaN values
    mask = ~y.isna()
    X = features[mask]
    y = y[mask]

    print(f"\nDataset prepared:")
    print(f"  Total samples: {len(X)}")
    print(f"  Features: {len(X.columns)}")

    # Step 5: Train model
    predictor = PricePredictor()
    predictor.train(X, y)

    # Step 6: Show feature importance
    predictor.print_feature_importance()

    return predictor


def predict_next_move(symbol, sentiment_score, predictor=None):
    """
    Predict next day's price movement for a stock.

    Parameters:
    - symbol: Stock ticker
    - sentiment_score: Current sentiment score
    - predictor: Trained model (if None, will try to load saved model)

    Returns:
    - Predicted price change percentage
    """

    from data_collector import get_latest_prices
    from feature_engineer import add_technical_indicators, create_ml_features

    # Load model if not provided
    if predictor is None:
        predictor = PricePredictor.load(f'models/{symbol}_predictor.pkl')
        if predictor is None:
            print(f"No trained model found for {symbol}")
            return None

    # Get recent data
    df = get_latest_prices(symbol, days=60)  # Need enough for indicators

    if df is None or len(df) < 50:
        print(f"Not enough recent data for {symbol}")
        return None

    # Add indicators
    df = add_technical_indicators(df)

    # Create features for the most recent day
    features = create_ml_features(df, sentiment_score=sentiment_score)

    if len(features) == 0:
        print("Error: Could not create features")
        return None

    # Get the last row (most recent data)
    latest_features = features.iloc[[-1]]

    # Make prediction
    prediction = predictor.predict(latest_features)

    return prediction


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    # Test the price predictor

    # Train a model for Apple
    predictor = train_model_for_stock('AAPL', sentiment_score=0.3)

    if predictor:
        # Save the model
        predictor.save('models/AAPL_predictor.pkl')

        # Test prediction
        print(f"\n{'='*70}")
        print("Testing Prediction")
        print(f"{'='*70}")

        prediction = predict_next_move('AAPL', sentiment_score=0.5, predictor=predictor)

        if prediction:
            print(f"\nPredicted price change for tomorrow: {prediction:+.2f}%")

            if prediction > 0.5:
                print("Signal: BUY (price expected to rise)")
            elif prediction < -0.5:
                print("Signal: SELL (price expected to fall)")
            else:
                print("Signal: HOLD (small movement expected)")