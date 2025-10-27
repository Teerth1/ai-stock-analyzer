# Feature Engineering for Stock Price Prediction
# Calculates technical indicators to use as ML features

# ============================================================================
# TECHNICAL INDICATORS
# ============================================================================

def calculate_rsi(prices, period=14):
    """
    Calculate RSI (Relative Strength Index).

    RSI measures momentum - shows if stock is overbought (>70) or oversold (<30).

    Formula:
    1. Calculate price changes (gains and losses)
    2. Average gain over period / Average loss over period = RS
    3. RSI = 100 - (100 / (1 + RS))

    Parameters:
    - prices: Series of closing prices
    - period: Look-back period (default: 14 days)

    Returns:
    - Series with RSI values (0-100)
    """

    # Calculate price changes
    delta = prices.diff()

    # Separate gains and losses
    gains = delta.where(delta > 0, 0)  # Keep gains, zero out losses
    losses = -delta.where(delta < 0, 0)  # Keep losses (as positive), zero out gains

    # Calculate average gains and losses
    avg_gains = gains.rolling(window=period, min_periods=period).mean()
    avg_losses = losses.rolling(window=period, min_periods=period).mean()

    # Calculate RS (Relative Strength)
    rs = avg_gains / avg_losses

    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))

    return rsi


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence).

    MACD shows trend direction and momentum.

    Formula:
    1. MACD Line = 12-day EMA - 26-day EMA
    2. Signal Line = 9-day EMA of MACD Line
    3. Histogram = MACD Line - Signal Line

    Parameters:
    - prices: Series of closing prices
    - fast: Fast EMA period (default: 12)
    - slow: Slow EMA period (default: 26)
    - signal: Signal line period (default: 9)

    Returns:
    - Dictionary with 'macd', 'signal', and 'histogram'
    """

    # Calculate EMAs (Exponential Moving Averages)
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()

    # MACD line
    macd_line = ema_fast - ema_slow

    # Signal line (EMA of MACD)
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()

    # Histogram
    histogram = macd_line - signal_line

    return {
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    }


def calculate_bollinger_bands(prices, period=20, std_dev=2):
    """
    Calculate Bollinger Bands.

    Bollinger Bands show volatility and relative price levels.

    Formula:
    1. Middle Band = 20-day SMA
    2. Upper Band = Middle Band + (2 � Standard Deviation)
    3. Lower Band = Middle Band - (2 � Standard Deviation)

    Parameters:
    - prices: Series of closing prices
    - period: Look-back period (default: 20)
    - std_dev: Number of standard deviations (default: 2)

    Returns:
    - Dictionary with 'upper', 'middle', 'lower', and 'bandwidth'
    """

    # Middle band (Simple Moving Average)
    middle_band = prices.rolling(window=period).mean()

    # Standard deviation
    std = prices.rolling(window=period).std()

    # Upper and lower bands
    upper_band = middle_band + (std_dev * std)
    lower_band = middle_band - (std_dev * std)

    # Bandwidth (measure of volatility)
    bandwidth = (upper_band - lower_band) / middle_band

    # Calculate position within bands (0 = at lower band, 1 = at upper band)
    bb_position = (prices - lower_band) / (upper_band - lower_band)

    return {
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band,
        'bandwidth': bandwidth,
        'position': bb_position
    }


def calculate_moving_averages(prices, windows=[10, 50, 200]):
    """
    Calculate multiple moving averages.

    Moving averages smooth price data to show trends.

    Parameters:
    - prices: Series of closing prices
    - windows: List of periods (default: [10, 50, 200] days)

    Returns:
    - Dictionary with MA values for each window
    """

    mas = {}
    for window in windows:
        mas[f'ma_{window}'] = prices.rolling(window=window).mean()

    return mas


def calculate_price_momentum(prices, period=10):
    """
    Calculate price momentum (rate of change).

    Momentum = ((Current Price - Price N days ago) / Price N days ago) * 100

    Parameters:
    - prices: Series of closing prices
    - period: Look-back period (default: 10)

    Returns:
    - Series with momentum percentages
    """

    momentum = ((prices - prices.shift(period)) / prices.shift(period)) * 100
    return momentum


def calculate_volatility(returns, period=20):
    """
    Calculate rolling volatility (standard deviation of returns).

    Parameters:
    - returns: Series of returns
    - period: Look-back period (default: 20)

    Returns:
    - Series with volatility values
    """

    volatility = returns.rolling(window=period).std()
    return volatility


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def add_technical_indicators(df):
    """
    Add all technical indicators to the dataframe.

    Parameters:
    - df: DataFrame with stock data (must have 'Close' column)

    Returns:
    - DataFrame with added technical indicator columns
    """

    print("Calculating technical indicators...")

    # Make a copy to avoid modifying original
    df = df.copy()

    # RSI
    df['RSI'] = calculate_rsi(df['Close'])

    # MACD
    macd = calculate_macd(df['Close'])
    df['MACD'] = macd['macd']
    df['MACD_Signal'] = macd['signal']
    df['MACD_Hist'] = macd['histogram']

    # Bollinger Bands
    bb = calculate_bollinger_bands(df['Close'])
    df['BB_Upper'] = bb['upper']
    df['BB_Middle'] = bb['middle']
    df['BB_Lower'] = bb['lower']
    df['BB_Bandwidth'] = bb['bandwidth']
    df['BB_Position'] = bb['position']

    # Moving Averages
    mas = calculate_moving_averages(df['Close'], windows=[10, 50])
    df['MA_10'] = mas['ma_10']
    df['MA_50'] = mas['ma_50']

    # Price vs Moving Averages (percentage difference)
    df['Price_vs_MA10'] = ((df['Close'] - df['MA_10']) / df['MA_10']) * 100
    df['Price_vs_MA50'] = ((df['Close'] - df['MA_50']) / df['MA_50']) * 100

    # Momentum
    df['Momentum_10'] = calculate_price_momentum(df['Close'], period=10)

    # Volatility
    if 'Return' in df.columns:
        df['Volatility'] = calculate_volatility(df['Return'], period=20)

    # Volume changes
    df['Volume_Change'] = df['Volume'].pct_change() * 100

    print(f"Added {len([col for col in df.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Future_Return']])} technical indicators")

    return df


def create_ml_features(df, sentiment_score=0.0):
    """
    Create feature set for ML model.

    Parameters:
    - df: DataFrame with technical indicators
    - sentiment_score: Sentiment score to add as a feature (default: 0.0)

    Returns:
    - DataFrame with just the features needed for ML
    """

    # Select feature columns
    feature_columns = [
        'RSI',
        'MACD',
        'MACD_Hist',
        'BB_Position',
        'BB_Bandwidth',
        'Price_vs_MA10',
        'Price_vs_MA50',
        'Momentum_10',
        'Volatility',
        'Volume_Change',
        'Return'  # Yesterday's return
    ]

    # Add sentiment score as a feature
    df = df.copy()
    df['Sentiment'] = sentiment_score
    feature_columns.append('Sentiment')

    # Create features dataframe
    features = df[feature_columns].copy()

    # Drop rows with NaN values (from indicator calculations)
    features = features.dropna()

    return features


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    # Test feature engineering
    from data_collector import get_stock_data

    print("="*70)
    print("TESTING FEATURE ENGINEERING")
    print("="*70)

    # Get stock data
    df = get_stock_data('AAPL')

    if df is not None:
        print(f"\nOriginal data shape: {df.shape}")
        print(f"Columns: {list(df.columns)}")

        # Add technical indicators
        df_with_features = add_technical_indicators(df)

        print(f"\nData with features shape: {df_with_features.shape}")
        print(f"\nNew feature columns:")
        new_cols = [col for col in df_with_features.columns if col not in ['Open', 'High', 'Low', 'Close', 'Volume', 'Return', 'Future_Return']]
        for col in new_cols:
            print(f"  - {col}")

        # Show sample data
        print("\nSample data with features (last 5 days):")
        print(df_with_features[['Close', 'RSI', 'MACD', 'BB_Position', 'MA_10', 'Momentum_10']].tail())

        # Create ML features
        ml_features = create_ml_features(df_with_features, sentiment_score=0.5)
        print(f"\nML features shape: {ml_features.shape}")
        print("\nML feature columns:")
        print(list(ml_features.columns))