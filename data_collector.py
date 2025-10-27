# Historical Stock Data Collector
# Fetches historical price data for stocks using Yahoo Finance API

import yfinance as yf

# ============================================================================
# DATA COLLECTION FUNCTIONS
# ============================================================================

def fetch_historical_data(symbol, start_date=None, end_date=None, period='1y'):
    """
    Fetch historical stock data from Yahoo Finance.

    Parameters:
    - symbol: Stock ticker (e.g., 'AAPL')
    - start_date: Start date as string 'YYYY-MM-DD' (optional)
    - end_date: End date as string 'YYYY-MM-DD' (optional)
    - period: Time period if start/end not specified (default: '1y' = 1 year)
              Options: '1d', '5d', '1mo', '3mo', '6mo', '1y', '2y', '5y', 'max'

    Returns:
    - DataFrame with columns: Open, High, Low, Close, Volume, and calculated returns
    """

    print(f"Fetching historical data for {symbol}...")

    try:
        # Create ticker object
        ticker = yf.Ticker(symbol)

        # Fetch data
        if start_date and end_date:
            # Use specific date range
            df = ticker.history(start=start_date, end=end_date)
        else:
            # Use period
            df = ticker.history(period=period)

        if df.empty:
            print(f"No data found for {symbol}")
            return None

        # Calculate daily returns (percentage change)
        # Return = (Today's Close - Yesterday's Close) / Yesterday's Close * 100
        df['Return'] = df['Close'].pct_change() * 100

        # Calculate future return (what we want to predict)
        # Shift returns back 1 day - this is what we'll predict
        df['Future_Return'] = df['Return'].shift(-1)

        # Remove rows with NaN values in Future_Return (can't predict for last day)
        df = df[:-1]  # Remove last row

        print(f" Fetched {len(df)} days of data for {symbol}")
        print(f"  Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")

        return df

    except Exception as e:
        print(f"Error fetching data for {symbol}: {e}")
        return None


def get_latest_prices(symbol, days=10):
    """
    Get the most recent price data for a stock.
    Used for real-time predictions.

    Parameters:
    - symbol: Stock ticker
    - days: Number of recent days to fetch (default: 10)

    Returns:
    - DataFrame with recent price data
    """

    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=f'{days}d')

        if df.empty:
            return None

        # Calculate returns
        df['Return'] = df['Close'].pct_change() * 100

        return df

    except Exception as e:
        print(f"Error fetching latest prices for {symbol}: {e}")
        return None


def get_stock_data(symbol, use_cache=False):
    """
    Get stock data - fetches from Yahoo Finance API.

    Parameters:
    - symbol: Stock ticker
    - use_cache: Whether to use cached data (not implemented yet)

    Returns:
    - DataFrame with historical stock data
    """

    # Fetch data (2 years for training)
    df = fetch_historical_data(symbol, period='2y')
    return df


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def print_data_summary(df):
    """
    Print a summary of the fetched data.

    Parameters:
    - df: DataFrame with stock data
    """

    print("\n" + "="*70)
    print("DATA SUMMARY")
    print("="*70)
    print(f"Total days: {len(df)}")
    print(f"Date range: {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
    print(f"\nPrice statistics:")
    print(f"  Highest Close: ${df['Close'].max():.2f}")
    print(f"  Lowest Close: ${df['Close'].min():.2f}")
    print(f"  Average Close: ${df['Close'].mean():.2f}")
    print(f"\nReturn statistics:")
    print(f"  Average Daily Return: {df['Return'].mean():.3f}%")
    print(f"  Std Dev (Volatility): {df['Return'].std():.3f}%")
    print(f"  Best Day: {df['Return'].max():.3f}%")
    print(f"  Worst Day: {df['Return'].min():.3f}%")
    print("="*70)


# ============================================================================
# TEST CODE
# ============================================================================

if __name__ == "__main__":
    # Test the data collector
    print("="*70)
    print("TESTING DATA COLLECTOR")
    print("="*70)

    # Test with Apple stock
    symbol = 'AAPL'

    # Fetch data
    df = get_stock_data(symbol)

    if df is not None:
        # Print summary
        print_data_summary(df)

        # Show first few rows
        print("\nFirst 5 days:")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].head())

        # Show last few rows
        print("\nLast 5 days:")
        print(df[['Open', 'High', 'Low', 'Close', 'Volume', 'Return']].tail())