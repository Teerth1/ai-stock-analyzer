# AI Stock Sentiment & Price Analyzer
# Analyzes stocks using FinBERT sentiment analysis and ML price prediction

from sentiment_analyzer import FinBertAnalyzer, analyze_sentiment_finbert
from data_collector import get_latest_prices
from feature_engineer import add_technical_indicators
from price_predictor import PricePredictor, train_model_for_stock, predict_next_move
import requests
from datetime import datetime, timedelta

# Configuration
NEWS_API_KEY = '874bce09c7764ba594e67966e4392c68'  # Get free key at https://newsapi.org/




def fetch_news(symbol, api_key, days_back=2):
    """Fetch recent news articles about a stock."""
    date_from = (datetime.now() - timedelta(days=days_back)).strftime('%Y-%m-%d')
    url = f'https://newsapi.org/v2/everything?q={symbol}&from={date_from}&sortBy=publishedAt&language=en&apiKey={api_key}'

    try:
        response = requests.get(url)
        data = response.json()
        if data['status'] == 'ok':
            return data['articles']
        else:
            print(f"Error fetching news: {data.get('message', 'Unknown error')}")
            return []
    except Exception as e:
        print(f"Exception fetching news: {e}")
        return []


def get_stock_sentiment(symbol, api_key, finbert_analyzer):
    """Get overall sentiment for a stock based on recent news."""
    articles = fetch_news(symbol, api_key)

    if not articles:
        return 0, 0

    sentiments = []

    for article in articles[:15]:
        title = article.get('title', '')
        description = article.get('description', '')
        text = f"{title}. {description}"

        if len(text) < 20:
            continue

        sentiment = analyze_sentiment_finbert(text, finbert_analyzer)
        sentiments.append(sentiment)

    if sentiments:
        avg_sentiment = sum(sentiments) / len(sentiments)
        return avg_sentiment, len(sentiments)
    else:
        return 0, 0



# STOCK ANALYSIS


def analyze_stock(symbol, news_api_key=None, use_ml=True):
    """
    Comprehensive stock analysis using AI and ML.

    Parameters:
    - symbol: Stock ticker (e.g., 'AAPL')
    - news_api_key: NewsAPI key for sentiment analysis (optional)
    - use_ml: Whether to use ML price prediction (default: True)

    Returns:
    - Dictionary with analysis results
    """

    print("="*70)
    print(f"AI STOCK ANALYSIS: {symbol}")
    print("="*70)

    results = {
        'symbol': symbol,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

    # Step 1: Get current stock data
    print("\n[1/4] Fetching stock data...")
    df = get_latest_prices(symbol, days=60)

    if df is None or len(df) < 50:
        print(f"Error: Not enough data for {symbol}")
        return None

    current_price = df['Close'].iloc[-1]
    results['current_price'] = current_price
    print(f"  Current Price: ${current_price:.2f}")

    # Step 2: Sentiment Analysis
    sentiment_score = 0
    if news_api_key and news_api_key != 'your_newsapi_key_here':
        print("\n[2/4] Analyzing news sentiment with FinBERT...")
        finbert = FinBertAnalyzer()
        sentiment_score, article_count = get_stock_sentiment(symbol, news_api_key, finbert)

        results['sentiment_score'] = sentiment_score
        results['articles_analyzed'] = article_count

        if article_count > 0:
            if sentiment_score > 0.15:
                sentiment_label = "BULLISH"
            elif sentiment_score < -0.15:
                sentiment_label = "BEARISH"
            else:
                sentiment_label = "NEUTRAL"

            print(f"  Sentiment Score: {sentiment_score:+.3f} ({sentiment_label})")
            print(f"  Based on {article_count} articles")
        else:
            print("  No recent news found")
    else:
        print("\n[2/4] Skipping sentiment analysis (no API key)")
        results['sentiment_score'] = 0
        results['articles_analyzed'] = 0

    # Step 3: Technical Indicators
    print("\n[3/4] Calculating technical indicators...")
    df_with_indicators = add_technical_indicators(df)

    latest = df_with_indicators.iloc[-1]

    results['technical_indicators'] = {
        'RSI': latest['RSI'],
        'MACD': latest['MACD'],
        'MACD_Signal': latest['MACD_Signal'],
        'BB_Position': latest['BB_Position'],
        'MA_10': latest['MA_10'],
        'MA_50': latest['MA_50'],
        'Price_vs_MA10': latest['Price_vs_MA10'],
        'Price_vs_MA50': latest['Price_vs_MA50']
    }

    # Interpret RSI
    rsi = latest['RSI']
    if rsi > 70:
        rsi_signal = "OVERBOUGHT"
    elif rsi < 30:
        rsi_signal = "OVERSOLD"
    else:
        rsi_signal = "NEUTRAL"

    print(f"  RSI: {rsi:.2f} ({rsi_signal})")
    print(f"  MACD: {latest['MACD']:.3f}")
    print(f"  Price vs 10-day MA: {latest['Price_vs_MA10']:+.2f}%")
    print(f"  Bollinger Band Position: {latest['BB_Position']:.2f}")

    # Step 4: ML Price Prediction
    if use_ml:
        print("\n[4/4] Making ML price prediction...")
        try:
            # Try to load existing model
            predictor = PricePredictor.load(f'models/{symbol}_predictor.pkl')

            if predictor is None:
                print("  No trained model found. Training new model...")
                predictor = train_model_for_stock(symbol, sentiment_score=sentiment_score)
                if predictor:
                    predictor.save(f'models/{symbol}_predictor.pkl')

            if predictor:
                prediction = predict_next_move(symbol, sentiment_score, predictor)

                if prediction is not None:
                    results['ml_prediction'] = prediction

                    if prediction > 0.5:
                        prediction_label = "UP"
                    elif prediction < -0.5:
                        prediction_label = "DOWN"
                    else:
                        prediction_label = "FLAT"

                    print(f"  Predicted Price Change: {prediction:+.2f}% ({prediction_label})")
                else:
                    results['ml_prediction'] = None
            else:
                results['ml_prediction'] = None

        except Exception as e:
            print(f"  Error in ML prediction: {e}")
            results['ml_prediction'] = None
    else:
        print("\n[4/4] Skipping ML prediction")
        results['ml_prediction'] = None

    # Final Recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    recommendation = make_recommendation(results)
    results['recommendation'] = recommendation

    print(f"\n{recommendation['action']}: {recommendation['reason']}")
    print(f"Confidence: {recommendation['confidence']}")

    print("\n" + "="*70)

    return results


def make_recommendation(results):
    """
    Make a trading recommendation based on all signals.

    Returns:
    - Dictionary with action, reason, and confidence
    """

    signals = []

    # Sentiment signal
    sentiment = results.get('sentiment_score', 0)
    if sentiment > 0.15:
        signals.append(('BUY', 'Positive news sentiment'))
    elif sentiment < -0.15:
        signals.append(('SELL', 'Negative news sentiment'))

    # RSI signal
    rsi = results.get('technical_indicators', {}).get('RSI', 50)
    if rsi > 70:
        signals.append(('SELL', 'Stock is overbought (RSI > 70)'))
    elif rsi < 30:
        signals.append(('BUY', 'Stock is oversold (RSI < 30)'))

    # ML prediction signal
    ml_pred = results.get('ml_prediction')
    if ml_pred is not None:
        if ml_pred > 0.5:
            signals.append(('BUY', f'ML predicts +{ml_pred:.2f}% increase'))
        elif ml_pred < -0.5:
            signals.append(('SELL', f'ML predicts {ml_pred:.2f}% decrease'))

    # Count signals
    buy_signals = sum(1 for action, _ in signals if action == 'BUY')
    sell_signals = sum(1 for action, _ in signals if action == 'SELL')

    # Make decision
    if buy_signals > sell_signals:
        action = "BUY"
        reasons = [reason for act, reason in signals if act == 'BUY']
        confidence = "High" if buy_signals >= 2 else "Medium"
    elif sell_signals > buy_signals:
        action = "SELL"
        reasons = [reason for act, reason in signals if act == 'SELL']
        confidence = "High" if sell_signals >= 2 else "Medium"
    else:
        action = "HOLD"
        reasons = ["Mixed signals - waiting for clearer trend"]
        confidence = "Low"

    return {
        'action': action,
        'reason': ', '.join(reasons),
        'confidence': confidence
    }



# MAIN FUNCTION


if __name__ == "__main__":
    import sys

    # Get stock symbol from command line or use default
    if len(sys.argv) > 1:
        symbol = sys.argv[1].upper()
    else:
        symbol = 'GOOGL'  # Default to Google

    # Analyze the stock
    results = analyze_stock(
        symbol=symbol,
        news_api_key=NEWS_API_KEY,  # Set your key at top of file
        use_ml=True
    )

    # You can access results programmatically
    if results:
        print(f"\nAnalysis completed for {results['symbol']}")
