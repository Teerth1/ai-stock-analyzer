# AI Stock Sentiment & Price Analyzer

A stock analysis tool that uses **FinBERT (transformer-based NLP)** for sentiment analysis and **Random Forest ML** for price movement prediction.

## Features

- **AI Sentiment Analysis**: Uses FinBERT (BERT model trained on 1.8M financial texts) to analyze news sentiment
- **ML Price Prediction**: Random Forest regression model predicts future price movements
- **Technical Indicators**: Calculates RSI, MACD, Bollinger Bands, and Moving Averages
- **Intelligent Recommendations**: Combines multiple signals for BUY/SELL/HOLD decisions
- **No Trading Risk**: Analysis tool only - doesn't execute actual trades

## Technologies Used

- **Python 3.14**
- **Transformers (Hugging Face)**: Pre-trained FinBERT model for NLP
- **PyTorch**: Deep learning framework for running FinBERT
- **scikit-learn**: Random Forest ML model
- **pandas & numpy**: Data manipulation and analysis
- **yfinance**: Historical stock data from Yahoo Finance
- **NewsAPI**: Financial news data source (optional)

## Project Structure

```
stockAnalyzer/
├── stock_analyzer.py        # Main analysis tool
├── demo.py                  # Quick demo script
├── sentiment_analyzer.py    # FinBERT sentiment analysis
├── price_predictor.py       # ML price prediction model
├── feature_engineer.py      # Technical indicators
├── data_collector.py        # Stock data fetching
├── requirements.txt         # Python dependencies
└── README.md               # This file
```

## Installation

1. Install required libraries:
```bash
pip install -r requirements.txt
```

2. (Optional) Get NewsAPI key from https://newsapi.org/ for sentiment analysis

## Quick Start

### Run Demo
```bash
python demo.py
```

This will analyze AAPL, TSLA, and MSFT using technical indicators and show you how the system works.

### Analyze a Specific Stock
```bash
python stock_analyzer.py AAPL
```

Replace `AAPL` with any stock ticker (TSLA, MSFT, GOOGL, etc.)

## How It Works

```
Input: Stock Symbol (e.g., "AAPL")
  │
  ├─→ [1] Fetch Recent News → FinBERT AI → Sentiment Score (-1 to +1)
  │
  ├─→ [2] Fetch Price Data → Technical Indicators
  │                          ├─ RSI (overbought/oversold)
  │                          ├─ MACD (trend direction)
  │                          ├─ Bollinger Bands (volatility)
  │                          └─ Moving Averages (trend)
  │
  ├─→ [3] Combine Features → Random Forest ML → Price Prediction (% change)
  │
  └─→ [4] Multi-Signal Analysis → Recommendation (BUY/SELL/HOLD)
```

### Example Output

```
======================================================================
AI STOCK ANALYSIS: AAPL
======================================================================

[1/4] Fetching stock data...
  Current Price: $252.29

[2/4] Analyzing news sentiment with FinBERT...
  Sentiment Score: +0.650 (BULLISH)
  Based on 8 articles

[3/4] Calculating technical indicators...
  RSI: 46.47 (NEUTRAL)
  MACD: +1.234
  Price vs 10-day MA: +3.21%
  Bollinger Band Position: 0.75

[4/4] Making ML price prediction...
  Predicted Price Change: +1.6% (UP)

======================================================================
RECOMMENDATION
======================================================================

BUY: Positive news sentiment, ML predicts +1.60% increase
Confidence: High

======================================================================
```

## Configuration

Edit `stock_analyzer.py` to customize:

- `NEWS_API_KEY`: Your NewsAPI key (line 12)
- `use_ml`: Enable/disable ML prediction (default: True)
- Stock symbols to analyze

## Features Explained

### 1. FinBERT Sentiment Analysis
- Transformer-based AI model
- Trained specifically on financial news
- Much more accurate than rule-based approaches
- Understands financial context (e.g., "debt decreased" = positive)

### 2. Technical Indicators
- **RSI**: Relative Strength Index (overbought/oversold)
- **MACD**: Moving Average Convergence Divergence (trend)
- **Bollinger Bands**: Volatility and price levels
- **Moving Averages**: Trend direction (10-day, 50-day)

### 3. ML Price Prediction
- Random Forest regression model
- Learns from historical patterns
- Combines sentiment + technical indicators
- Predicts next-day price movement percentage

### 4. Multi-Signal Recommendation
- Combines all signals (sentiment, technical, ML)
- Provides BUY/SELL/HOLD recommendation
- Includes confidence level (High/Medium/Low)

## Training ML Models

First time analyzing a stock, the system will automatically:
1. Download 2 years of historical data
2. Calculate all technical indicators
3. Train a Random Forest model
4. Save the model for future use

Models are saved in `models/` directory and reused automatically.

## Resume Bullet Points

```
AI Stock Sentiment & Price Analyzer | Python, ML, NLP

• Built stock analysis system combining transformer-based NLP (FinBERT) for
  sentiment analysis and Random Forest regression for price prediction,
  integrating news sentiment with technical indicators (RSI, MACD, Bollinger Bands)

• Implemented end-to-end ML pipeline with feature engineering, model training,
  and multi-signal analysis, achieving X% accuracy on historical stock data

• Integrated NewsAPI and Yahoo Finance APIs for real-time data collection and
  automated analysis combining NLP and time-series predictions

• Technologies: Python, PyTorch, Transformers, scikit-learn, pandas, numpy
```

## What Makes This Project Impressive

✅ **Combines Multiple AI/ML Techniques**: NLP (FinBERT) + ML (Random Forest)
✅ **End-to-End Pipeline**: Data collection → Feature engineering → Training → Prediction
✅ **Production-Ready Code**: Modular design, error handling, model persistence
✅ **Real-World Application**: Actual financial analysis with real data
✅ **Modern Tech Stack**: Transformers, PyTorch, scikit-learn

## Safety & Disclaimer

- **This is an analysis tool only** - it does not execute real trades
- **Not financial advice** - for educational purposes only
- **Historical performance ≠ future results**
- Always do your own research before investing

## Future Enhancements

- [ ] Backtesting framework to test on historical data
- [ ] Performance tracking and visualization
- [ ] More data sources (Twitter, Reddit sentiment)
- [ ] Deep learning models (LSTM for time series)
- [ ] Web interface for easy access

## Author

Built as a machine learning and NLP learning project.

## License

Educational/Personal Use
