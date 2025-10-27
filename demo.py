# Quick Demo of Stock Analyzer
# Run this to see the AI stock analysis in action

from stock_analyzer import analyze_stock

print("""
TPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPW
Q                                                                   Q
Q         AI Stock Sentiment & Price Analyzer - DEMO                Q
Q                                                                   Q
Q  This tool uses:                                                  Q
Q  " FinBERT (AI) for sentiment analysis of financial news         Q
Q  " Random Forest ML for price movement prediction                Q
Q  " Technical indicators (RSI, MACD, Bollinger Bands)             Q
Q                                                                   Q
ZPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPPP]
""")

# Demo stocks to analyze
demo_stocks = ['AAPL', 'TSLA', 'MSFT']

print("\nAnalyzing popular stocks without news sentiment...")
print("(Add your NewsAPI key to stock_analyzer.py for sentiment analysis)\n")

for symbol in demo_stocks:
    print(f"\n{'='*70}")
    print(f"Analyzing {symbol}...")
    print(f"{'='*70}\n")

    # Analyze without news API (just technical + ML)
    results = analyze_stock(
        symbol=symbol,
        news_api_key=None,  # Set this in stock_analyzer.py for sentiment
        use_ml=False  # Set to True after models are trained
    )

    if results:
        print(f"\n Analysis complete for {symbol}")
        print(f"  Current Price: ${results['current_price']:.2f}")
        print(f"  Recommendation: {results['recommendation']['action']}")

    print("\n" + "="*70)

    # Small delay between stocks
    import time
    time.sleep(1)

print("\n\n" + "="*70)
print("DEMO COMPLETE")
print("="*70)
print("\nTo use the full features:")
print("1. Add your NewsAPI key to stock_analyzer.py (line 12)")
print("2. Train ML models by running: python stock_analyzer.py AAPL")
print("3. Then you can analyze any stock with full AI capabilities!")
print("\nUsage: python stock_analyzer.py [STOCK_SYMBOL]")
print("Example: python stock_analyzer.py TSLA")
print("="*70)
