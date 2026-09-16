# BTC 5M Predictor — Render Free

Render settings:
- Runtime: Python 3
- Build: `pip install -r requirements.txt`
- Start: `uvicorn app:app --host 0.0.0.0 --port $PORT`
- Plan: Free

The app uses public Polymarket market-data endpoints and Binance BTCUSDT 1-minute candles.
It does not place orders. Browser history is stored in Safari localStorage.

Render Free can spin down after 15 minutes without inbound traffic, so this is an interactive
free MVP rather than guaranteed 24/7 background collection.
